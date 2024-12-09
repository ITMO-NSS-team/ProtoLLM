import json
import logging
from typing import Type

import pika
from protollm_sdk.models.job_context_models import PromptModel, ChatCompletionModel, PromptTransactionModel, \
    PromptWrapper, ChatCompletionTransactionModel
from protollm_sdk.object_interface.redis_wrapper import RedisWrapper

from protollm_llm_core.config import (
    RABBIT_MQ_HOST, RABBIT_MQ_PORT,
    RABBIT_MQ_PASSWORD, RABBIT_MQ_LOGIN,
    REDIS_PREFIX
)
from protollm_llm_core.config import REDIS_HOST, REDIS_PORT, MODEL_PATH, QUEUE_NAME
from protollm_llm_core.models.base import BaseLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLM_wrap:

    def __init__(self, llm_model: Type[BaseLLM]):
        self.llm = llm_model(model_path=MODEL_PATH)
        logger.info('Loaded model')

        self.redis_bd = RedisWrapper(REDIS_HOST, REDIS_PORT)
        logger.info('connected to redis')

        self.models = {
            'single_generate': PromptModel,
            'chat_completion': ChatCompletionModel,
        }

    def start_connection(self):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=RABBIT_MQ_HOST,
                port=RABBIT_MQ_PORT,
                virtual_host='/',
                credentials=pika.PlainCredentials(
                    username=RABBIT_MQ_LOGIN,
                    password=RABBIT_MQ_PASSWORD
                )
            )
        )

        channel = connection.channel()
        logger.info('connected to the broker')

        channel.queue_declare(queue=QUEUE_NAME)
        logger.info('A queue has been announced')

        channel.basic_consume(
            on_message_callback=self._callback,
            queue=QUEUE_NAME,
            auto_ack=True
        )

        channel.start_consuming()
        logger.info('start consuming')

    def _dump_from_body(self, message_body) -> PromptModel | ChatCompletionModel:
        return PromptModel(**message_body['kwargs'])

    def _callback(self, ch, method, properties, body):
        logger.info(json.loads(body))
        prompt_wrapper = PromptWrapper(prompt=json.loads(body)['kwargs'])
        transaction: PromptTransactionModel | ChatCompletionTransactionModel = prompt_wrapper.prompt
        func_result = self.llm(transaction)

        logger.info(f'The llm response on the task {transaction.prompt.job_id} has been generated')
        logger.info(f'{REDIS_PREFIX}:{transaction.prompt.job_id}\n {func_result}')
        self.redis_bd.save_item(f'{REDIS_PREFIX}:{transaction.prompt.job_id}', {"content": func_result})
        logger.info(f'The response on the task {transaction.prompt.job_id} was written in radish')
