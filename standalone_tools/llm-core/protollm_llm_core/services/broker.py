import json
import logging
from typing import Type

import pika
from protollm_sdk.models.job_context_models import PromptModel, ChatCompletionModel, PromptTransactionModel, \
    PromptWrapper, ChatCompletionTransactionModel
from protollm_sdk.object_interface.redis_wrapper import RedisWrapper

from protollm_llm_core.models.base import BaseLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMWrap:

    def __init__(self,
                 llm_model: BaseLLM,
                 redis_host: str,
                 redis_port: str,
                 queue_name: str,
                 rabbit_host: str,
                 rabbit_port: str,
                 rabbit_login: str,
                 rabbit_password: str,
                 redis_prefix: str):
        self.llm = llm_model
        logger.info('Loaded model')

        self.redis_bd = RedisWrapper(redis_host, redis_port)
        self.redis_prefix = redis_prefix
        logger.info('connected to redis')

        self.models = {
            'single_generate': PromptModel,
            'chat_completion': ChatCompletionModel,
        }

        self.queue_name = queue_name
        self.rabbit_host = rabbit_host
        self.rabbit_port = rabbit_port
        self.rabbit_login = rabbit_login
        self.rabbit_password = rabbit_password

    def start_connection(self):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.rabbit_host,
                port=self.rabbit_port,
                virtual_host='/',
                credentials=pika.PlainCredentials(
                    username=self.rabbit_login,
                    password=self.rabbit_password
                )
            )
        )

        channel = connection.channel()
        logger.info('connected to the broker')

        channel.queue_declare(queue=self.queue_name)
        logger.info('A queue has been announced')

        channel.basic_consume(
            on_message_callback=self._callback,
            queue=self.queue_name,
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
        logger.info(f'{self.redis_prefix}:{transaction.prompt.job_id}\n {func_result}')
        self.redis_bd.save_item(f'{self.redis_prefix}:{transaction.prompt.job_id}', {"content": func_result})
        logger.info(f'The response on the task {transaction.prompt.job_id} was written in radish')
