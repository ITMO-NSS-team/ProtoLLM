import json
import logging

import pika
from protollm_sdk.models.job_context_models import PromptModel, ChatCompletionModel, PromptTransactionModel, \
    PromptWrapper, ChatCompletionTransactionModel
from protollm_sdk.object_interface.redis_wrapper import RedisWrapper

from protollm_llm_core.models.base import BaseLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMWrap:
    """
    A wrapper for handling interactions with an LLM model, Redis database, and RabbitMQ message broker.

    This class provides a mechanism for consuming messages from RabbitMQ, processing them with a language model,
    and storing the results in Redis.
    """

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
        """
        Initialize the LLMWrap class with the necessary configurations.

        :param llm_model: The language model to use for processing prompts.
        :type llm_model: BaseLLM
        :param redis_host: Hostname for the Redis server.
        :type redis_host: str
        :param redis_port: Port for the Redis server.
        :type redis_port: str
        :param queue_name: Name of the RabbitMQ queue to consume messages from.
        :type queue_name: str
        :param rabbit_host: Hostname for the RabbitMQ server.
        :type rabbit_host: str
        :param rabbit_port: Port for the RabbitMQ server.
        :type rabbit_port: str
        :param rabbit_login: Login for RabbitMQ authentication.
        :type rabbit_login: str
        :param rabbit_password: Password for RabbitMQ authentication.
        :type rabbit_password: str
        :param redis_prefix: Prefix for Redis keys to store results.
        :type redis_prefix: str
        """
        self.llm = llm_model
        logger.info('Loaded model')

        self.redis_bd = RedisWrapper(redis_host, redis_port)
        self.redis_prefix = redis_prefix
        logger.info('Connected to Redis')

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
        """
        Establish a connection to the RabbitMQ broker and start consuming messages from the specified queue.
        """
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
        logger.info('Connected to the broker')

        channel.queue_declare(queue=self.queue_name)
        logger.info('Queue has been declared')

        channel.basic_consume(
            on_message_callback=self._callback,
            queue=self.queue_name,
            auto_ack=True
        )

        channel.start_consuming()
        logger.info('Started consuming messages')

    def _dump_from_body(self, message_body) -> PromptModel | ChatCompletionModel:
        """
        Deserialize the message body into a PromptModel or ChatCompletionModel.

        :param message_body: The body of the message to deserialize.
        :type message_body: dict
        :return: A deserialized PromptModel or ChatCompletionModel.
        :rtype: PromptModel | ChatCompletionModel
        """
        return PromptModel(**message_body['kwargs'])

    def _callback(self, ch, method, properties, body):
        """
        Callback function to handle messages consumed from RabbitMQ.

        This function processes the message using the language model and saves the result in Redis.

        :param ch: The channel object.
        :type ch: pika.adapters.blocking_connection.BlockingChannel
        :param method: Delivery method object.
        :type method: pika.spec.Basic.Deliver
        :param properties: Message properties.
        :type properties: pika.spec.BasicProperties
        :param body: The message body.
        :type body: bytes
        """
        logger.info(json.loads(body))
        prompt_wrapper = PromptWrapper(prompt=json.loads(body)['kwargs'])
        transaction: PromptTransactionModel | ChatCompletionTransactionModel = prompt_wrapper.prompt
        func_result = self.llm(transaction)

        logger.info(f'The LLM response for task {transaction.prompt.job_id} has been generated')
        logger.info(f'{self.redis_prefix}:{transaction.prompt.job_id}\n{func_result}')
        self.redis_bd.save_item(f'{self.redis_prefix}:{transaction.prompt.job_id}', {"content": func_result})
        logger.info(f'The response for task {transaction.prompt.job_id} was written to Redis')
