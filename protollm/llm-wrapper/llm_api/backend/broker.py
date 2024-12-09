import pika
import logging

from llm_api.config import Config
from protollm.sdk.sdk.sdk.models.job_context_models import (
    ResponseModel, ChatCompletionTransactionModel, PromptTransactionModel)
from protollm.sdk.sdk.utils.reddis import RedisWrapper
import json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def send_task(config: Config, queue_name: str, transaction: PromptTransactionModel | ChatCompletionTransactionModel, task_type='generate'):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=config.rabbit_host,
                                  port=config.rabbit_port,
                                  virtual_host='/',
                                  credentials=pika.PlainCredentials(
                                      username=config.rabbit_login,
                                      password=config.rabbit_password)))
    channel = connection.channel()

    channel.queue_declare(queue=queue_name)

    task = {
        "type": "task",
        "task": task_type,
        "args": [],
        "kwargs": transaction.model_dump(),
        "id": transaction.prompt.job_id,
        "retries": 0,
        "eta": None
    }

    message = json.dumps(task)
    channel.basic_publish(
        exchange='',
        routing_key=queue_name,
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2,
        ))
    connection.close()


async def get_result(config: Config, task_id: str, redis_db: RedisWrapper) -> ResponseModel:
    logger.info(f"Trying to get data from redis")
    logger.info(f"Redis key: {config.redis_prefix}:{task_id}")
    while True:
        try:
            p = await redis_db.wait_item(f"{config.redis_prefix}:{task_id}", timeout=90)
            break
        except Exception as ex:
            logger.info(f"Trying to get data from redis")

    model_text = p.decode()

    response = ResponseModel.model_validate_json(model_text)

    return response
