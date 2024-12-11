import pika
import logging
import json

from protollm_api.config import Config
from protollm_sdk.models.job_context_models import (
    ResponseModel, ChatCompletionTransactionModel, PromptTransactionModel)
from protollm_sdk.utils.reddis import RedisWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def send_task(config: Config, queue_name: str, transaction: PromptTransactionModel | ChatCompletionTransactionModel, task_type='generate'):
    """
    Sends a task to the RabbitMQ queue.

    Args:
        config (Config): Configuration object containing RabbitMQ connection details.
        queue_name (str): Name of the RabbitMQ queue where the task will be published.
        transaction (PromptTransactionModel | ChatCompletionTransactionModel): Transaction data to be sent.
        task_type (str, optional): The type of task to be executed (default is 'generate').

    Raises:
        Exception: If there is an error during the connection or message publishing process.
    """
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=config.rabbit_host,
            port=config.rabbit_port,
            virtual_host='/',
            credentials=pika.PlainCredentials(
                username=config.rabbit_login,
                password=config.rabbit_password
            )
        )
    )
    channel = connection.channel()

    # Declare the queue if it does not exist
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

    # Publish the message to the RabbitMQ queue
    channel.basic_publish(
        exchange='',
        routing_key=queue_name,
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2,  # Make message persistent
        )
    )
    connection.close()


async def get_result(config: Config, task_id: str, redis_db: RedisWrapper) -> ResponseModel:
    """
    Retrieves the result of a task from Redis.

    Args:
        config (Config): Configuration object containing Redis connection details.
        task_id (str): ID of the task whose result is to be retrieved.
        redis_db (RedisWrapper): Redis wrapper object to interact with the Redis database.

    Returns:
        ResponseModel: Parsed response model containing the result.

    Raises:
        Exception: If the result is not found within the timeout period or other errors occur.
    """
    logger.info(f"Trying to get data from Redis")
    logger.info(f"Redis key: {config.redis_prefix}:{task_id}")
    while True:
        try:
            # Wait for the result to be available in Redis
            p = await redis_db.wait_item(f"{config.redis_prefix}:{task_id}", timeout=90)
            break
        except Exception as ex:
            logger.info(f"Retrying to get data from Redis: {ex}")

    # Decode and validate the result
    model_text = p.decode()
    response = ResponseModel.model_validate_json(model_text)

    return response
