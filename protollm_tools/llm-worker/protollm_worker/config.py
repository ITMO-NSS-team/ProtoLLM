import os

REDIS_PREFIX = os.environ.get("REDIS_PREFIX", "llm-api")
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")

RABBIT_MQ_HOST = os.environ.get("RABBIT_MQ_HOST", "rabbitmq")
RABBIT_MQ_PORT = os.environ.get("RABBIT_MQ_PORT", "5672")
RABBIT_MQ_LOGIN = os.environ.get("RABBIT_MQ_LOGIN", "admin")
RABBIT_MQ_PASSWORD = os.environ.get("RABBIT_MQ_PASSWORD", "admin")

QUEUE_NAME = os.environ.get("QUEUE_NAME", "llm-api-queue")
MODEL_PATH = os.environ.get("MODEL_PATH")
TOKENS_LEN = int(os.environ.get("TOKENS_LEN", "16384"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "2"))
GPU_MEMORY_UTILISATION = float(os.environ.get("GPU_MEMORY_UTILISATION", "0.9"))
