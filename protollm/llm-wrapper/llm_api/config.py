import os

class Config:
    def __init__(self,
                 inner_llm_url: str = "10.32.2.5:8670",
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_prefix: str = "proto-llm",
                 rabbit_host: str = "localhost",
                 rabbit_port: int = 5672,
                 rabbit_login: str = "admin",
                 rabbit_password: str = "admin",
                 queue_name: str = "proto-llm-queue"):

        self.inner_lln_url = inner_llm_url
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_prefix = redis_prefix
        self.rabbit_host = rabbit_host
        self.rabbit_port = rabbit_port
        self.rabbit_login = rabbit_login
        self.rabbit_password = rabbit_password
        self.queue_name = queue_name

    @classmethod
    def read_from_env(cls) ->'Config':
        return Config(os.environ.get("INNER_LLM_URL", "10.32.2.5:8670"),
                      os.environ.get("REDIS_HOST", "10.32.15.21"),
                      os.environ.get("REDIS_PORT", "6379"),
                      os.environ.get("REDIS_PREFIX", "proto-llm"),
                      os.environ.get("RABBIT_MQ_HOST", "10.32.15.21"),
                      os.environ.get("RABBIT_MQ_PORT", "5672"),
                      os.environ.get("RABBIT_MQ_LOGIN", "admin"),
                      os.environ.get("RABBIT_MQ_PASSWORD", "admin"),
                      os.environ.get("QUEUE_NAME", "proto-llm-queue"))