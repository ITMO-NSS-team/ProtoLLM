import os


class Config:
    def __init__(
            self,
            inner_llm_url: str = "localhost:8670",
            redis_host: str = "localhost",
            redis_port: int = 6379,
            redis_prefix: str = "llm-api",
            rabbit_host: str = "localhost",
            rabbit_port: int = 5672,
            rabbit_login: str = "admin",
            rabbit_password: str = "admin",
            queue_name: str = "llm-api-queue"
    ):
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
    def read_from_env(cls) -> 'Config':
        return Config(
            os.environ.get("INNER_LLM_URL", "localhost:8670"),
            os.environ.get("REDIS_HOST", "localhost"),
            os.environ.get("REDIS_PORT", "6379"),
            os.environ.get("REDIS_PREFIX", "llm-api"),
            os.environ.get("RABBIT_MQ_HOST", "localhost"),
            os.environ.get("RABBIT_MQ_PORT", "5672"),
            os.environ.get("RABBIT_MQ_LOGIN", "admin"),
            os.environ.get("RABBIT_MQ_PASSWORD", "admin"),
            os.environ.get("QUEUE_NAME", "llm-api-queue")
        )

    @classmethod
    def read_from_env_file(cls, path: str) -> 'Config':
        with open(path) as file:
            lines = file.readlines()
        env_vars = {}
        for line in lines:
            key, value = line.split("=")
            env_vars[key] = value
        return Config(
            env_vars.get("INNER_LLM_URL", "localhost:8670"),
            env_vars.get("REDIS_HOST", "localhost"),
            int(env_vars.get("REDIS_PORT", "6379")),
            env_vars.get("REDIS_PREFIX", "llm-api"),
            env_vars.get("RABBIT_MQ_HOST", "localhost"),
            int(env_vars.get("RABBIT_MQ_PORT", "5672")),
            env_vars.get("RABBIT_MQ_LOGIN", "admin"),
            env_vars.get("RABBIT_MQ_PASSWORD", "admin"),
            env_vars.get("QUEUE_NAME", "llm-api-queue")
        )
