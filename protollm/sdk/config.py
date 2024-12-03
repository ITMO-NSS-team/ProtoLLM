import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    llm_api_host = os.environ.get("LLM_API_HOST", "localhost")
    llm_api_port = os.environ.get("LLM_API_PORT", "6672")

    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", "6379")

    rabbit_mq_host = os.environ.get("RABBIT_HOST", "localhost")
    rabbit_mq_port = os.environ.get("RABBIT_PORT", "5672")
    rabbit_mq_login = os.environ.get("RABBIT_MQ_LOGIN", "admin")
    rabbit_mq_password = os.environ.get("RABBIT_MQ_PASSWORD", "admin")

    text_embedder_host = os.environ.get("TEXT_EMB_HOST", "localhost")
    text_embedder_port = os.environ.get("TEXT_EMB_PORT", "9942")

    vector_bd_host = os.environ.get("VECTOR_HOST", "localhost")
    vector_db_port = os.environ.get("VECTOR_PORT", "9941")

    job_invocation_type = os.environ.get("JOB_INVOCATION_TYPE", "worker")

    celery_queue_name = os.environ.get("CELERY_QUEUE_NAME", "celery")

    @classmethod
    def reload_invocation_type(cls):
        cls.job_invocation_type = os.environ.get("JOB_INVOCATION_TYPE", "worker")
