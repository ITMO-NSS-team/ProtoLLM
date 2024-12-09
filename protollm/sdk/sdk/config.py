import os
from dataclasses import dataclass
from distutils.command.config import config


@dataclass(frozen=True)
class Config:
    llm_api_host = os.environ.get("PROTO_LLM_LLM_API_HOST", "10.32.15.21")
    llm_api_port = os.environ.get("PROTO_LLM_LLM_API_PORT", "6672")

    redis_host = os.environ.get("PROTO_LLM_REDIS_HOST", "10.32.15.21")
    redis_port = os.environ.get("PROTO_LLM_REDIS_PORT", "6379")

    rabbit_mq_host = os.environ.get("PROTO_LLM_RABBIT_HOST", "10.32.15.21")
    rabbit_mq_port = os.environ.get("PROTO_LLM_RABBIT_PORT", "5672")
    rabbit_mq_login = os.environ.get("RABBIT_MQ_LOGIN", "admin")
    rabbit_mq_password = os.environ.get("RABBIT_MQ_PASSWORD", "admin")

    text_embedder_host = os.environ.get("PROTO_LLM_TEXT_EMB_HOST","10.32.15.21")
    text_embedder_port = os.environ.get("PROTO_LLM_TEXT_EMB_PORT","9942")

    vector_bd_host = os.environ.get("ENV_VAR_VECTOR_HOST", "10.32.15.30")
    vector_db_port = os.environ.get("ENV_VAR_VECTOR_PORT", "9941")

    job_invocation_type = os.environ.get("PROTO_LLM_JOB_INVOCATION_TYPE", "worker")

    celery_queue_name = os.environ.get("PROTO_LLM_CELERY_QUEUE_NAME", "celery")

    @classmethod
    def reload_invocation_type(cls):
        cls.job_invocation_type = os.environ.get("PROTO_LLM_JOB_INVOCATION_TYPE", "worker")
