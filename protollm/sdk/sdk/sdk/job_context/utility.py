import os

from protollm.sdk.sdk.config import Config
from protollm.sdk.sdk.sdk.job_context.outer_llm_api import OuterLLMAPI
from protollm.sdk.sdk.sdk.job_context.job_invoke import JobInvoker, InvokeType
from protollm.sdk.sdk.sdk.job_context.vector_db import VectorDB
from protollm.sdk.sdk.env import (ENV_VAR_LLM_API_HOST, ENV_VAR_LLM_API_PORT, ENV_VAR_TEXT_EMB_HOST, ENV_VAR_TEXT_EMB_PORT,
                            ENV_VAR_REDIS_PORT, ENV_VAR_REDIS_HOST, ENV_VAR_VECTOR_PORT, ENV_VAR_VECTOR_HOST,
                            ENV_VAR_JOB_INVOCATION_TYPE)
from protollm.sdk.sdk.sdk.job_context.job_context import JobContext
from protollm.sdk.sdk.sdk.job_context.llm_api import LLMAPI
from protollm.sdk.sdk.sdk.job_context.result_storage import ResultStorage
from protollm.sdk.sdk.sdk.job_context.text_embedder import TextEmbedder


def construct_job_context(job_name: str, abstract_task = None) -> JobContext:
    """
    Create JobContext object with object access to functions and services, based on environment variable values.

    :param job_name: job name
    :type job_name: str
    :param abstract_task: optional reference to celery.task for recursive Job calling from Job
    :return: JobContext object
    """

    llm_api = LLMAPI(Config.llm_api_host, Config.llm_api_port)

    out_llm_api = OuterLLMAPI()

    text_embedder = TextEmbedder(Config.text_embedder_host, Config.text_embedder_port)

    result_storage = ResultStorage(redis_host=Config.redis_host,
                                   redis_port=Config.redis_port,
                                   prefix=job_name)

    invoke_type_str = Config.job_invocation_type.lower()
    match invoke_type_str:
        case "worker":
            invoke_type = InvokeType.Worker
        case "blocking":
            invoke_type = InvokeType.Blocking
        case _:
            raise ValueError(f"Found unknown invocation type '{invoke_type_str}'.")
    job_invoker = JobInvoker(abstract_task, result_storage, invoke_type)

    vector_db = VectorDB(vector_bd_host=Config.vector_bd_host, vector_db_port=Config.vector_db_port)

    return JobContext(
        llm_api=llm_api,
        outer_llm_api=out_llm_api,
        text_embedder=text_embedder,
        result_storage=result_storage,
        vector_db=vector_db,
        job_invoker=job_invoker,
    )
