from dataclasses import dataclass

import logging

from protollm.sdk.sdk.sdk.job_context.outer_llm_api import OuterLLMAPI
from protollm.sdk.sdk.sdk.job_context.job_invoke import JobInvoker
from protollm.sdk.sdk.sdk.job_context.vector_db import VectorDB
from protollm.sdk.sdk.sdk.job_context.result_storage import ResultStorage
from protollm.sdk.sdk.sdk.job_context.text_embedder import TextEmbedder
# from sdk.sdk.job_context.langchain_wrapper import LangchainLLMAPI
from protollm.sdk.sdk.sdk.job_context.llm_api import LLMAPI

logger = logging.getLogger(__name__)


@dataclass
class JobContext:
    """
    The class contains contextual services for executing Job
    """
    llm_api: LLMAPI
    outer_llm_api: OuterLLMAPI
    text_embedder: TextEmbedder
    result_storage: ResultStorage
    vector_db: VectorDB
    job_invoker: JobInvoker


