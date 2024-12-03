from dataclasses import dataclass

import logging

from protollm.sdk.jobs.outer_llm_api import OuterLLMAPI
from protollm.sdk.jobs.job_invoke import JobInvoker
from protollm.sdk.jobs.vector_db import VectorDB
from protollm.sdk.jobs.result_storage import ResultStorage
from protollm.sdk.jobs.text_embedder import TextEmbedder
from protollm.sdk.jobs.llm_api import LLMAPI

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


