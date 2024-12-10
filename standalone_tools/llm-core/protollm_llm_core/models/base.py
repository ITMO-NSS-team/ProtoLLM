import os
from abc import ABC

from protollm_sdk.models.job_context_models import PromptTransactionModel, ChatCompletionTransactionModel, PromptModel, \
    ChatCompletionModel


class BaseLLM(ABC):
    def __init__(self, model_path_or_url: str):
        ...

    def __call__(self, transaction: PromptTransactionModel | ChatCompletionTransactionModel) -> str:
        pass

    def generate(
            self,
            message: PromptModel,
            tokens_limit=None,
            temperature=None,
            stop_words=None,
    ) -> str:
        ...

    def create_completion(
            self,
            messages: ChatCompletionModel,
            tokens_limit=None,
            temperature=None,
            stop_words=None,
    ) -> str:
        ...

class LocalLLM(BaseLLM, ABC):
    def __init__(self, model_path_or_url: str, n_ctx=8192):
        if not os.path.exists(model_path_or_url):
            raise ValueError("Invalid model_path_or_url. Must be a valid path")

    def generate(
            self,
            message: PromptModel,
            tokens_limit=None,
            temperature=None,
            repeat_penalty=1.1,
            stop_words=None,
    ) -> str:
        ...

    def create_completion(
            self,
            messages: ChatCompletionModel,
            tokens_limit=None,
            temperature=None,
            repeat_penalty=1.1,
            stop_words=None,
    ) -> str:
        ...

class APIlLLM(BaseLLM, ABC):
    def __init__(self, model_path_or_url: str, token: str, default_model: str = None, timeout_sec: int = 10 * 60):
        if not (model_path_or_url.startswith("http://") or model_path_or_url.startswith("https://")):
            raise ValueError("Invalid model_path_or_url. Must be a valid URL")

    def generate(
            self,
            message: PromptModel,
            tokens_limit=None,
            temperature=None,
            stop_words=None,
            model = None,
    ) -> str:
        ...

    def create_completion(
            self,
            messages: ChatCompletionModel,
            tokens_limit=None,
            temperature=None,
            stop_words=None,
            model=None,
    ) -> str:
        ...