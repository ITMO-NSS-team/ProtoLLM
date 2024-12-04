from abc import ABC

from protollm_sdk.models.job_context_models import PromptTransactionModel, ChatCompletionTransactionModel


class BaseLLM(ABC):
    def __init__(self, model_path, n_ctx=8192):
        ...

    def __call__(self, transaction: PromptTransactionModel | ChatCompletionTransactionModel) -> str:
        pass

    def generate(
            self,
            message,
            tokens_limit=None,
            temperature=None,
            repeat_penalty=1.1,
            stop_words=None,
    ) -> str:
        ...

    def create_completion(
            self,
            messages: dict,
            tokens_limit=None,
            temperature=None,
            repeat_penalty=1.1,
            stop_words=None,
    ) -> str:
        ...
