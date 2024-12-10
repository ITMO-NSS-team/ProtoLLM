import logging
from time import sleep

from protollm_llm_core.models.base import APIlLLM, BaseLLM
from protollm_sdk.models.job_context_models import PromptTypes, PromptTransactionModel, ChatCompletionTransactionModel, \
    PromptModel, ChatCompletionModel
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAPILLM(APIlLLM, BaseLLM):
    def __init__(self, model_url: str, token: str, default_model: str = None, timeout_sec: int = 10 * 60):
        super().__init__(model_url)

        self.model = default_model
        self.token = token
        self.timeout_sec = timeout_sec
        self.client = OpenAI(
            api_key= token,
            base_url = model_url
        )
        self.handlers = {
            PromptTypes.SINGLE_GENERATION.value: self.generate,
            PromptTypes.CHAT_COMPLETION.value: self.create_completion,
        }

    def __call__(self, transaction: PromptTransactionModel | ChatCompletionTransactionModel):
        prompt_type: PromptTypes = transaction.prompt_type
        func = self.handlers[prompt_type]
        return func(transaction.prompt, **transaction.prompt.meta.model_dump())

    def _chat_completion(
            self,
            messages: list[dict[str, str]],
            temperature: float,
            tokens_limit: int,
            model: str or None
    ) -> str:
        """
        Set up a chat completion

        :param messages: history of messages
        :type messages: list[dict[str, str]]
        :param temperature: degree of creativity in the responses
        :type temperature: float
        :param tokens_limit: maximum number of tokens in the response
        :type tokens_limit: int
        :param model: model name
        :type model: str or None
        :return: str
        """
        sleep(1)  # TODO remake in singleton
        # GPT chats don't respond more than 1 message at second
        model = model or self.model
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            n=1,
            max_tokens=tokens_limit,
            timeout=self.timeout_sec)

        result = response.choices[0].message.content
        logger.info(result)
        return result

    def generate(
            self,
            prompt: PromptModel,
            tokens_limit=None,
            temperature=None,
            model = None,
            **kwargs
    ):
        if temperature is None:
            temperature = 0.5
        messages = [{"role": "user", "content": prompt.content}]
        logger.info(f"start generated from single prompt {prompt.content} and temp {temperature}")
        try:
            result = self._chat_completion(
                messages,
                temperature,
                tokens_limit,
                model,
            )
            return result
        except Exception as ex:
            msg = f"The response generation has been interrupted. Error: {ex}."
            logger.error(msg)
            raise Exception(msg)

    def create_completion(
            self,
            prompt: ChatCompletionModel,
            tokens_limit=None,
            temperature=None,
            model = None,
            **kwargs
    ):
        if temperature is None:
            temperature = 0.5
        logger.info(f"start generated from chat completion {prompt.messages}")
        messages = [{"role": unit.role, "content": unit.content} for unit in prompt.messages]
        try:
            result = self._chat_completion(
                messages,
                temperature,
                tokens_limit,
                model
            )
            return result
        except Exception as ex:
            msg = f"The response generation has been interrupted. Error: {ex}."
            logger.error(msg)
            raise Exception(msg)