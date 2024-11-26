import logging
from vllm import LLM, SamplingParams
from config import GPU_MEMORY_UTILISATION, TENSOR_PARALLEL_SIZE, TOKENS_LEN

from models.base import BaseLLM
from protollm.sdk.sdk.sdk.models.job_context_models import PromptModel, ChatCompletionModel, PromptTransactionModel, \
    ChatCompletionTransactionModel, PromptTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VllMModel(BaseLLM):
    def __init__(self, model_path, n_ctx=8192):
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILISATION,
            max_model_len=TOKENS_LEN

        )
        self.handlers = {
            PromptTypes.SINGLE_GENERATION.value: self.generate,
            PromptTypes.CHAT_COMPLETION.value: self.create_completion,
        }

    def __call__(self, transaction: PromptTransactionModel | ChatCompletionTransactionModel):
        prompt_type: PromptTypes = transaction.prompt_type
        func = self.handlers[prompt_type]
        return func(transaction.prompt, **transaction.prompt.meta.model_dump())

    def generate(self, prompt: PromptModel,
                 tokens_limit=8096,
                 temperature=None,
                 repeat_penalty=1.1,
                 stop_words=None,
                 **kwargs
                 ):
        if temperature is None:
            temperature = 0.2
        if stop_words is None:
            stop_words = []
        if tokens_limit <= 8096:
            tokens_limit = 8096
        logger.info(f"start generated from single prompt {prompt.content} and temp {temperature}")

        generated_text = self.model.generate(
            prompt.content, SamplingParams(temperature=temperature, top_p=0.95, max_tokens=tokens_limit)
        )
        logger.info(f"generation result {generated_text}")
        # response = generated_text['choices'][0]['text']
        return generated_text[0].outputs[0].text

    def create_completion(self, prompt: ChatCompletionModel,
                          tokens_limit=8096,
                          temperature=None,
                          repeat_penalty=1.1,
                          stop_words=None,
                          **kwargs
                          ):
        if temperature is None:
            temperature = 2
        if stop_words is None:
            stop_words = []
        if tokens_limit <= 8096:
            tokens_limit = 8096
        logger.info(f"start generated from chat completion {prompt.messages}")

        messages = prompt.model_dump()['messages']
        response = self.model.chat(
            messages,
            SamplingParams(temperature=temperature, top_p=0.95,  max_tokens=tokens_limit)
        )
        logger.info(f"generation result {response}")
        return response[0].outputs[0].text
