# from typing import Any, List, Optional
#
# from langchain_core.callbacks.manager import CallbackManagerForLLMRun
# from langchain_core.language_models.llms import LLM
#
# from sdk.sdk.job_context.llm_api import LLMAPI
# from sdk.sdk.models.job_context_models import LLMRequest
#
#
# class LangchainLLMAPI(LLM):
#     """A custom chat model that echoes the first `n` characters of the input.
#
#     When contributing an implementation to LangChain, carefully document
#     the model including the initialization parameters, include
#     an example of how to initialize the model and include any relevant
#     links to the underlying models documentation or API.
#
#     Example:
#
#         .. code-block:: python
#
#             model = CustomChatModel(n=2)
#             result = model.invoke([HumanMessage(content="hello")])
#             result = model.batch([[HumanMessage(content="hello")],
#                                  [HumanMessage(content="world")]])
#     """
#
#     def __init__(self, llm_api: LLMAPI, **kwargs: Any):
#         super().__init__(**kwargs)
#         self.llm_api = llm_api
#
#     def _call(
#         self,
#         request: LLMRequest,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> str:
#         """Run the LLM on the given input.
#
#         Override this method to implement the LLM logic.
#
#         Args:
#             prompt: The prompt to generate from.
#             stop: Stop words to use when generating. Model output is cut off at the
#                 first occurrence of any of the stop substrings.
#                 If stop tokens are not supported consider raising NotImplementedError.
#             run_manager: Callback manager for the run.
#             **kwargs: Arbitrary additional keyword arguments. These are usually passed
#                 to the model provider API call.
#
#         Returns:
#             The model output as a string. Actual completions SHOULD NOT include the prompt.
#         """
#         response = self.llm_api.inference(request)
#         return response.text
#
#     @property
#     def _llm_type(self) -> str:
#         """Get the type of language model used by this chat model. Used for logging purposes only."""
#         return "custom"
