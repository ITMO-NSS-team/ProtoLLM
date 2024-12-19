import json
from typing import Any, Dict, List, Optional, Union

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.outputs import ChatGeneration, ChatResult


class ChatRESTServer(BaseChatModel):
    model: Optional[str] = 'llama3'
    base_url: str = 'http://10.32.2.2:8672'
    timeout: Optional[int] = None
    """Timeout for the request stream"""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-rest-server"

    def _convert_messages_to_rest_server_messages(
            self, messages: List[BaseMessage]
    ) -> List[Dict[str, Union[str, List[str]]]]:
        chat_messages: List = []
        for message in messages:
            role = ""
        match message:
            case HumanMessage():
                role = "user"
            case AIMessage():
                role = "assistant"
            case SystemMessage():
                role = "system"
            case _:
                raise ValueError("Received unsupported message type.")


        content = ""
        if isinstance(message.content, str):
            content = message.content
        else:
            raise ValueError(
                "Unsupported message content type. "
                "Must have type 'text' "
            )
        chat_messages.append(
            {
                "role": role,
                "content": content
            }
        )
        return chat_messages

    def create_chat(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": self._convert_messages_to_rest_server_messages(
                messages)
        }

        if self.base_url=='mock':
            return {'test':'test'}

        response = requests.post(
            url=f'{self.base_url}/v1/chat/completions',
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout
        )
        response.encoding = "utf-8"
        match response.status_code:
            case 200:
                pass  # Status code is 200, no action needed
            case 404:
                raise ValueError(
                    "CustomWeb call failed with status code 404. "
                    "Maybe you need to connect to the corporate network."
                )
            case _:
                optional_detail = response.text
                raise ValueError(
                    f"CustomWeb call failed with status code "
                    f"{response.status_code}. "
                    f"Details: {optional_detail}"
                )
        return json.loads(response.text)

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        response = self._create_chat(messages, stop, **kwargs)
        chat_generation = ChatGeneration(
            message=AIMessage(
                content=response['choices'][0]['message']['content']),
            generation_info=response,
        )
        return ChatResult(generations=[chat_generation])

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model_name": self.model,
            "url": self.base_url
        }
