import logging
from typing import List, Optional, Any

import requests
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatResult,
    ChatGeneration,
)
from pydantic import PrivateAttr


# Define the custom LLM class
class Llama31ChatModel(BaseChatModel):
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.5
    max_tokens: int = 3000
    logging_level: int = logging.INFO

    _logger: logging.Logger = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True  # Allows arbitrary types like the logger

    def __init__(self, **data):
        super().__init__(**data)
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(level=self.logging_level)

    @property
    def _llm_type(self) -> str:
        return "llama31"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Convert messages to format expected by the API
        context = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                role = "user"  # default
            context.append({"role": role, "content": message.content})

        payload = {
            "model": self.model,
            "messages": context,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if stop is not None:
            payload["stop"] = stop

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )

        if response.status_code == 200:
            assistant_response = response.json()["choices"][0]["message"]["content"]
            self._logger.info("REQUEST: %s", payload)
            self._logger.info("RESPONSE: %s", assistant_response)
            ai_message = AIMessage(content=assistant_response)
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])
        else:
            response.raise_for_status()
