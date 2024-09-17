from abc import ABC, abstractmethod


class AgentResponseParser(ABC):
    @abstractmethod
    def parse_function_call(self, response):
        pass
