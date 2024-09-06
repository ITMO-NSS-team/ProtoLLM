import importlib
from abc import ABC, abstractmethod


class LargeLanguageModelAgent(ABC):
    @abstractmethod
    def loop_until_satisfactory(self, user_prompt):
        pass

    @staticmethod
    def _execute_function(module_name, function_name, parameters):
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        return func(**parameters)
