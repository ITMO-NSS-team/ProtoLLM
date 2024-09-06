import ast
import json

from protollm.agents.agent_response_parsers.agent_response_parser import AgentResponseParser


class Llama31ResponseParser(AgentResponseParser):
    """
    Parses a response from the assistant to extract function calls and their parameters.
    """
    def __init__(
        self, function_name_field: str = "name", parameters_field: str = "parameters"
    ):
        self.function_name_field = function_name_field
        self.parameters_field = parameters_field

    def parse_function_call(self, response: str):
        try:
            parsed_response = ast.literal_eval(response)
            match parsed_response:
                case list():
                    return [
                        self._parse_single_function_call(function_call)
                        for function_call in parsed_response
                    ]
                case dict():
                    return self._parse_single_function_call(parsed_response)
                case _:
                    raise ValueError("Response must be a list or a dictionary")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse function call: {e}")

    def _parse_single_function_call(self, function_call: dict):
        data = function_call
        function_name = data[self.function_name_field]
        parameters = data[self.parameters_field]

        # Handle cases where parameters contain complex objects as JSON strings or Python literals
        for key, value in parameters.items():
            if isinstance(value, str):
                # Attempt to parse the string as JSON
                try:
                    parsed_value = json.loads(value)
                    parameters[key] = parsed_value
                except json.JSONDecodeError:
                    # If JSON parsing fails, try parsing as a Python literal (e.g., dict or list)
                    try:
                        parsed_value = ast.literal_eval(value)
                        parameters[key] = parsed_value
                    except (ValueError, SyntaxError):
                        # If it fails, the value remains as a string
                        continue

        return function_name, parameters
