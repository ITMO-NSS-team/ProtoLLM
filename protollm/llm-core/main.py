from services.broker import LLM_wrap
from models.cpp_models import CppModel
from models.vllm_models import VllMModel


if __name__ == "__main__":
    llm_wrap = LLM_wrap(VllMModel)
    llm_wrap.start_connection()
