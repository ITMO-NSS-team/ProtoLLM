from protollm_llm_core.models.vllm_models import VllMModel
from protollm_llm_core.services.broker import LLM_wrap

if __name__ == "__main__":
    llm_wrap = LLM_wrap(VllMModel)
    llm_wrap.start_connection()
