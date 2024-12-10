from protollm_llm_core.config import MODEL_PATH, REDIS_HOST, REDIS_PORT, QUEUE_NAME
from protollm_llm_core.models.vllm_models import VllMModel
from protollm_llm_core.services.broker import LLMWrap
from protollm_llm_core.config import (
    RABBIT_MQ_HOST, RABBIT_MQ_PORT,
    RABBIT_MQ_PASSWORD, RABBIT_MQ_LOGIN,
    REDIS_PREFIX
)

if __name__ == "__main__":
    llm_model = VllMModel(model_path=MODEL_PATH)
    llm_wrap = LLMWrap(llm_model=llm_model,
                       redis_host= REDIS_HOST,
                       redis_port= REDIS_PORT,
                       queue_name= QUEUE_NAME,
                       rabbit_host= RABBIT_MQ_HOST,
                       rabbit_port= RABBIT_MQ_PORT,
                       rabbit_login= RABBIT_MQ_LOGIN,
                       rabbit_password= RABBIT_MQ_PASSWORD,
                       redis_prefix= REDIS_PREFIX)
    llm_wrap.start_connection()
