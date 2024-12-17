from protollm_worker.config import MODEL_PATH, REDIS_HOST, REDIS_PORT, QUEUE_NAME
from protollm_worker.models.open_api_llm import OpenAPILLM
from protollm_worker.services.broker import LLMWrap
from protollm_worker.config import (
    RABBIT_MQ_HOST, RABBIT_MQ_PORT,
    RABBIT_MQ_PASSWORD, RABBIT_MQ_LOGIN,
    REDIS_PREFIX
)

if __name__ == "__main__":
    llm_model = OpenAPILLM(model_url="https://api.vsegpt.ru/v1",
                           token="sk-or-vv-7fcc4ab944ca013feb7608fb7c0f001e5c12c32abf66233aad414183b4191a79",
                           default_model="openai/gpt-4o-2024-08-06",
                           app_tag="test-protollm-worker")
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
