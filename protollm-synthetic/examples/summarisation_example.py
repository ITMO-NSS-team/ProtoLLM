from samplefactory.synthetic_pipelines.chains import SummarisationChain
from samplefactory.utils import Dataset, VLLMChatOpenAI
import pandas as pd
import os
import asyncio

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Python is a popular programming language."
]

df = pd.DataFrame(texts, columns=["content"])
df.to_json("tmp_data/tmp_sample_summarization_dataset.json", index=False)

dataset = Dataset(path="tmp_data/tmp_sample_summarization_dataset.json")
# Expected output: a list of summaries
expected_summaries = [
    "The fox jumps over the dog.",
    "AI is changing the world.",
    "Python is a popular language."
]

# proxy_url = os.environ.get("PROXY_URL")
# openai_api_key = os.environ.get("CHATGPT_OPENAI_API_KEY")
# llm=ChatOpenAI(
#         api_key=openai_api_key,
#         http_client=httpx.AsyncClient(proxy=proxy_url) if proxy_url else None,
#         timeout=60.0
#     )

qwen_large_api_key = os.environ.get("QWEN2VL_OPENAI_API_KEY")
qwen_large_api_base = os.environ.get("QWEN2VL_OPENAI_API_BASE")

llm=VLLMChatOpenAI(
        api_key=qwen_large_api_key,
        base_url=qwen_large_api_base,
        model="/model",
        max_tokens=2048,
        # max_concurrency=10
    )

summarisation_chain = SummarisationChain(llm=llm)
actual_summaries = asyncio.run(summarisation_chain.run(dataset, n_examples=3))
print(actual_summaries)