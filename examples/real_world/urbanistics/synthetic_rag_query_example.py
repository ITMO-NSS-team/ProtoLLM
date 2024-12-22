import os
from protollm_synthetic.synthetic_pipelines.chains import RAGChain
from protollm_synthetic.utils import Dataset, VLLMChatOpenAI
import asyncio

dataset = Dataset(data_col='content', path='data/sample_data_city_rag.json')

print(dataset.data)

qwen_large_api_key = os.environ.get("QWEN_OPENAI_API_KEY")
qwen_large_api_base = os.environ.get("QWEN_OPENAI_API_BASE")

llm=VLLMChatOpenAI(
        api_key=qwen_large_api_key,
        base_url=qwen_large_api_base,
        model="/model",
        max_tokens=2048,
    )

rag_chain = RAGChain(llm=llm)
asyncio.run(rag_chain.run(dataset, 
                          n_examples=3))  

print(rag_chain.data)