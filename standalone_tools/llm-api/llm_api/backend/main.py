from fastapi import FastAPI
from llm_api.config import Config
from llm_api.backend.endpoints import get_router

app = FastAPI()

config = Config.read_from_env()

app.include_router(get_router(config))
