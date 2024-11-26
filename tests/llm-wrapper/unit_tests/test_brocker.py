import uuid

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from llm_api.backend.broker import send_task, get_result
from protollm.sdk.sdk.sdk.models.job_context_models import ResponseModel, ChatCompletionTransactionModel, ChatCompletionModel, \
    PromptMeta, ChatCompletionUnit, PromptTypes
import json

@pytest.mark.asyncio
async def test_send_task(test_local_config):
    prompt = ChatCompletionModel(
        job_id= str(uuid.uuid4()),
        meta= PromptMeta(),
        messages= [ChatCompletionUnit(role="user", content="test request")]
    )
    transaction = ChatCompletionTransactionModel(prompt=prompt, prompt_type=PromptTypes.CHAT_COMPLETION.value)

    with patch("llm_api.backend.broker.pika.BlockingConnection") as mock_connection:
        mock_channel = MagicMock()
        mock_connection.return_value.channel.return_value = mock_channel

        await send_task(test_local_config, test_local_config.queue_name, transaction)

        mock_connection.assert_called_once()
        mock_channel.queue_declare.assert_called_with(queue=test_local_config.queue_name)
        mock_channel.basic_publish.assert_called_once()


@pytest.mark.asyncio
async def test_get_result(test_local_config):
    redis_mock = MagicMock()
    redis_mock.wait_item = AsyncMock(return_value=json.dumps({"content": "return test success"}).encode())
    task_id = str(uuid.uuid4())

    response = await get_result(test_local_config, task_id, redis_mock)

    redis_mock.wait_item.assert_called_once_with(f"{test_local_config.redis_prefix}:{task_id}", timeout=90)
    assert response == ResponseModel(content="return test success")


@pytest.mark.asyncio
async def test_get_result_with_exception(test_local_config):
    redis_mock = MagicMock()
    redis_mock.wait_item = AsyncMock(side_effect=[Exception("Redis error"), json.dumps({"content": "return test success"}).encode()])
    task_id = str(uuid.uuid4())

    response = await get_result(test_local_config, task_id, redis_mock)

    assert redis_mock.wait_item.call_count == 2
    assert response == ResponseModel(content="return test success")
