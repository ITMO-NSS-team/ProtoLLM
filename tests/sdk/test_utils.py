import uuid

from protollm.sdk.sdk.sdk.models.job_context_models import ResponseModel
from protollm.sdk.sdk.sdk.object_interface.redis_wrapper import RedisWrapper
from protollm.sdk.sdk.utils.reddis import get_reddis_wrapper, load_result


def test_get_reddis_wrapper():
    redis_wrapper = get_reddis_wrapper()
    assert isinstance(redis_wrapper, RedisWrapper)

def test_load_result():
    job_id = str(uuid.uuid4())
    prefix = None
    redis = get_reddis_wrapper()
    redis.save_item(job_id, {"content": "value"})

    result = load_result(redis, job_id, prefix)

    assert ResponseModel.model_validate_json(result.decode()) == ResponseModel(content= "value")