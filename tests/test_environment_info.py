import json

from src.environment_info import collect


def test_collect_has_required_fields_and_is_json_serialisable():
    info = collect()
    for key in ("python", "platform", "cpu_name", "os_cpu_count_logical", "packages", "timestamp_utc"):
        assert key in info
    assert info["packages"]["numpy"] is not None
    json.dumps(info)
