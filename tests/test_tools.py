import json
from lm_agent.tools import execute_tool

def test_execute_tool_invalid_json():
    res = execute_tool("read_file", "invalid args")
    parsed = json.loads(res)
    assert "error" in parsed

def test_execute_tool_unknown_tool():
    res = execute_tool("non_existent", '{"arg": "val"}')
    parsed = json.loads(res)
    assert "Unknown tool" in parsed["error"]

def test_execute_tool_get_time():
    res = execute_tool("get_current_time", '{"timezone": "UTC"}')
    parsed = json.loads(res)
    assert "time" in parsed
    assert parsed["timezone"] == "UTC"
