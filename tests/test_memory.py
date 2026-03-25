import json
from lm_agent.memory import validate_json_output, deduplicate_tool_calls, estimate_conversation_tokens, prune_conversation

def test_validate_json_output():
    valid, parsed = validate_json_output('{"key": "value"}')
    assert valid is True
    assert parsed["key"] == "value"

    valid, parsed = validate_json_output('Here is the json:\n```json\n{"hello": "world"}\n```')
    assert valid is True
    assert parsed["hello"] == "world"

    valid, parsed = validate_json_output('Invalid JSON {missing quotes}')
    assert valid is False

def test_deduplicate_tool_calls():
    history = []
    tools = [
        {"id": "call_1", "function_name": "read_file", "arguments": '{"file_path": "a.txt"}'},
        {"id": "call_2", "function_name": "read_file", "arguments": '{"file_path": "a.txt"}'},
        {"id": "call_3", "function_name": "get_current_time", "arguments": '{"timezone": "UTC"}'}
    ]
    unique = deduplicate_tool_calls(tools, history)
    assert len(unique) == 2
    assert unique[0]["id"] == "call_1"
    assert unique[1]["id"] == "call_3"

def test_prune_conversation():
    messages = [
        {"role": "system", "content": "You are an AI."},
        {"role": "user", "content": "A" * 100},
        {"role": "assistant", "content": "B" * 100},
        {"role": "user", "content": "C" * 100},
        {"role": "assistant", "content": "D" * 100},
    ]
    # Restrict to very few tokens to force pruning
    pruned = prune_conversation(messages, max_tokens=10)
    # Even when pruned aggressively, it should keep system and at least 4 non-system messages
    assert pruned[0]["role"] == "system"
    assert len(pruned) == 5 # Not enough history to prune below 4 interactions
