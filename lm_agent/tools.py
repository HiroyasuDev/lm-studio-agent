import json
from datetime import datetime
from pathlib import Path

# Tool Definitions (No mocked web_search)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time in ISO-8601 format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone name, e.g. 'UTC'"}
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file to read."}
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List all files and folders in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory_path": {"type": "string", "description": "Absolute path to the directory."}
                },
                "required": ["directory_path"],
            },
        },
    }
]

def execute_tool(name: str, arguments: str) -> str:
    """Safely execute a valid tool."""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid JSON arguments: {arguments}"})

    try:
        if name == "get_current_time":
            tz = args.get("timezone", "UTC")
            return json.dumps({"time": datetime.now().isoformat(), "timezone": tz})

        elif name == "read_file":
            path = Path(args["file_path"])
            if not path.exists():
                return json.dumps({"error": f"File not found: {path}"})
            if path.stat().st_size > 50_000:
                return json.dumps({"error": f"File too large: {path.stat().st_size} bytes"})
            return json.dumps({"content": path.read_text(encoding="utf-8", errors="replace")[:10000]})

        elif name == "list_directory":
            path = Path(args["directory_path"])
            if not path.exists():
                return json.dumps({"error": f"Directory not found: {path}"})
            entries = [{"name": i.name, "type": "dir" if i.is_dir() else "file"} for i in sorted(path.iterdir())[:50]]
            return json.dumps({"entries": entries, "total": len(list(path.iterdir()))})

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})
