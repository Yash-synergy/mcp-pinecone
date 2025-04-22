"""
Tool definitions for custom MCP tools.

This module defines the available tools, their schemas, and related constants.
"""
import mcp.types as types
from enum import Enum
from typing import List, Dict
import os


class ToolName(str, Enum):
    """Enum of available tool names"""
    GET_TIME = "get-time"
    SEMANTIC_SEARCH = "semantic-search"


# Maps tool names to their corresponding Python scripts
TOOL_PROCESS_MAP = {
    ToolName.GET_TIME: os.path.join(os.path.dirname(__file__), "tool_time.py"),  # Use absolute path to the script
}


# Define available tools with their schemas
SERVER_TOOLS = [
    # Semantic search tool
	# •	A string filter used to match a specific metadata field called "category" in your stored vectors.
	# •	This assumes that when you inserted (upserted) data into Pinecone, you included a metadata["category"] field.
	# •	Example use case:
	# •	You embed both "research_papers" and "news_articles" into Pinecone.
	# •	You can filter: "category": "research_paper" to search only within that group.
    types.Tool(
        name=ToolName.SEMANTIC_SEARCH,
        description="Search pinecone for documents",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
                "namespace": {
                    "type": "string",
                    "description": "Optional namespace to search in",
                },
                "category": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "date_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date"},
                        "end": {"type": "string", "format": "date"},
                    },
                },
            },
            "required": ["query"],
        },
    ),
    # Time tool
    types.Tool(
        name=ToolName.GET_TIME,
        description="Get the current date and time",
        inputSchema={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string", 
                    "description": "Optional format string (default: RFC 3339)"
                },
                "timezone": {
                    "type": "string",
                    "description": "Optional timezone (default: UTC)"
                }
            },
            "required": [],
        },
    ),
] 