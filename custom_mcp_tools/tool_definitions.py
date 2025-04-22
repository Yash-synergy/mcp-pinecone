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
    READ_DOCUMENT = "read-document"
    LIST_DOCUMENTS = "list-documents"
    PINECONE_STATS = "pinecone-stats"
    PROCESS_DOCUMENT = "process-document"
    DEBUG_PINECONE = "debug-pinecone"  # New debug tool


# Maps tool names to their corresponding Python scripts
TOOL_PROCESS_MAP = {
    ToolName.GET_TIME: os.path.join(os.path.dirname(__file__), "tool_time.py"),  # Use absolute path to the script
}


# Define available tools with their schemas
SERVER_TOOLS = [
    # Pinecone tools
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
    types.Tool(
        name=ToolName.READ_DOCUMENT,
        description="Read a document from pinecone",
        inputSchema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
                "namespace": {
                    "type": "string",
                    "description": "Optional namespace to read from",
                },
            },
            "required": ["document_id"],
        },
    ),
    types.Tool(
        name=ToolName.LIST_DOCUMENTS,
        description="List all documents in the knowledge base by namespace",
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Namespace to list documents in",
                }
            },
            "required": ["namespace"],
        },
    ),
    types.Tool(
        name=ToolName.PINECONE_STATS,
        description="Get stats about the Pinecone index specified in this server",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name=ToolName.PROCESS_DOCUMENT,
        description="Process a document and store it in Pinecone",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "document_id": {"type": "string"},
                "namespace": {"type": "string"},
                "metadata": {"type": "object"},
            },
            "required": ["content", "document_id", "namespace"],
        },
    ),
    types.Tool(
        name=ToolName.DEBUG_PINECONE,
        description="Debug the Pinecone client connection and configuration",
        inputSchema={
            "type": "object",
            "properties": {
                "include_raw": {
                    "type": "boolean",
                    "description": "Whether to include raw client details in the response"
                },
                "test_namespace": {
                    "type": "string",
                    "description": "Optional namespace to test access to"
                }
            },
            "required": [],
        },
    ),
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