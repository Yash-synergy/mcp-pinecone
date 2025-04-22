"""
Tool handler implementations for custom MCP tools.

This module contains the implementations of the various tool handlers that
are used by the server.py to process tool requests.
"""
import json
import logging
import subprocess
from typing import List, Dict, Any, Optional, Sequence, Union
import mcp.types as types
import os

# Import ToolName and TOOL_PROCESS_MAP using try/except to handle both cases
try:
    # Try relative import first (for installed package)
    from .tool_definitions import ToolName, TOOL_PROCESS_MAP
    from .pinecone_client import PineconeClient
except ImportError:
    # Fall back to absolute import (for direct script execution)
    from custom_mcp_tools_copy.tool_definitions import ToolName, TOOL_PROCESS_MAP
    from custom_mcp_tools_copy.pinecone_client import PineconeClient

logger = logging.getLogger("custom-mcp-tools-copy")


def call_tool(action, args):
    """
    Call a tool using the tool dispatcher pattern.
    This is adapted from the original tool_dispatcher.py
    """
    if action not in TOOL_PROCESS_MAP:
        raise ValueError(f"Unknown tool: {action}")

    try:
        logger.info(f"Launching tool: {TOOL_PROCESS_MAP[action]} with args: {args}")
        
        proc = subprocess.Popen(
            ["python", TOOL_PROCESS_MAP[action]],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        request = {
            "type": "tool",
            "payload": {"action": action, **(args or {})}
        }
        
        request_json = json.dumps(request)
        logger.debug(f"Sending request to tool: {request_json}")
        
        proc.stdin.write(request_json + '\n')
        proc.stdin.flush()

        # Wait for response with timeout
        stdout_data, stderr_data = proc.communicate(timeout=10)
        
        if stderr_data:
            logger.warning(f"Tool stderr output: {stderr_data}")
            
        if not stdout_data:
            logger.error("No output received from tool")
            return {"error": "No output received from tool"}
            
        logger.debug(f"Raw tool output: '{stdout_data}'")
        
        # Try to parse the response line by line until we find valid JSON
        response = None
        for line in stdout_data.splitlines():
            line = line.strip()
            if not line:
                continue
                
            try:
                response = json.loads(line)
                logger.info(f"Tool result: {response}")
                break
            except json.JSONDecodeError:
                logger.debug(f"Not valid JSON: '{line}'")
                continue
                
        if response is None:
            logger.error("Could not parse JSON response from tool")
            return {"error": "Invalid JSON response from tool"}
            
        return response
    except subprocess.TimeoutExpired:
        proc.terminate()
        logger.error(f"Tool execution timed out after 10 seconds")
        return {"error": "Tool execution timed out"}
    except Exception as e:
        logger.error(f"Error executing tool {action}: {e}")
        return {"error": str(e)}


def semantic_search(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    Read a document from the pinecone knowledge base
    """
    query = arguments.get("query")
    top_k = arguments.get("top_k", 10)
    
    # Handle filter creation
    filters = {}
    if arguments.get("category"):
        filters["category"] = arguments.get("category")
    if arguments.get("tags"):
        filters["tags"] = {"$in": arguments.get("tags")}
    if arguments.get("date_range"):
        date_filter = {}
        if arguments.get("date_range").get("start"):
            date_filter["$gte"] = arguments.get("date_range").get("start")
        if arguments.get("date_range").get("end"):
            date_filter["$lte"] = arguments.get("date_range").get("end")
        if date_filter:
            filters["date"] = date_filter
            
    namespace = arguments.get("namespace")
    
    logger.info(f"Searching with query '{query}', filters {filters}, namespace {namespace}")

    try:
        results = pinecone_client.search_records(
            query=query,
            top_k=top_k,
            filter=filters,
            include_metadata=True,
            namespace=namespace,
        )

        matches = results.get("matches", [])
        logger.info(f"Found {len(matches)} matches")

        # Format results with rich context
        if not matches:
            return [types.TextContent(type="text", text=json.dumps({
                "results": [],
                "message": "No matching documents found"
            }))]
            
        formatted_text = "Retrieved Contexts:\n\n"
        formatted_results = []
        
        for i, match in enumerate(matches, 1):
            metadata = match.get("metadata", {})
            result_text = f"Result {i} | Similarity: {match['score']:.3f} | Document ID: {match['id']}\n"
            result_text += f"{metadata.get('text', '').strip()}\n"
            result_text += "-" * 10 + "\n\n"
            formatted_text += result_text
            
            # Add to structured results for JSON
            formatted_results.append({
                "id": match['id'],
                "score": match['score'],
                "text": metadata.get('text', '').strip(),
                "metadata": metadata
            })

        # Return both a formatted text version and a JSON structure
        return [types.TextContent(type="text", text=json.dumps({
            "results": formatted_results,
            "formatted_text": formatted_text,
            "count": len(matches)
        }))]
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return [types.TextContent(type="text", text=json.dumps({
            "error": str(e),
            "message": "Failed to perform semantic search"
        }))]


def debug_pinecone_client(pinecone_client: PineconeClient) -> None:
    """
    Debug helper function for Pinecone client
    """
    try:
        client_type = type(pinecone_client).__name__
        logger.info(f"Pinecone client type: {client_type}")
        
        # Check if methods exist
        logger.info(f"Client has stats method: {hasattr(pinecone_client, 'stats')}")
        logger.info(f"Client has search_records method: {hasattr(pinecone_client, 'search_records')}")
        
        # Check environment variables
        logger.info(f"PINECONE_API_KEY set: {bool(os.environ.get('PINECONE_API_KEY'))}")
        logger.info(f"PINECONE_INDEX_NAME: {os.environ.get('PINECONE_INDEX_NAME')}")
    except Exception as e:
        logger.error(f"Error in debug_pinecone_client: {e}")


def pinecone_stats(pinecone_client: PineconeClient) -> list[types.TextContent]:
    """
    Get stats about the Pinecone index specified in this server
    """
    try:
        logger.info("Getting Pinecone index stats")
        logger.info(f"Using PineconeClient type: {type(pinecone_client).__name__}")
        
        # Debug pinecone client
        logger.info(f"Pinecone client config - Index name: {os.environ.get('PINECONE_INDEX_NAME', 'unknown')}")
        
        # Run the debug function
        debug_pinecone_client(pinecone_client)
        
        stats = pinecone_client.stats()
        logger.info(f"Raw stats response: {stats}")
        
        # Extract and log namespaces
        namespaces = stats.get('namespaces', {})
        logger.info(f"Found {len(namespaces)} namespaces: {list(namespaces.keys())}")
        
        return [types.TextContent(type="text", text=json.dumps({
            "stats": stats,
            "index_name": os.environ.get("PINECONE_INDEX_NAME", "unknown"),
            "total_vectors": stats.get("total_vector_count", 0)
        }))]
    except Exception as e:
        logger.error(f"Error getting index stats: {e}", exc_info=True)  # Include full traceback
        return [types.TextContent(type="text", text=json.dumps({
            "error": str(e),
            "message": "Failed to get Pinecone stats",
            "error_type": type(e).__name__
        }))]


def read_document(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    Read a single Pinecone document by ID
    """
    try:
        document_id = arguments.get("document_id")
        namespace = arguments.get("namespace")
        if not document_id:
            raise ValueError("document_id is required")

        logger.info(f"Reading document: {document_id} from namespace: {namespace}")

        # Fetch the record using your existing fetch_records method
        record = pinecone_client.fetch_records([document_id], namespace=namespace)

        # Get the vector data for this document
        vector = record.vectors.get(document_id)
        if not vector:
            logger.warning(f"Document {document_id} not found")
            return [types.TextContent(type="text", text=json.dumps({
                "error": "Document not found",
                "document_id": document_id
            }))]

        # Get metadata from the vector
        metadata = vector.metadata if hasattr(vector, "metadata") else {}
        logger.info(f"Document found with metadata keys: {list(metadata.keys())}")

        # Format the document content for human reading
        formatted_content = []
        formatted_content.append(f"Document ID: {document_id}")
        formatted_content.append("")  # Empty line for spacing

        if metadata:
            formatted_content.append("Metadata:")
            for key, value in metadata.items():
                formatted_content.append(f"{key}: {value}")

        # Return both formatted text and structured data
        return [types.TextContent(type="text", text=json.dumps({
            "document_id": document_id,
            "metadata": metadata,
            "namespace": namespace or "default",
            "formatted_text": "\n".join(formatted_content),
            "has_vector": True
        }))]
    except Exception as e:
        logger.error(f"Error reading document: {e}")
        return [types.TextContent(type="text", text=json.dumps({
            "error": str(e),
            "message": f"Failed to read document: {document_id}",
            "document_id": document_id
        }))] 