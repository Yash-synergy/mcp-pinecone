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
from dataclasses import dataclass
import os

# Import ToolName and TOOL_PROCESS_MAP using try/except to handle both cases
try:
    # Try relative import first (for installed package)
    from .tool_definitions import ToolName, TOOL_PROCESS_MAP
    from .pinecone_client import PineconeClient, PineconeRecord
    from .chunking import Chunk, create_chunker
    from .utils import MCPToolError
except ImportError:
    # Fall back to absolute import (for direct script execution)
    from custom_mcp_tools.tool_definitions import ToolName, TOOL_PROCESS_MAP
    from custom_mcp_tools.pinecone_client import PineconeClient, PineconeRecord
    from custom_mcp_tools.chunking import Chunk, create_chunker
    from custom_mcp_tools.utils import MCPToolError

logger = logging.getLogger("custom-mcp-tools")

@dataclass
class EmbeddingResult:
    embedded_chunks: List[PineconeRecord]
    total_embedded: int


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


def list_documents(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    List all documents in the knowledge base by namespace
    """
    try:
        namespace = arguments.get("namespace")
        logger.info(f"Listing documents in namespace: {namespace}")
        logger.info(f"Using PineconeClient type: {type(pinecone_client).__name__}")
        
        # Run debug function
        try:
            if 'debug_pinecone_client' in globals():
                debug_pinecone_client(pinecone_client)
            else:
                from custom_mcp_tools.server import debug_pinecone_client
                debug_pinecone_client(pinecone_client)
        except Exception as debug_err:
            logger.error(f"Error running debug client: {debug_err}")
        
        # Check if namespace exists and try variations if not found
        try:
            # Log index stats before listing records
            stats = pinecone_client.stats()
            namespaces = stats.get('namespaces', {})
            logger.info(f"Index namespaces before listing: {list(namespaces.keys())}")
            
            # If namespace not found, try variations
            if namespace and namespace not in namespaces:
                logger.warning(f"Requested namespace '{namespace}' not found in index")
                
                # Try lowercase
                lowercase_namespace = namespace.lower()
                if lowercase_namespace in namespaces:
                    logger.info(f"Found lowercase variant '{lowercase_namespace}' - using it instead")
                    namespace = lowercase_namespace
                else:
                    # Try to find case-insensitive match
                    for ns in namespaces.keys():
                        if ns.lower() == namespace.lower():
                            logger.info(f"Found case-insensitive match: '{ns}' - using it instead")
                            namespace = ns
                            break
                            
                # Try to find partial matches
                vessel_namespaces = [ns for ns in namespaces.keys() if 'vessel' in ns.lower()]
                if vessel_namespaces and 'vessel' in namespace.lower():
                    logger.info(f"Found related vessel namespaces: {vessel_namespaces}")
        except Exception as stats_error:
            logger.error(f"Error checking namespaces: {stats_error}")
        
        # Perform the actual list operation
        logger.info(f"Calling list_records with namespace: {namespace}")
        results = pinecone_client.list_records(namespace=namespace)
        logger.info(f"List records raw response: {results}")
        
        vectors = results.get("vectors", [])
        logger.info(f"Found {len(vectors)} vectors in namespace: {namespace or 'default'}")
        
        for i, vector in enumerate(vectors[:5]):  # Log first 5 vectors
            logger.info(f"Vector {i} ID: {vector.get('id')}, metadata keys: {list(vector.get('metadata', {}).keys())}")
        
        return [types.TextContent(type="text", text=json.dumps({
            "vectors": vectors,
            "count": len(vectors),
            "namespace": namespace or "default",
            "pagination_token": results.get("pagination_token")
        }))]
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)  # Include full traceback
        return [types.TextContent(type="text", text=json.dumps({
            "error": str(e),
            "message": "Failed to list documents",
            "error_type": type(e).__name__
        }))]


def debug_pinecone_client(pinecone_client: PineconeClient) -> None:
    """
    Debug function to check the state of the Pinecone client
    """
    try:
        logger.info("====== DEBUG PINECONE CLIENT ======")
        logger.info(f"Client type: {type(pinecone_client).__name__}")
        
        # Check if the client has necessary attributes
        has_pc = hasattr(pinecone_client, 'pc')
        has_index = hasattr(pinecone_client, 'index')
        logger.info(f"Has Pinecone instance: {has_pc}, Has index: {has_index}")
        
        if has_pc:
            logger.info(f"Pinecone instance type: {type(pinecone_client.pc).__name__}")
            
        if has_index:
            logger.info(f"Index type: {type(pinecone_client.index).__name__}")
            logger.info(f"Index name: {getattr(pinecone_client.index, 'name', 'unknown')}")
            logger.info(f"Index host: {getattr(pinecone_client.index, 'host', 'unknown')}")
        
        # Try to access environment variables
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        logger.info(f"Env vars - API key present: {bool(api_key)}, Index name: {index_name}")
        
        logger.info("====== END DEBUG ======")
    except Exception as e:
        logger.error(f"Error during debug: {e}", exc_info=True)


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


def embed_document(
    chunks: list[Chunk], pinecone_client: PineconeClient
) -> EmbeddingResult:
    """
    Embed a list of chunks.
    Uses the Pinecone client to generate embeddings with the inference API.
    """
    embedded_chunks = []
    for chunk in chunks:
        content = chunk.content
        chunk_id = chunk.id
        metadata = chunk.metadata

        if not content or not chunk_id:
            logger.warning(f"Skipping invalid chunk: {chunk}")
            continue

        embedding = pinecone_client.generate_embeddings(content)
        record = {
            "id": chunk_id,
            "embedding": embedding,
            "text": content,
            "metadata": metadata,
        }
        embedded_chunks.append(record)
    return EmbeddingResult(
        embedded_chunks=embedded_chunks,
        total_embedded=len(embedded_chunks),
    )


def process_document(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    Process a document and store it in Pinecone
    """
    try:
        if not arguments:
            raise ValueError("Arguments are required")
        
        content = arguments.get("content")
        document_id = arguments.get("document_id")
        namespace = arguments.get("namespace")
        metadata = arguments.get("metadata", {})
        
        if not content:
            raise ValueError("Content is required")
        
        if not document_id:
            raise ValueError("Document ID is required")
        
        logger.info(f"Processing document: {document_id} for namespace: {namespace}")
        logger.info(f"Content length: {len(content)} characters")
        
        # Create a chunker for the document
        chunker = create_chunker()
        chunks = chunker.chunk_text(
            content=content, 
            document_id=document_id, 
            metadata=metadata
        )
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Embed the chunks
        embedding_result = embed_document(chunks, pinecone_client)
        
        logger.info(f"Embedded {embedding_result.total_embedded} chunks")
        
        # Upsert the records to Pinecone
        upsert_result = pinecone_client.upsert_records(
            embedding_result.embedded_chunks, 
            namespace=namespace
        )
        
        vectors_upserted = upsert_result.get("upserted_count", 0)
        logger.info(f"Upserted {vectors_upserted} vectors to namespace: {namespace}")
        
        # Return a summary of the processing
        summary = {
            "document_id": document_id,
            "chunks_created": len(chunks),
            "chunks_embedded": embedding_result.total_embedded,
            "vectors_upserted": vectors_upserted,
            "namespace": namespace or "default",
            "success": True,
            "message": f"Successfully processed document: {document_id}"
        }
        
        return [types.TextContent(type="text", text=json.dumps(summary))]
    
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        error_summary = {
            "error": str(e),
            "message": "Failed to process document",
            "document_id": arguments.get("document_id") if arguments else None,
            "success": False
        }
        return [types.TextContent(type="text", text=json.dumps(error_summary))]


def debug_pinecone_tool(
    arguments: dict | None, pinecone_client: PineconeClient
) -> list[types.TextContent]:
    """
    Debug tool for Pinecone integration
    """
    try:
        include_raw = arguments.get("include_raw", False) if arguments else False
        test_namespace = arguments.get("test_namespace") if arguments else None
        
        logger.info(f"Running Pinecone debug tool with test_namespace: {test_namespace}")
        
        # Get debug info from the client
        debug_info = {}
        if hasattr(pinecone_client, 'debug_info'):
            debug_info = pinecone_client.debug_info()
        else:
            debug_info = {
                "client_type": type(pinecone_client).__name__,
                "has_custom_debug": False
            }
            
        # Try to access the specified namespace if provided
        if test_namespace:
            try:
                logger.info(f"Testing access to namespace: {test_namespace}")
                results = pinecone_client.list_records(namespace=test_namespace, limit=5)
                debug_info["test_namespace_results"] = {
                    "namespace": test_namespace,
                    "success": True,
                    "vectors_count": len(results.get("vectors", [])),
                    "vectors": results.get("vectors", [])[:2]  # Include only first 2 for brevity
                }
            except Exception as ns_error:
                debug_info["test_namespace_results"] = {
                    "namespace": test_namespace,
                    "success": False,
                    "error": str(ns_error)
                }
        
        # Get a list of all vessel-related namespaces
        try:
            stats = pinecone_client.stats()
            namespaces = stats.get("namespaces", {})
            vessel_namespaces = [ns for ns in namespaces.keys() if 'vessel' in ns.lower()]
            debug_info["vessel_namespaces"] = vessel_namespaces
            
            # Add suggestion if vesselinfo not found
            if vessel_namespaces and 'vesselinfo' not in namespaces:
                closest_match = None
                for ns in vessel_namespaces:
                    if 'info' in ns.lower():
                        closest_match = ns
                        break
                
                if closest_match:
                    debug_info["suggested_namespace"] = closest_match
                    debug_info["suggestion"] = f"Try using '{closest_match}' instead of 'vesselinfo'"
        except Exception as vessel_error:
            debug_info["vessel_namespaces_error"] = str(vessel_error)
            
        # Format and return the debug information
        return [types.TextContent(type="text", text=json.dumps(debug_info, indent=2))]
    except Exception as e:
        logger.error(f"Error in debug_pinecone_tool: {e}")
        return [types.TextContent(type="text", text=json.dumps({
            "error": str(e),
            "message": "Failed to debug Pinecone",
            "error_type": type(e).__name__
        }))] 