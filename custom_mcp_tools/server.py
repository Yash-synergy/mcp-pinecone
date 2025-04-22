#!/usr/bin/env python3
"""
Custom MCP Tools Server with Pinecone integration.

Environment Variables:
    PINECONE_API_KEY - Your Pinecone API key (required for Pinecone operations)
    PINECONE_INDEX_NAME - Name of your Pinecone index (required for Pinecone operations)
    OPENAI_API_KEY - Your OpenAI API key (required for embedding generation)

To run with environment variables:
    PINECONE_API_KEY=your_api_key PINECONE_INDEX_NAME=your_index_name python custom_mcp_tools/server.py

Or set them in your environment before running:
    export PINECONE_API_KEY=your_api_key
    export PINECONE_INDEX_NAME=your_index_name
    python custom_mcp_tools/server.py
"""
import logging
import asyncio
import json
import os
import sys
from typing import Union, Sequence

# Now import mcp modules
import mcp.types as types
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# Get environment variables
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Import OpenAI for embeddings
from openai import OpenAI

# Import our constants
from .constants import INFERENCE_DIMENSION

# Import Pinecone client and related modules
try:
    # Try relative import first (for installed package)
    from .pinecone_client import PineconeClient
    from .tool_definitions import ToolName, SERVER_TOOLS
    from .tool_handlers import (
        call_tool, 
        list_documents, 
        semantic_search, 
        pinecone_stats, 
        read_document, 
        process_document,
        debug_pinecone_tool
    )
except ImportError:
    # Fall back to absolute import (for direct script execution)
    from custom_mcp_tools.pinecone_client import PineconeClient
    from custom_mcp_tools.tool_definitions import ToolName, SERVER_TOOLS
    from custom_mcp_tools.tool_handlers import (
        call_tool, 
        list_documents, 
        semantic_search, 
        pinecone_stats, 
        read_document, 
        process_document,
        debug_pinecone_tool
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("custom-mcp-tools")

# Log warning if environment variables are not set
if not PINECONE_API_KEY:
    logger.warning("PINECONE_API_KEY environment variable not set. Pinecone operations will fail.")
if not PINECONE_INDEX_NAME:
    logger.warning("PINECONE_INDEX_NAME environment variable not set. Pinecone operations will fail.")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not set. Embedding generation will fail.")

# Create OpenAI client for embeddings
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Create a custom PineconeClient that uses OpenAI embeddings
class CustomPineconeClient(PineconeClient):
    def __init__(self):
        """
        Initialize the CustomPineconeClient with API key and index name from environment variables.
        """
        logger.info("Initializing CustomPineconeClient")
        # Log environment variables (redacted for security)
        has_api_key = bool(PINECONE_API_KEY)
        has_index_name = bool(PINECONE_INDEX_NAME)
        logger.info(f"Environment variables: PINECONE_API_KEY present: {has_api_key}, "
                   f"PINECONE_INDEX_NAME present: {has_index_name} (value: {PINECONE_INDEX_NAME})")
        
        if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
            error_msg = "Missing required Pinecone configuration: "
            if not PINECONE_API_KEY:
                error_msg += "PINECONE_API_KEY "
            if not PINECONE_INDEX_NAME:
                error_msg += "PINECONE_INDEX_NAME"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Set environment variables for the parent class to use
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        os.environ["PINECONE_INDEX_NAME"] = PINECONE_INDEX_NAME
        
        try:
            # Initialize the parent class (which will use the environment variables)
            logger.info(f"Calling PineconeClient.__init__() with index: {PINECONE_INDEX_NAME}")
            super().__init__()
            logger.info(f"PineconeClient.__init__() completed successfully")
            
            # Test if the index reference is valid
            try:
                logger.info("Testing index access by getting stats")
                stats = self.stats()
                namespace_count = len(stats.get("namespaces", {}))
                logger.info(f"Index access successful. Found {namespace_count} namespaces.")
            except Exception as stats_error:
                logger.error(f"Error accessing index after initialization: {stats_error}", exc_info=True)
                
            logger.info(f"CustomPineconeClient initialized with index: {PINECONE_INDEX_NAME}")
        except Exception as init_error:
            logger.error(f"Error during PineconeClient initialization: {init_error}", exc_info=True)
            raise

    def generate_embeddings(self, text: str) -> list[float]:
        """
        Generate embeddings using OpenAI's API to ensure compatibility with the existing index.
        """
        try:
            logger.info(f"Generating embedding for text: {text[:50]}...")
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = response.data[0].embedding
            logger.info(f"Generated embedding with dimension: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def debug_info(self) -> dict:
        """
        Return debug information about the Pinecone client and index
        """
        debug_data = {
            "client_type": type(self).__name__,
            "has_pinecone": hasattr(self, "pc"),
            "has_index": hasattr(self, "index"),
            "api_key_present": bool(PINECONE_API_KEY),
            "index_name": PINECONE_INDEX_NAME,
        }
        
        if hasattr(self, "pc"):
            debug_data["pinecone_type"] = type(self.pc).__name__
        
        if hasattr(self, "index"):
            debug_data["index_type"] = type(self.index).__name__
            try:
                # Try to get index properties
                debug_data["index_name"] = getattr(self.index, "name", "unknown")
                debug_data["index_host"] = getattr(self.index, "host", "unknown")
            except:
                debug_data["index_properties_error"] = "Could not access index properties"
                
        # Get namespaces
        try:
            stats = self.stats()
            namespaces = stats.get("namespaces", {})
            debug_data["namespaces_count"] = len(namespaces)
            debug_data["namespaces"] = list(namespaces.keys())
            
            # Add lowercase versions for comparison
            debug_data["namespaces_lowercase"] = [ns.lower() for ns in namespaces.keys()]
        except Exception as e:
            debug_data["namespaces_error"] = str(e)
            
        return debug_data

# Create server
server = Server("custom-mcp-tools")

def register_tools(server):
    """Register tools with the MCP server"""
    
    # Initialize Pinecone client
    pinecone_client = None
    try:
        # Check if Pinecone environment variables are set before attempting to initialize
        if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
            logger.error("Missing required Pinecone environment variables. Tools requiring Pinecone will be unavailable.")
        else:
            pinecone_client = CustomPineconeClient()
            logger.info(f"Custom Pinecone client initialized successfully with index: {PINECONE_INDEX_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {e}")
    
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        logger.info("Listing tools")
        return SERVER_TOOLS

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> Sequence[Union[types.TextContent, types.ImageContent]]:
        try:
            logger.info(f"Calling tool: {name} with args: {arguments}")
            
            # Handle dispatcher-based tools
            if name == ToolName.GET_TIME:
                # Call the tool using the dispatcher
                tool_result = call_tool(name, arguments)
                if "error" in tool_result:
                    logger.error(f"Error from tool {name}: {tool_result['error']}")
                    return [types.TextContent(type="text", text=json.dumps(tool_result))]
                return [types.TextContent(type="text", text=json.dumps(tool_result))]
            
            # Handle Pinecone tools
            elif name in [ToolName.SEMANTIC_SEARCH, ToolName.READ_DOCUMENT, 
                         ToolName.LIST_DOCUMENTS, ToolName.PINECONE_STATS, 
                         ToolName.PROCESS_DOCUMENT, ToolName.DEBUG_PINECONE]:
                if not pinecone_client:
                    logger.error("Pinecone client not initialized")
                    raise ValueError("Pinecone client not initialized")
                
                logger.info(f"Handling Pinecone tool: {name}")
                result = None
                
                if name == ToolName.SEMANTIC_SEARCH:
                    result = semantic_search(arguments, pinecone_client)
                elif name == ToolName.PINECONE_STATS:
                    result = pinecone_stats(pinecone_client)
                elif name == ToolName.READ_DOCUMENT:
                    result = read_document(arguments, pinecone_client)
                elif name == ToolName.LIST_DOCUMENTS:
                    result = list_documents(arguments, pinecone_client)
                elif name == ToolName.PROCESS_DOCUMENT:
                    result = process_document(arguments, pinecone_client)
                elif name == ToolName.DEBUG_PINECONE:
                    result = debug_pinecone_tool(arguments, pinecone_client)
                
                logger.info(f"Pinecone tool result: {result}")
                return result
            else:
                logger.error(f"Unknown tool: {name}")
                raise ValueError(f"Unknown tool: {name}")
                
        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}", exc_info=True)  # Add traceback
            # Return error as a valid response
            return [types.TextContent(type="text", text=json.dumps({
                "error": str(e),
                "error_type": type(e).__name__,
                "message": f"Failed to execute tool: {name}"
            }))]

# Add this new function to help with Pinecone debugging
def debug_pinecone_client(client):
    """
    Log debug information about the Pinecone client
    """
    try:
        logger.info("====== DEBUG PINECONE CLIENT ======")
        logger.info(f"Client type: {type(client).__name__}")
        logger.info(f"Has Pinecone instance: {hasattr(client, 'pc')}, Has index: {hasattr(client, 'index')}")
        
        if hasattr(client, 'pc'):
            logger.info(f"Pinecone instance type: {type(client.pc).__name__}")
            
        if hasattr(client, 'index'):
            logger.info(f"Index type: {type(client.index).__name__}")
            logger.info(f"Index name: {getattr(client.index, 'name', 'unknown')}")
            logger.info(f"Index host: {getattr(client.index, 'host', 'unknown')}")
            
        logger.info(f"Env vars - API key present: {bool(PINECONE_API_KEY)}, Index name: {PINECONE_INDEX_NAME}")
        
        # Try to get namespaces
        try:
            stats = client.stats()
            namespaces = stats.get("namespaces", {})
            logger.info(f"Found {len(namespaces)} namespaces: {list(namespaces.keys())[:10]} {'...' if len(namespaces) > 10 else ''}")
            
            # Check for variations of 'vesselinfo'
            vessel_namespaces = [ns for ns in namespaces.keys() if 'vessel' in ns.lower()]
            if vessel_namespaces:
                logger.info(f"Vessel-related namespaces: {vessel_namespaces}")
        except Exception as e:
            logger.error(f"Error getting namespaces: {e}")
            
        logger.info("====== END DEBUG ======")
    except Exception as e:
        logger.error(f"Error in debug_pinecone_client: {e}")

async def main():
    logger.info("Starting Custom MCP Tools server")
    
    # Log Pinecone configuration status
    if PINECONE_API_KEY and PINECONE_INDEX_NAME:
        logger.info(f"Using Pinecone index: {PINECONE_INDEX_NAME}")
    else:
        logger.warning("Pinecone not fully configured. Set PINECONE_API_KEY and PINECONE_INDEX_NAME environment variables to enable Pinecone tools.")

    # Register tools
    register_tools(server)

    # Run the server with stdio
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("Server running with stdio connection")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="custom-mcp-tools",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise 