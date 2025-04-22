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
    from .tool_handlers import call_tool, semantic_search
except ImportError:
    # Fall back to absolute import (for direct script execution)
    from custom_mcp_tools_copy.pinecone_client import PineconeClient
    from custom_mcp_tools_copy.tool_definitions import ToolName, SERVER_TOOLS
    from custom_mcp_tools_copy.tool_handlers import call_tool, semantic_search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("custom-mcp-tools-copy")

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

# Create server
server = Server("custom-mcp-tools-copy")

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
            
            # Handle get-time tool
            if name == ToolName.GET_TIME:
                # Call the tool using the dispatcher
                tool_result = call_tool(name, arguments)
                if "error" in tool_result:
                    logger.error(f"Error from tool {name}: {tool_result['error']}")
                    return [types.TextContent(type="text", text=json.dumps(tool_result))]
                return [types.TextContent(type="text", text=json.dumps(tool_result))]
            
            # Handle semantic search tool
            elif name == ToolName.SEMANTIC_SEARCH:
                if not pinecone_client:
                    logger.error("Pinecone client not initialized")
                    raise ValueError("Pinecone client not initialized")
                
                logger.info(f"Handling semantic search")
                return semantic_search(arguments, pinecone_client)
            
            # Handle unknown tool
            else:
                logger.error(f"Unknown tool: {name}")
                raise ValueError(f"Unknown tool: {name}")
                
        except Exception as e:
            logger.error(f"Error handling tool call: {e}")
            return [types.TextContent(type="text", text=json.dumps({
                "error": str(e),
                "message": f"Error handling tool call: {name}"
            }))]

async def main():
    """Main function to start the server"""
    # Register the tools with the server
    register_tools(server)
    
    # Start the server using stdio
    try:
        logger.info("Starting MCP server using stdio")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
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
    except Exception as e:
        logger.error(f"Error running server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 