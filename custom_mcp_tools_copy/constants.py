"""
Constants for the Pinecone integration.
These are used throughout the MCP implementation.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pinecone API configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "mcp-pinecone")

# Embedding model configuration
INFERENCE_MODEL = "text-embedding-ada-002"  # OpenAI model to use
INFERENCE_DIMENSION = 1536  # Dimension of OpenAI's text-embedding-ada-002 