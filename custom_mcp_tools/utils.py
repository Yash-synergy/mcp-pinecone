"""
Utility functions and classes for working with Pinecone and MCP tools.
"""


class MCPToolError(Exception):
    """Custom exception for MCP tool errors"""

    def __init__(self, message: str, code: int = 500):
        self.code = code
        self.message = message
        super().__init__(message)


def is_valid_vector_uri(uri: str) -> bool:
    """
    Validate vector URI format

    Parameters:
        uri: The URI to validate.

    Returns:
        bool: True if the URI is valid, False otherwise.
    """
    try:
        if not uri.startswith("pinecone://vectors/"):
            return False
        vector_id = uri.split("/")[-1]
        return bool(vector_id.strip())  # Ensure non-empty ID
    except Exception:
        return False 