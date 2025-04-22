"""
Utility functions for the custom MCP tools.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("custom-mcp-tools-copy")

class MCPToolError(Exception):
    """Base exception for MCP tool errors."""
    pass


def format_response(result: Any, error: Optional[str] = None) -> Dict[str, Any]:
    """
    Format a response for the MCP tool interface.
    
    Args:
        result: The result of the tool operation.
        error: An optional error message if the operation failed.
        
    Returns:
        A dictionary with the formatted response.
    """
    if error:
        logger.error(f"Error in tool: {error}")
        return {"error": error}
    
    logger.info(f"Tool completed successfully")
    return {"result": result} 