#!/usr/bin/env python3
"""
Run script for the Custom MCP Tools Server

This script ensures the proper Python path setup before running the server.
"""
import os
import sys
import asyncio

# Add the current directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Now import the server's main function
from custom_mcp_tools_copy.server import main

if __name__ == "__main__":
    try:
        print("Starting Custom MCP Tools Server...")
        asyncio.run(main())
    except Exception as e:
        print(f"Server error: {e}")
        raise 