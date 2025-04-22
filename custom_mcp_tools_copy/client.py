#!/usr/bin/env python3
import asyncio
import json
import sys
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack
import os

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp import ClientSession
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
print("Loading environment variables...")
load_dotenv()

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in environment variables!")
    print("Please ensure your .env file contains OPENAI_API_KEY=your_api_key_here")
    sys.exit(1)
else:
    # Print a masked version of the key for debugging
    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
    print(f"OpenAI API key loaded: {masked_key}")

# Initialize OpenAI client
try:
    client = OpenAI(api_key=api_key)
    print("OpenAI client initialized successfully")
except Exception as e:
    print(f"ERROR initializing OpenAI client: {e}")
    sys.exit(1)

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
    
    server_script_path = sys.argv[1]
    exit_stack = AsyncExitStack()
    
    # Determine the server type
    is_python = server_script_path.endswith('.py')
    is_js = server_script_path.endswith('.js')
    
    if not (is_python or is_js):
        raise ValueError("Server script must be a .py or .js file")

    command = "python" if is_python else "node"
    server_params = StdioServerParameters(
        command=command,
        args=[server_script_path],
        env=None
    )

    print(f"Connecting to stdio MCP server with command: {command} {server_script_path}")
    
    # Start the server
    try:
        stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport
        
        # Create MCP client session
        session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        
        # Initialize the session
        await session.initialize()
        
        # List available tools
        response = await session.list_tools()
        tools = response.tools
        print("\nConnected to MCP server")
        print("Available tools:", [tool.name for tool in tools])
        
        # Create function schemas for OpenAI from MCP tools
        function_schemas = []
        for tool in tools:
            function_schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"Call the {tool.name} tool",
                    "parameters": tool.inputSchema
                }
            })
        
        print("\nStarting chat session. Type 'exit' to quit.")
        
        while True:
            try:
                user_input = input("\n💬 User: ")
                if user_input.lower() == 'exit':
                    break
                
                # Initialize conversation with user query
                messages = [{"role": "user", "content": user_input}]
                
                # Process the conversation until we get a final answer without function calls
                while True:
                    # Ask LLM
                    print("Calling GPT-4o Mini...")
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            tools=function_schemas,
                            tool_choice="auto"
                        )
                        print("Response received from GPT-4o Mini")
                    except Exception as e:
                        print(f"ERROR calling OpenAI API: {e}")
                        break
                    
                    message = response.choices[0].message
                    print(f"DEBUG: Message object: {message}")
                    
                    # Check if LLM wants to call a tool
                    if message.tool_calls:
                        tool_call = message.tool_calls[0]
                        tool_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        
                        print(f"[Calling tool {tool_name} with args {arguments}]")
                        
                        # Call the MCP tool
                        result = await session.call_tool(tool_name, arguments)
                        result_text = result.content[0].text if result.content else "No result"
                        parsed_result = json.loads(result_text)
                        result_value = parsed_result.get('result', result_text)
                        
                        # Add the assistant's message with the tool call
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }
                            ]
                        })
                        
                        # Add the tool response
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result_value)
                        })
                        
                        # Continue the loop to let LLM respond to the tool result
                        continue
                    
                    # If we reach here, we have a direct response (no tool call)
                    if message.content is not None:
                        print("💬 Assistant:", message.content)
                    else:
                        print("DEBUG: Received empty content from LLM")
                    
                    # Break out of the conversation loop
                    break
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    finally:
        # Clean up resources
        await exit_stack.aclose()
        print("Disconnected from server")

if __name__ == "__main__":
    asyncio.run(main()) 