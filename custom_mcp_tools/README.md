# Custom MCP Tools

This is a custom implementation of the MCP (Model Control Protocol) that integrates with your existing tool dispatcher pattern.

## Project Structure

- `server.py`: MCP server that registers tools and handles requests
- `client.py`: Interactive client that connects to the MCP server
- Tool implementations:
  - `tool_square.py`: Squares a number
  - `tool_add.py`: Adds two numbers
  - `tool_reverse.py`: Reverses a string
  - `tool_time.py`: Returns the current time

## How It Works

This implementation follows the Model Control Protocol (MCP) while preserving your existing tool dispatcher pattern:

1. The `server.py` file:
   - Initializes and configures an MCP server
   - Defines and registers the available tools
   - Handles tool dispatch using your existing tool dispatcher pattern
   - Routes requests to the appropriate tool implementation

2. The individual tool files:
   - Each tool is a separate Python script
   - Accepts standardized JSON requests via stdin
   - Returns standardized JSON responses via stdout
   - Maintains independence between tools

3. The `client.py` file:
   - Provides an interactive interface for users
   - Connects to the MCP server
   - Handles user input and displays results

## Running the System

1. Start the server:
   ```
   python custom_mcp_tools/server.py
   ```

2. In another terminal, start the client:
   ```
   python custom_mcp_tools/client.py
   ```

3. Use the client to interact with tools:
   ```
   > square 5
   Result: 25
   
   > add 3 4
   Result: 7
   
   > reverse hello
   Result: olleh
   
   > time
   Current time: 2023-08-25 14:30:45
   ```

## Adaptation to MCP

This implementation demonstrates how to adapt your existing tool architecture to work with the MCP library while preserving the individual tool implementation pattern. 