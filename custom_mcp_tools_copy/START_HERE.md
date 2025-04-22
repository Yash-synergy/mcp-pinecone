# Getting Started with Custom MCP Tools

This package provides a simplified set of MCP tools focused on semantic search with Pinecone and a time utility.

## Quick Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your environment variables by creating a `.env` file:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_pinecone_index_name
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Start the server:
   ```bash
   python -m custom_mcp_tools.server
   ```

## Available Tools

1. **Semantic Search** - Search your vector database using natural language queries:
   ```json
   {
     "name": "semantic-search",
     "arguments": {
       "query": "Your search query here",
       "top_k": 3,
       "namespace": "optional_namespace"
     }
   }
   ```

2. **Get Time** - Get the current date and time:
   ```json
   {
     "name": "get-time",
     "arguments": {}
   }
   ```

## Pinecone Configuration

This tool requires a Pinecone account and index. The index should:
- Be configured with a dimension of 1536 (for OpenAI embeddings)
- Have appropriate namespaces for your data

## Adding Content

To add content to your vector database, you'll need to use the Pinecone Python SDK directly. This simplified package only provides search capabilities.

## Troubleshooting

If you encounter issues:
- Check that your environment variables are set correctly
- Verify your Pinecone index exists and is accessible
- Ensure your OpenAI API key is valid

## Running the Project

### Prerequisites
- Python 3.8 or higher
- Required packages:
  ```
  pip install mcp-python openai python-dotenv
  ```
- OpenAI API key in a `.env` file:
  ```
  OPENAI_API_KEY=your_api_key_here
  ```

### Step 1: Start the MCP Server
Open a terminal and run:

```
python custom_mcp_tools/server.py
```

This will start the MCP server that exposes several tools:
- `square`: Squares a number
- `add`: Adds two numbers
- `reverse-string`: Reverses a string
- `get-time`: Returns the current time

### Step 2: Connect with the Client
In another terminal, run:

```
python custom_mcp_tools/client.py custom_mcp_tools/server.py
```

### Step 3: Interact with GPT-4o Mini
Once connected, you can ask questions or give instructions in natural language. Examples:
- "What's 5 squared?"
- "Add 3 and 7 for me"
- "Can you reverse the word 'hello'?"
- "What time is it right now?"
- Type 'exit' to quit the client

Behind the scenes:
1. Your query is sent to GPT-4o Mini
2. GPT-4o Mini decides which tool to call (if any)
3. The client executes the tool call via the MCP server
4. GPT-4o Mini formulates a response using the tool's result
5. You see the final answer

## Integrating with Claude Desktop

To add this MCP server to Claude Desktop:
1. Open Claude Desktop
2. Go to File → Settings → Developer → Edit config
3. Add the following to the config file:

```json
{
  "mcpServers": {
    "custom-tools": {
      "command": "python",
      "args": ["<FULL_PATH>/custom_mcp_tools/server.py"]
    }
  }
}
```

Replace `<FULL_PATH>` with the actual path to your server file.

4. Restart Claude Desktop
5. Use the tools by typing "/custom-tools" in the chat

## How It Works

This implementation demonstrates:
1. **MCP Server**: Defines tools and handles requests
2. **Tool Implementations**: Separate Python scripts for each tool
3. **MCP Client**: Connects to the server and executes tools
4. **LLM Integration**: Uses GPT-4o Mini to decide which tools to call based on user input

The flow is:
1. User types a question or instruction
2. GPT-4o Mini analyzes it and determines if a tool call is needed
3. If yes, it selects the appropriate tool and arguments
4. The client calls the tool via the MCP server
5. The result is sent back to GPT-4o Mini
6. GPT-4o Mini formulates a natural language response
7. The answer is displayed to the user

## What's Next?

This implementation serves as a starting point for building more sophisticated AI agents that can interact with tools. You can:
1. Add new tools to the server.py file
2. Modify existing tools to perform more complex operations
3. Connect to external APIs or databases
4. Build multi-step workflows that chain multiple tool calls together

Enjoy experimenting with MCP and GPT-4o Mini! 