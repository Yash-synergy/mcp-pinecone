# Custom MCP Tools for Semantic Search and Time

This package provides a set of custom MCP (Model Context Protocol) tools for semantic search with Pinecone and retrieving the current time.

## Features

- **Semantic Search**: Search for semantically similar content in your Pinecone vector database
- **Time Tool**: Get the current date and time
- **Pinecone Stats**: Get statistics about your Pinecone index
- **Read Document**: Retrieve specific documents from your Pinecone index

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Set the following environment variables:

```bash
export PINECONE_API_KEY=your_pinecone_api_key
export PINECONE_INDEX_NAME=your_pinecone_index_name
export OPENAI_API_KEY=your_openai_api_key  # Required for generating embeddings
```

Alternatively, you can create a `.env` file in the project directory with these variables.

## Starting the Server and Client

### Start the Server

Run the server using the provided run script:

```bash
python run_custom_server_copy.py
```

### Start the Client

In a separate terminal, run the client and point it to the server script:

```bash
python custom_mcp_tools_copy/client.py run_custom_server_copy.py
```

The client initiates a chat session where you can interact with the available tools through GPT-4o Mini.

## Available Tools

### 1. Semantic Search

The Semantic Search tool allows you to search for documents in your Pinecone vector database based on semantic similarity.

Parameters:
- `query` (required): The text to search for
- `top_k` (optional): Number of results to return (default: 5)
- `namespace` (optional): Pinecone namespace to search in
- `category` (optional): Filter by category
- `tags` (optional): Filter by tags
- `date_range` (optional): Filter by date range

### 2. Get Time

The Get Time tool returns the current date and time.

Parameters:
- `format` (optional): Date format string
- `timezone` (optional): Timezone (default: UTC)

### 3. Pinecone Stats

The Pinecone Stats tool provides information about your Pinecone index configuration.

Parameters:
- None required

### 4. Read Document

The Read Document tool retrieves a specific document from your Pinecone index by its ID.

Parameters:
- `document_id` (required): The ID of the document to retrieve
- `namespace` (optional): Pinecone namespace to read from

## File Structure and Description

The package contains the following files:

- **server.py**: Main MCP server implementation that handles client requests and routes to appropriate tool handlers
- **client.py**: Client application that connects to the MCP server and provides a chat interface using GPT-4o Mini
- **tool_definitions.py**: Defines the available tools with their schemas and descriptions
- **tool_handlers.py**: Contains the implementation logic for each tool
- **pinecone_client.py**: Handles interaction with Pinecone API for vector search operations
- **chunking.py**: Utility for breaking documents into chunks for vector embedding
- **utils.py**: Common utility functions used across the codebase
- **constants.py**: Configuration constants and default values
- **tool_time.py**: Implementation of the time-related tool functionality
- **.env**: Contains environment variables and API keys
- **requirements.txt**: List of Python package dependencies
- **__init__.py**: Package initialization file
- **PINECONE_README.md**: Documentation specific to Pinecone integration
- **START_HERE.md**: Getting started guide with step-by-step instructions

## Requirements

- Python 3.9+
- Pinecone account
- OpenAI account

## License

MIT 