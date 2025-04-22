# Custom MCP Tools for Semantic Search and Time

This package provides a set of custom MCP (Machine Chat Protocol) tools for semantic search with Pinecone and retrieving the current time.

## Features

- **Semantic Search**: Search for semantically similar content in your Pinecone vector database
- **Time Tool**: Get the current date and time

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

## Starting the Server

Run the server:

```bash
python -m custom_mcp_tools.server
```

## Available Tools

### Semantic Search

The Semantic Search tool allows you to search for documents in your Pinecone vector database based on semantic similarity.

Parameters:
- `query` (required): The text to search for
- `top_k` (optional): Number of results to return (default: 5)
- `namespace` (optional): Pinecone namespace to search in
- `category` (optional): Filter by category
- `tags` (optional): Filter by tags
- `date_range` (optional): Filter by date range

### Get Time

The Get Time tool returns the current date and time.

Parameters:
- `format` (optional): Date format string
- `timezone` (optional): Timezone (default: UTC)

## Requirements

- Python 3.9+
- Pinecone account
- OpenAI account

## License

MIT 