# Pinecone Integration for MCP Tools

This integration allows you to use Pinecone vector database with your MCP tools server. It provides semantic search, document storage, and document retrieval capabilities.

## Features

- Smart document chunking with token awareness
- Semantic search using Pinecone
- Document storage and retrieval with metadata
- Namespace support for organizing documents
- Embedding generation using OpenAI

## Setup

1. Install the required Python packages:
   ```
   pip install pinecone-client openai tiktoken python-dotenv pydantic
   ```

2. Configure the environment variables:
   ```
   PINECONE_API_KEY=your_api_key
   PINECONE_INDEX_NAME=your_index_name
   OPENAI_API_KEY=your_openai_api_key
   ```

   You can set these in a `.env` file in the `custom_mcp_tools` directory or set them in your environment.

## Available Tools

### semantic-search
Search for documents in Pinecone by semantic similarity.
```
Arguments:
- query: The search query text
- top_k: Number of results to return (default: 10)
- namespace: Optional namespace to search in
- category: Optional category filter
- tags: Optional list of tags to filter by
- date_range: Optional date range filter
```

### read-document
Read a document from Pinecone by ID.
```
Arguments:
- document_id: The ID of the document to read
- namespace: Optional namespace to read from
```

### process-document
Process a document, chunk it, generate embeddings, and store it in Pinecone.
```
Arguments:
- document_id: ID for the document
- text: The document text content
- metadata: Metadata for the document (object)
- namespace: Optional namespace to store in
```

### list-documents
List all documents in a namespace.
```
Arguments:
- namespace: The namespace to list documents from
```

### pinecone-stats
Get statistics about the Pinecone index.
```
Arguments: None
```

## Implementation Details

- `chunking.py` - Smart chunking implementation with token awareness
- `pinecone_client.py` - Pinecone client for vector database operations
- `constants.py` - Configuration constants
- `utils.py` - Utility functions and error handlers

The system uses OpenAI's text-embedding-ada-002 model for generating embeddings by default. The `CustomPineconeClient` class in `server.py` overrides the embedding generation to use OpenAI.

## Example Usage

1. Start the MCP server:
   ```
   python custom_mcp_tools/server.py
   ```

2. Use the client to interact with the server:
   ```
   python custom_mcp_tools/client.py
   ```

3. Example questions:
   - "Search my documents for information about machine learning"
   - "Store this article: {content}"
   - "What documents do I have in the 'research' namespace?" 