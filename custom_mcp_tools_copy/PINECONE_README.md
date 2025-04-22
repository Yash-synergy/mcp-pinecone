# Pinecone Integration for Semantic Search

This document explains how to set up and use the Pinecone integration for the custom MCP tools.

## Prerequisites

1. A Pinecone account (sign up at [pinecone.io](https://www.pinecone.io/))
2. An OpenAI API key for generating embeddings

## Setup

1. Create a Pinecone index with the following configuration:
   - Dimensions: 1536 (for OpenAI embeddings)
   - Metric: cosine
   - Pod type: p1 or s1 (depending on your needs)

2. Set the following environment variables:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_pinecone_index_name
   OPENAI_API_KEY=your_openai_api_key
   ```

## Semantic Search

The semantic search tool allows you to search your Pinecone index for semantically similar content.

Example usage:
```json
{
  "name": "semantic-search",
  "arguments": {
    "query": "What are the safety procedures for engine maintenance?",
    "top_k": 5,
    "namespace": "vessel_manuals"
  }
}
```

Parameters:
- `query` (required): The search query text
- `top_k`: Number of results to return (default: 5)
- `namespace`: The Pinecone namespace to search in
- `category`: Filter by category
- `tags`: Filter by tags
- `date_range`: Filter by date range

## Namespaces

Pinecone namespaces allow you to segment your data. Common namespaces you might use:
- `vessel_manuals`: Technical documentation
- `policies`: Company policies
- `reports`: Safety reports
- `general`: General information

## Troubleshooting

If you encounter issues:
1. Verify your API keys are correct
2. Check that your index exists and is properly configured
3. Ensure your namespaces are created and populated
4. Verify you have enough permission to access the index 