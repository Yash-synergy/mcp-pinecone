"""
Pinecone client implementation for MCP tools.
Handles vector database operations including embedding, upserting, and searching.
"""
from pinecone import Pinecone, ServerlessSpec, FetchResponse, UpsertResponse
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import os

# Import constants
from .constants import INFERENCE_DIMENSION, PINECONE_INDEX_NAME, PINECONE_API_KEY

# Load environment variables if not already loaded
load_dotenv()

logger = logging.getLogger("pinecone-client")


class PineconeRecord(BaseModel):
    """
    Represents a record in Pinecone
    """
    id: str
    embedding: List[float]
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        """
        Convert to dictionary format for JSON serialization
        """
        return {
            "id": self.id,
            "embedding": self.embedding,
            "text": self.text,
            "metadata": self.metadata,
        }


class PineconeClient:
    """
    A client for interacting with Pinecone.
    """

    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        # Initialize index after checking/creating
        self.ensure_index_exists()
        desc = self.pc.describe_index(PINECONE_INDEX_NAME)
        self.index = self.pc.Index(
            name=PINECONE_INDEX_NAME,
            host=desc.host,  # Get the proper host from the index description
        )

    def ensure_index_exists(self):
        """
        Check if index exists, create if it doesn't.
        """
        try:
            indexes = self.pc.list_indexes()

            exists = any(index["name"] == PINECONE_INDEX_NAME for index in indexes)
            if exists:
                logger.warning(f"Index {PINECONE_INDEX_NAME} already exists")
                return

            self.create_index()

        except Exception as e:
            logger.error(f"Error checking/creating index: {e}")
            raise

    def create_index(self):
        """
        Create a serverless index with integrated inference.
        """
        try:
            return self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=INFERENCE_DIMENSION,
                metric="cosine",
                deletion_protection="disabled",  # Consider enabling for production
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for a given text.
        Note: This is a placeholder method that should be overridden by the implementing class
        to use a specific embedding model (e.g., OpenAI).
        
        Parameters:
            text: The text to generate embeddings for.
            
        Returns:
            List[float]: The embeddings for the text.
        """
        # This should be overridden by a subclass to provide the actual implementation
        # In the default implementation, we'll just raise an error
        raise NotImplementedError(
            "generate_embeddings method must be implemented by a subclass"
        )

    def upsert_records(
        self,
        records: List[PineconeRecord],
        namespace: Optional[str] = None,
    ) -> UpsertResponse:
        """
        Upsert records into the Pinecone index.

        Parameters:
            records: List of records to upsert.
            namespace: Optional namespace to upsert into.

        Returns:
            Dict[str, Any]: The response from Pinecone.
        """
        try:
            vectors = []
            for record in records:
                # Don't continue if there's no vector values
                if not record.embedding:
                    continue

                vector_values = record.embedding
                raw_text = record.text
                record_id = record.id
                metadata = record.metadata

                logger.info(f"Record: {metadata}")

                # Add raw text to metadata
                metadata["text"] = raw_text
                vectors.append((record_id, vector_values, metadata))

            return self.index.upsert(vectors=vectors, namespace=namespace)

        except Exception as e:
            logger.error(f"Error upserting records: {e}")
            raise

    def search_records(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Search records.

        Parameters:
            query: The query to search for (text or vector).
            top_k: The number of results to return.
            namespace: Optional namespace to search in.
            filter: Optional filter to apply to the search.
            include_metadata: Whether to include metadata in the search results.

        Returns:
            Dict[str, Any]: The search results from Pinecone.
        """
        try:
            # If query is text, use our custom function to get embeddings
            if isinstance(query, str):
                vector = self.generate_embeddings(query)
            else:
                vector = query

            return self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata,
                filter=filter,
            )
        except Exception as e:
            logger.error(f"Error searching records: {e}")
            raise

    def stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the index.

        Returns:
            Dict[str, Any]: A dictionary containing index statistics.
        """
        try:
            logger.info(f"Getting stats for index: {PINECONE_INDEX_NAME}")
            stats = self.index.describe_index_stats()
            logger.info(f"Raw stats response type: {type(stats).__name__}")
            logger.info(f"Stats response structure: {dir(stats) if hasattr(stats, '__dict__') else 'No attributes'}")
            
            # Convert namespaces to dict
            namespaces_dict = {}
            logger.info(f"Processing namespaces of type: {type(stats.namespaces).__name__}")
            
            try:
                for ns_name, ns_summary in stats.namespaces.items():
                    namespaces_dict[ns_name] = {
                        "vector_count": ns_summary.vector_count,
                    }
                logger.info(f"Processed {len(namespaces_dict)} namespaces")
            except Exception as ns_error:
                logger.error(f"Error processing namespaces: {ns_error}", exc_info=True)
                # Try a different approach if the above fails
                if hasattr(stats, 'namespaces') and isinstance(stats.namespaces, dict):
                    namespaces_dict = stats.namespaces
                    logger.info(f"Using raw namespaces dictionary with {len(namespaces_dict)} entries")

            result = {
                "namespaces": namespaces_dict,
                "dimension": getattr(stats, "dimension", None),
                "index_fullness": getattr(stats, "index_fullness", None),
                "total_vector_count": getattr(stats, "total_vector_count", 0),
            }
            
            logger.info(f"Returning stats with {len(namespaces_dict)} namespaces")
            return result
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            raise

    def delete_records(
        self, ids: List[str], namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete records by ID.

        Parameters:
            ids: List of record IDs to delete.
            namespace: Optional namespace to delete from.
        """
        try:
            return self.index.delete(ids=ids, namespace=namespace)
        except Exception as e:
            logger.error(f"Error deleting records: {e}")
            raise

    def fetch_records(
        self, ids: List[str], namespace: Optional[str] = None
    ) -> FetchResponse:
        """
        Fetch specific records by ID.

        Parameters:
            ids: List of record IDs to fetch.
            namespace: Optional namespace to fetch from.

        Returns:
            FetchResponse: The response from Pinecone.
        """
        try:
            return self.index.fetch(ids=ids, namespace=namespace)
        except Exception as e:
            logger.error(f"Error fetching records: {e}")
            raise

    def list_records(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List records in the index using pagination.

        Parameters:
            prefix: Optional prefix to filter records by.
            limit: The number of records to return per page.
            namespace: Optional namespace to list records from.
        """
        try:
            # Log namespace information for debugging
            try:
                stats = self.stats()
                namespaces = stats.get('namespaces', {})
                logger.info(f"Available namespaces: {list(namespaces.keys())}")
                
                # Check for case sensitivity issues
                if namespace and namespace not in namespaces:
                    # Try lowercase version of namespace if original not found
                    lowercase_namespace = namespace.lower()
                    if lowercase_namespace in namespaces:
                        logger.info(f"Using lowercase namespace '{lowercase_namespace}' instead of '{namespace}'")
                        namespace = lowercase_namespace
                    
                    # Try to find namespace with different case
                    for ns in namespaces.keys():
                        if ns.lower() == namespace.lower():
                            logger.info(f"Found matching namespace with different case: '{ns}', original: '{namespace}'")
                            namespace = ns
                            break
            except Exception as stats_error:
                logger.warning(f"Error checking namespaces: {stats_error}")
                
            # Using list_paginated for single-page results
            response = self.index.list_paginated(
                prefix=prefix, limit=limit, namespace=namespace
            )

            # Check if response is None
            if response is None:
                logger.error("Received None response from Pinecone list_paginated")
                return {"vectors": [], "namespace": namespace, "pagination_token": None}

            # Log raw response info for debugging
            logger.info(f"Raw list_paginated response type: {type(response).__name__}")
            try:
                logger.info(f"Response attributes: {dir(response)[:20] if hasattr(response, '__dir__') else 'No attributes'}")
                logger.info(f"Response has vectors attribute: {hasattr(response, 'vectors')}")
                if not hasattr(response, 'vectors'):
                    logger.warning("No vectors attribute found in response")
                    
                    # Check if response is a string (JSON)
                    if isinstance(response, str):
                        try:
                            import json
                            parsed = json.loads(response)
                            if 'vectors' in parsed:
                                logger.info("Found vectors in parsed JSON response")
                                # Process vectors from JSON
                                processed_vectors = [
                                    {"id": v.get("id"), "metadata": v.get("metadata", {})}
                                    for v in parsed.get("vectors", [])
                                ]
                                return {
                                    "vectors": processed_vectors,
                                    "namespace": namespace,
                                    "pagination_token": parsed.get("pagination_token")
                                }
                        except Exception as json_error:
                            logger.warning(f"Error parsing response as JSON: {json_error}")
            except Exception as attr_error:
                logger.warning(f"Error checking response attributes: {attr_error}")
            
            # Handle the case where vectors might be None
            vectors = getattr(response, "vectors", None)
            logger.info(f"Processed {len(vectors) if vectors else 0} vectors")
            
            processed_vectors = []
            try:
                for v in vectors or []:
                    vector_id = getattr(v, "id", None)
                    metadata = getattr(v, "metadata", {})
                    processed_vectors.append({
                        "id": vector_id,
                        "metadata": metadata,
                    })
                logger.info(f"Processed {len(processed_vectors)} vectors")
            except Exception as vector_error:
                logger.error(f"Error processing vectors: {vector_error}", exc_info=True)

            # Check pagination
            has_pagination = hasattr(response, "pagination")
            next_token = None
            if has_pagination and hasattr(response.pagination, "next"):
                next_token = response.pagination.next
                logger.info(f"Pagination token present: {bool(next_token)}")
            
            result = {
                "vectors": processed_vectors,
                "namespace": getattr(response, "namespace", namespace),
                "pagination_token": next_token,
            }
            
            logger.info(f"Returning {len(processed_vectors)} vectors")
            return result
        except Exception as e:
            logger.error(f"Error listing records: {e}", exc_info=True)
            # Return empty result instead of raising
            return {"vectors": [], "namespace": namespace, "pagination_token": None} 