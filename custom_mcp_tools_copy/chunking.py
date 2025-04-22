"""
Text chunking utilities for semantic search.

This module provides the core chunking functionality for processing text into chunks
that can be used for semantic search.
"""
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set

logger = logging.getLogger("custom-mcp-tools-copy")

@dataclass
class Chunk:
    """A chunk of text with associated metadata."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Chunker for text documents."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize a TextChunker.
        
        Args:
            chunk_size: The target size of each chunk in characters.
            chunk_overlap: The overlap between chunks in characters.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Initialized TextChunker with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_text(
        self,
        content: str,
        document_id: str,
        metadata: Dict[str, Any] = None,
    ) -> List[Chunk]:
        """
        Chunk a text document into semantic chunks.
        
        Args:
            content: The text content to chunk.
            document_id: The ID of the document.
            metadata: Additional metadata to add to each chunk.
            
        Returns:
            A list of Chunk objects.
        """
        if not content:
            logger.warning("Empty content provided for chunking")
            return []
        
        if not metadata:
            metadata = {}
            
        logger.info(f"Chunking document {document_id} of length {len(content)}")
        
        # Simple chunking by character count with overlap
        chunks = []
        text_length = len(content)
        start = 0
        
        # Generate chunk IDs based on document ID
        chunk_counter = 0
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # If this is not the last chunk, try to break at a sentence or paragraph
            if end < text_length:
                # Look for a good breaking point within the last 20% of the chunk
                break_search_start = max(start + int(self.chunk_size * 0.8), start)
                
                # Try to find a paragraph break
                last_paragraph = content.rfind("\n\n", break_search_start, end)
                if last_paragraph != -1:
                    end = last_paragraph + 2  # Include the newlines
                else:
                    # Try to find a newline
                    last_newline = content.rfind("\n", break_search_start, end)
                    if last_newline != -1:
                        end = last_newline + 1  # Include the newline
                    else:
                        # Try to find a sentence break
                        for sentence_end in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                            last_sentence = content.rfind(sentence_end, break_search_start, end)
                            if last_sentence != -1:
                                end = last_sentence + len(sentence_end)
                                break
            
            # Extract the chunk text
            chunk_text = content[start:end].strip()
            
            # Create a unique ID for this chunk
            chunk_id = f"{document_id}_chunk_{chunk_counter}"
            chunk_counter += 1
            
            # Create chunk metadata
            chunk_metadata = {
                **metadata,
                "document_id": document_id,
                "chunk_index": chunk_counter - 1,
                "text": chunk_text,  # Include the text in metadata for retrieval
            }
            
            # Create and add the chunk
            chunk = Chunk(
                id=chunk_id,
                content=chunk_text,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            
            # Ensure progress (avoid infinite loop if overlap is too large)
            if start <= 0 or start >= text_length:
                break
        
        logger.info(f"Created {len(chunks)} chunks for document {document_id}")
        return chunks


def create_chunker() -> TextChunker:
    """
    Create a text chunker with default settings.
    
    Returns:
        A configured TextChunker instance.
    """
    return TextChunker(chunk_size=1000, chunk_overlap=200) 