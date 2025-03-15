from typing import List, Dict
import numpy as np
from .vector_store import VectorStore

class RAGEngine:
    def __init__(self, vector_store: VectorStore, max_context_length: int = 2000):
        """Initialize the RAG engine.
        
        Args:
            vector_store (VectorStore): Vector store instance for document retrieval
            max_context_length (int): Maximum context length in tokens
        """
        self.vector_store = vector_store
        self.max_context_length = max_context_length
    
    def get_relevant_context(
        self, 
        query: str, 
        conversation_history: List[Dict[str, str]], 
        num_chunks: int = 3
    ) -> str:
        """Get relevant context for the query using RAG.
        
        Args:
            query (str): User's query
            conversation_history (List[Dict[str, str]]): Previous conversation
            num_chunks (int): Number of document chunks to retrieve
            
        Returns:
            str: Retrieved context
        """
        # Get relevant document chunks
        relevant_chunks = self.vector_store.similarity_search(
            query,
            k=num_chunks
        )
        
        # Prepare context from relevant chunks
        context = "\n\n".join(relevant_chunks)
        
        # Add recent conversation history for context
        recent_history = self._get_recent_history(conversation_history)
        if recent_history:
            context = f"{context}\n\nRecent conversation:\n{recent_history}"
        
        return self._truncate_context(context)
    
    def _get_recent_history(
        self, 
        conversation_history: List[Dict[str, str]], 
        max_turns: int = 3
    ) -> str:
        """Get recent conversation history.
        
        Args:
            conversation_history (List[Dict[str, str]]): Full conversation history
            max_turns (int): Maximum number of conversation turns to include
            
        Returns:
            str: Formatted recent conversation history
        """
        if not conversation_history:
            return ""
        
        recent_messages = conversation_history[-max_turns*2:]  # Get last N turns (2 messages per turn)
        formatted_history = []
        
        for msg in recent_messages:
            role = "User" if msg["role"] == "human" else "Assistant"
            formatted_history.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted_history)
    
    def _truncate_context(self, context: str) -> str:
        """Truncate context to maximum length while preserving sentence boundaries.
        
        Args:
            context (str): Full context
            
        Returns:
            str: Truncated context
        """
        if len(context) <= self.max_context_length:
            return context
        
        # Split into sentences and accumulate until max length
        sentences = context.split(". ")
        truncated = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence) + 2  # Add 2 for ". "
            if current_length + sentence_length > self.max_context_length:
                break
            truncated.append(sentence)
            current_length += sentence_length
        
        return ". ".join(truncated) + "." 