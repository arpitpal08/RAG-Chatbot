from typing import List
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self, collection_name: str = "rag_documents"):
        """Initialize the vector store using ChromaDB.
        
        Args:
            collection_name (str): Name of the ChromaDB collection
        """
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./data/chroma"
        ))
        
        # Initialize the embedding function (using Sentence Transformers)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents (List[str]): List of document texts to add
        """
        # Generate unique IDs for documents
        doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Add documents to collection
        self.collection.add(
            documents=documents,
            ids=doc_ids
        )
    
    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """Perform similarity search to retrieve relevant documents.
        
        Args:
            query (str): Query text
            k (int): Number of documents to retrieve
            
        Returns:
            List[str]: List of relevant document texts
        """
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Extract and return the document texts
        return results["documents"][0]
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_function
        ) 