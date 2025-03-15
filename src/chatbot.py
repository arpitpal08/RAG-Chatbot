from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from .rag_engine import RAGEngine
from .vector_store import VectorStore

load_dotenv()

class RAGChatbot:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the RAG-powered chatbot.
        
        Args:
            model_name (str): Name of the LLM model to use
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vector_store = VectorStore()
        self.rag_engine = RAGEngine(self.vector_store)
        self.conversation_history: List[Dict[str, str]] = []
        
        # Initialize the chat prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant powered by RAG technology. "
                      "Use the following context to answer the user's question: {context}"),
            ("human", "{question}")
        ])
        
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )
    
    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the vector store for RAG retrieval.
        
        Args:
            documents (List[str]): List of document texts to add
        """
        self.vector_store.add_documents(documents)
    
    def get_response(self, question: str) -> str:
        """Get a response from the chatbot using RAG.
        
        Args:
            question (str): User's question
            
        Returns:
            str: Chatbot's response
        """
        # Retrieve relevant context using RAG
        context = self.rag_engine.get_relevant_context(
            question, 
            self.conversation_history
        )
        
        # Generate response using the LLM
        response = self.llm_chain.predict(
            context=context,
            question=question
        )
        
        # Update conversation history
        self.conversation_history.append({
            "role": "human",
            "content": question
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = [] 