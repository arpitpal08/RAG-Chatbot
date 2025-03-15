import pytest
from src.chatbot import RAGChatbot
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine

def test_chatbot_initialization():
    """Test chatbot initialization."""
    chatbot = RAGChatbot()
    assert isinstance(chatbot.vector_store, VectorStore)
    assert isinstance(chatbot.rag_engine, RAGEngine)
    assert len(chatbot.conversation_history) == 0

def test_add_documents():
    """Test adding documents to the chatbot."""
    chatbot = RAGChatbot()
    documents = [
        "This is a test document about AI.",
        "RAG systems combine retrieval with generation."
    ]
    chatbot.add_documents(documents)
    # Verify documents were added (indirect test through a query)
    response = chatbot.get_response("What is RAG?")
    assert response != ""
    assert len(chatbot.conversation_history) == 2

def test_conversation_history():
    """Test conversation history management."""
    chatbot = RAGChatbot()
    question = "What is artificial intelligence?"
    response = chatbot.get_response(question)
    
    assert len(chatbot.conversation_history) == 2
    assert chatbot.conversation_history[0]["role"] == "human"
    assert chatbot.conversation_history[0]["content"] == question
    assert chatbot.conversation_history[1]["role"] == "assistant"
    assert chatbot.conversation_history[1]["content"] == response

def test_clear_history():
    """Test clearing conversation history."""
    chatbot = RAGChatbot()
    chatbot.get_response("Test question")
    assert len(chatbot.conversation_history) > 0
    
    chatbot.clear_history()
    assert len(chatbot.conversation_history) == 0 