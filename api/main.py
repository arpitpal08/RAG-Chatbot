from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from src.chatbot import RAGChatbot

app = FastAPI(
    title="RAG Chatbot API",
    description="API for an AI-powered chatbot using Retrieval-Augmented Generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = RAGChatbot()

class Message(BaseModel):
    content: str

class DocumentInput(BaseModel):
    documents: List[str]

@app.post("/chat")
async def chat(message: Message):
    """Get a response from the chatbot.
    
    Args:
        message (Message): User's message
        
    Returns:
        dict: Chatbot's response
    """
    try:
        response = chatbot.get_response(message.content)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents")
async def add_documents(doc_input: DocumentInput):
    """Add documents to the chatbot's knowledge base.
    
    Args:
        doc_input (DocumentInput): List of documents to add
        
    Returns:
        dict: Success message
    """
    try:
        chatbot.add_documents(doc_input.documents)
        return {"message": f"Successfully added {len(doc_input.documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_history():
    """Clear the conversation history.
    
    Returns:
        dict: Success message
    """
    try:
        chatbot.clear_history()
        return {"message": "Conversation history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 