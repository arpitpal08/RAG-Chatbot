# AI-Powered RAG Chatbot

A sophisticated chatbot implementation using Retrieval-Augmented Generation (RAG) for enhanced query accuracy and contextual responses.

## Features

- **RAG Implementation**: Utilizes LangChain and ChromaDB for efficient document retrieval and context augmentation
- **Advanced LLM Integration**: Leverages state-of-the-art language models for natural conversations
- **Vector Database**: Implements ChromaDB for efficient similarity search and document retrieval
- **Context Management**: Smart context window management for maintaining conversation coherence
- **API Integration**: RESTful API endpoints for easy integration with other applications

## Tech Stack

- Python 3.9+
- LangChain
- ChromaDB
- FastAPI
- Hugging Face Transformers
- PyTorch

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arpitpal20/RAG-Chatbot.git
cd RAG-Chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Usage

1. Start the server:
```bash
python main.py
```

2. Access the API at `http://localhost:8000`

3. Use the chatbot through:
- Web interface: `http://localhost:8000/docs`
- API endpoints
- Python client

## Project Structure

```
RAG-Chatbot/
├── src/
│   ├── __init__.py
│   ├── chatbot.py        # Core chatbot implementation
│   ├── rag_engine.py     # RAG implementation
│   ├── vector_store.py   # ChromaDB integration
│   └── utils.py          # Utility functions
├── api/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   └── routes.py        # API endpoints
├── tests/
│   ├── __init__.py
│   ├── test_chatbot.py
│   └── test_rag.py
├── docs/                # Documentation
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Arpit Pal - [@arpitpal696969](https://twitter.com/arpitpal696969) - palarpit894@gmail.com

Project Link: [https://github.com/arpitpal20/RAG-Chatbot](https://github.com/arpitpal20/RAG-Chatbot) 