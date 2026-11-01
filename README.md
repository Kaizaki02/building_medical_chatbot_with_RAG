
# Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS

# Medical Chatbot with RAG

An AI-powered medical chatbot built using **Retrieval-Augmented Generation (RAG)** to provide context-aware answers from medical PDF documents.

## Project Overview

This project implements a Medical Chatbot that retrieves relevant information from uploaded medical documents and uses a Large Language Model (LLM) to generate responses based on the retrieved context.

The system combines document processing, text chunking, vector embeddings, semantic search, and LLM-based response generation.

## Key Features

- **PDF Processing:** Extracts text from medical PDF documents using PyPDF.
- **Text Chunking:** Splits documents into smaller chunks for efficient retrieval.
- **Vector Embeddings:** Converts text into numerical embeddings.
- **Vector Database:** Uses Pinecone to store and retrieve document embeddings.
- **Semantic Search:** Retrieves relevant medical information based on user queries.
- **LLM Integration:** Generates context-aware answers using the configured language model.
- **Web Application:** Flask-based interface for interacting with the chatbot.

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| LLM Framework | LangChain |
| Vector Database | Pinecone |
| Embeddings | Sentence Transformers / Configured embedding model |
| PDF Processing | PyPDF |
| Environment Management | Python Dotenv |

## RAG Architecture

The application follows a Retrieval-Augmented Generation pipeline:

1. **Document Loading:** Load medical PDF documents.
2. **Text Extraction:** Extract textual information from PDFs.
3. **Text Splitting:** Divide extracted text into manageable chunks.
4. **Embedding Generation:** Convert chunks into vector representations.
5. **Vector Storage:** Store embeddings in Pinecone.
6. **Query Processing:** Accept a user's medical question.
7. **Semantic Retrieval:** Retrieve relevant document chunks.
8. **Response Generation:** Pass retrieved context to the LLM to generate an answer.

## Project Structure

```text
Medical-Chatbot/
│
├── research/
│   └── trials.ipynb
│
├── src/
│   ├── __init__.py
│   ├── helpers.py
│   └── prompt.py
│
├── static/
│   └── style.css
│
├── templates/
│
├── app.py
├── store_index.py
├── setup.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Kaizaki02/building_medical_chatbot_with_RAG.git

cd building_medical_chatbot_with_RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory.

Add your required API credentials:

```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
```

Add other variables required by your implementation.

**Important:** Never upload your actual API keys or `.env` file to GitHub.

### 5. Run the Application

If your Pinecone index has not been populated, run:

```bash
python store_index.py
```

Then start the Flask application:

```bash
python app.py
```

Open the local URL displayed in your terminal.

## Future Improvements

- Improve retrieval accuracy using reranking.
- Add conversation history and memory.
- Improve document handling for multiple medical PDFs.
- Implement source citations for retrieved information.
- Deploy the application to a cloud platform.

## Disclaimer

This chatbot is an experimental information-retrieval application and is not a substitute for professional medical advice, diagnosis, or treatment.

## Author

**Pratham Kumar Rana**

[GitHub Profile](https://github.com/Kaizaki02)


