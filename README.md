# Medical Chatbot with RAG (LangChain, Pinecone, Groq, Flask)

A medical question-answering chatbot built with **Retrieval-Augmented Generation (RAG)**. It answers questions using information retrieved from medical PDF documents, instead of relying only on the language model's memory.

## Project Overview

The app reads medical PDFs, splits them into chunks, converts each chunk into a vector embedding, and stores the vectors in Pinecone. When a user asks a question, the app finds the most similar chunks and passes them, together with the question, to an LLM running on Groq. The LLM answers from that context. A small Flask web page provides the chat interface.

## Key Features

- **PDF ingestion:** loads PDFs from the `data/` folder with PyPDF.
- **Text chunking:** `RecursiveCharacterTextSplitter` with 500-character chunks and 20-character overlap.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), run locally.
- **Vector database:** Pinecone serverless index, cosine similarity.
- **Semantic retrieval:** top 3 most similar chunks for each question.
- **LLM:** Groq through `langchain-groq` (`ChatGroq`), with a prompt that tells the model to answer only from the retrieved context and to say "I don't know" otherwise.
- **Web app:** Flask backend with a chat page (`templates/chat.html`).

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Web framework | Flask |
| RAG framework | LangChain |
| Vector database | Pinecone |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) via `langchain-huggingface` |
| LLM | Groq (`langchain-groq`) |
| PDF processing | PyPDF |
| Config | python-dotenv |

## How It Works

```
Ingestion (run once):
  PDF -> pages -> chunks (500 chars) -> embeddings (384-d) -> Pinecone
  Each record stores: id + vector + metadata {text, source}

Question answering:
  question -> embedding -> Pinecone top-3 similar chunks
           -> chunks inserted into the prompt as {context}
           -> Groq LLM -> answer -> chat page
```

## Project Structure

```text
building_medical_chatbot_with_RAG/
│
├── data/                  # Put your medical PDF(s) here
├── research/
│   └── trials.ipynb       # Notebook used to prototype each step
├── src/
│   ├── __init__.py
│   ├── helpers.py         # PDF loading, chunking, embedding helpers
│   └── prompt.py          # System prompt for the LLM
├── static/
│   └── style.css
├── templates/
│   └── chat.html          # Chat interface
├── app.py                 # Flask app and RAG chain
├── store_index.py         # One-time ingestion into Pinecone
├── setup.py
├── requirements.txt
├── template.sh            # Script that scaffolded the folders
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Kaizaki02/building_medical_chatbot_with_RAG.git
cd building_medical_chatbot_with_RAG
```

### 2. Create and activate an environment

With conda:

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

Or with venv:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API keys

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

- Pinecone key: https://app.pinecone.io (API Keys)
- Groq key: https://console.groq.com (API Keys)

**Never commit your `.env` file.** It is listed in `.gitignore`.

### 5. Add your documents

Place one or more medical PDFs in the `data/` folder. The PDF is not redistributed with this project, so use a document you have the right to use.

### 6. Build the vector index (run once)

```bash
python store_index.py
```

This creates the `medical-chatbot` Pinecone index (384 dimensions, cosine, AWS `us-east-1`) and uploads the chunks. The first run also downloads the embedding model (about 90 MB).

> **Run this once.** Each run gives the chunks new random IDs, so running it again adds duplicate records. To re-ingest, clear the index first.

### 7. Start the app

```bash
python app.py
```

Open http://127.0.0.1:8080 and ask a question.

## Changing the LLM

The model is set in `app.py`:

```python
chatmodel = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
```

Hosted model names change over time. To see the models your key can use:

```python
from groq import Groq
import os
client = Groq(api_key=os.environ["GROQ_API_KEY"])
print([m.id for m in client.models.list().data])
```

Pick a chat model from that list (not the speech models). Because LangChain separates the retriever from the model, switching providers only changes the line that creates `chatmodel`.

## API

| Route | Method | Description |
|---|---|---|
| `/` | GET | Chat page |
| `/api/chat` | POST | Form field `query` -> JSON `{ "reply": "..." }` |

## Troubleshooting and Lessons Learned

| Problem | Cause | Fix |
|---|---|---|
| `AttributeError: 'str' object has no attribute 'config'` | Passed the index name where an index object was expected | Use `PineconeVectorStore(index_name=..., embedding=...)` or `from_existing_index(...)` |
| Pinecone `401 Invalid API key` although `.env` was correct | An old `PINECONE_API_KEY` Windows variable took priority, since `load_dotenv()` does not override existing variables | `load_dotenv(override=True)` |
| `403: Gemini API free tier is not available for this project` | Account/project restriction on the provider, not a code bug | Switched the LLM to Groq |
| `NotFoundError` for a model name | The model was no longer offered by the provider | List the available models and use a current one |
| Page loads but chat fails with 404 | Page posted to a different URL and field name than the Flask route | Route `/api/chat`, field `query`, JSON reply |
| Duplicate chunks in retrieval | Ingestion ran more than once | Clear the index, then ingest once |

## Limitations and Future Improvements

- Add source citations (store the page number as metadata and show it with each answer).
- Improve retrieval with reranking and tuned chunk size and overlap.
- Add conversation memory.
- Deploy with a production server (for example `waitress` on Windows or `gunicorn` on Linux) instead of the Flask development server.
- Add automated tests for retrieval quality.

## Disclaimer

This chatbot is an experimental information-retrieval project. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

## License

See the [LICENSE](LICENSE) file.

## Author

**Pratham Kumar Rana**
[GitHub: Kaizaki02](https://github.com/Kaizaki02)