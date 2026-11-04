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
- **CI/CD:** Docker image built and pushed to AWS ECR, then deployed to AWS EC2 automatically with GitHub Actions.

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
| Containerization | Docker |
| Cloud / CI-CD | AWS (ECR, EC2), GitHub Actions |

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
├── .github/
│   └── workflows/
│       └── cicd.yaml      # GitHub Actions CI/CD pipeline
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
├── Dockerfile             # Container image definition
├── .dockerignore
├── setup.py
├── requirements.txt
├── template.sh            # Script that scaffolded the folders
└── README.md
```

## Installation and Setup (Local)

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

The index lives in Pinecone's cloud, so you do **not** need to run this again on AWS. The deployed container just connects to the existing index.

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

---

## AWS CI/CD Deployment with GitHub Actions

Every push to `main` builds the Docker image, pushes it to AWS ECR, and redeploys it on an EC2 machine that acts as a self-hosted GitHub runner.

```
GitHub push -> build Docker image -> push to ECR -> EC2 runner pulls image -> docker run
```

### Before you start

1. Run `python store_index.py` once locally so the Pinecone index exists.
2. Make sure `app.py` starts the server on all interfaces and port 8080, otherwise the container will not be reachable:

   ```python
   if __name__ == "__main__":
       app.run(host="0.0.0.0", port=8080)
   ```

### 1. Login to the AWS console

### 2. Create an IAM user for deployment

Give the user programmatic access with these policies:

```
1. AmazonEC2ContainerRegistryFullAccess   (ECR: stores your Docker image in AWS)
2. AmazonEC2FullAccess                    (EC2: the virtual machine that runs the app)
```

Create an access key for this user. You will store it in GitHub secrets (step 7), never in the code.

What the deployment does:

```
1. Build the Docker image of the source code
2. Push the Docker image to ECR
3. Launch your EC2 machine
4. Pull the image from ECR on EC2
5. Run the Docker image on EC2
```

### 3. Create an ECR repository

Create a private repo named `medicalbot` in ECR and save its URI:

```
<aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/medicalbot
```

### 4. Create the EC2 machine (Ubuntu)

- Launch an Ubuntu instance (t2.medium or larger is recommended, because the embedding model needs memory).
- In the **Security Group**, add an inbound rule: **Custom TCP, port 8080, source 0.0.0.0/0** (or restrict it to your IP).

### 5. Install Docker on the EC2 machine

Connect to the instance and run:

```bash
# optional
sudo apt-get update -y
sudo apt-get upgrade -y

# required
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### 6. Configure EC2 as a self-hosted runner

```
GitHub repo > Settings > Actions > Runners > New self-hosted runner > choose OS (Linux)
> run the commands one by one on the EC2 machine
```

Keep the runner running (the last step of the setup, `./run.sh`, or install it as a service with `sudo ./svc.sh install && sudo ./svc.sh start`).

### 7. Add GitHub secrets

`GitHub repo > Settings > Secrets and variables > Actions > New repository secret`

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Access key of the IAM user |
| `AWS_SECRET_ACCESS_KEY` | Secret key of the IAM user |
| `AWS_DEFAULT_REGION` | `us-east-1` |
| `ECR_REPO` | `medicalbot` (the repo name only, not the full URI) |
| `PINECONE_API_KEY` | Your Pinecone key |
| `GROQ_API_KEY` | Your Groq key |

### 8. Add the Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["python3", "app.py"]
```

And a `.dockerignore` so secrets and junk are not baked into the image:

```
.env
venv/
.git/
__pycache__/
research/
```

### 9. Add the workflow `.github/workflows/cicd.yaml`

```yaml
name: workflow

on:
  push:
    branches:
      - main
    paths-ignore:
      - 'README.md'

permissions:
  id-token: write
  contents: read

jobs:
  integration:
    name: Continuous Integration
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Lint code
        run: echo "Linting repository"

      - name: Run unit tests
        run: echo "Running unit tests"

  build-and-push-ecr-image:
    name: Continuous Delivery
    needs: integration
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Install Utilities
        run: |
          sudo apt-get update
          sudo apt-get install -y jq unzip

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_DEFAULT_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build, tag, and push image to Amazon ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: ${{ secrets.ECR_REPO }}
          IMAGE_TAG: latest
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

  continuous-deployment:
    name: Continuous Deployment
    needs: build-and-push-ecr-image
    runs-on: self-hosted
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_DEFAULT_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Pull latest image
        run: |
          docker pull ${{ steps.login-ecr.outputs.registry }}/${{ secrets.ECR_REPO }}:latest

      - name: Stop and remove old container
        run: |
          docker stop medicalbot || true
          docker rm medicalbot || true

      - name: Run Docker image
        run: |
          docker run -d -p 8080:8080 --name=medicalbot \
            -e PINECONE_API_KEY="${{ secrets.PINECONE_API_KEY }}" \
            -e GROQ_API_KEY="${{ secrets.GROQ_API_KEY }}" \
            ${{ steps.login-ecr.outputs.registry }}/${{ secrets.ECR_REPO }}:latest

      - name: Clean previous images and containers
        run: |
          docker system prune -f
```

### 10. Deploy

Push to `main`. Once the workflow is green, open:

```
http://<EC2-public-IP>:8080
```

### Deployment notes

- `app.py` uses `load_dotenv(override=True)`. This is fine in Docker because `.env` is excluded by `.dockerignore`, so the keys passed with `docker run -e` are the ones used.
- The first request after a fresh container start can be slow, since the embedding model is loaded (and downloaded on the first run) inside the container.
- For production, replace the Flask development server with `gunicorn` and put the app behind HTTPS (for example an Application Load Balancer or Nginx).
- Rotate your AWS access keys if they are ever exposed, and prefer a least-privilege IAM policy over the full-access policies once everything works.

---

## Troubleshooting and Lessons Learned

| Problem | Cause | Fix |
|---|---|---|
| `AttributeError: 'str' object has no attribute 'config'` | Passed the index name where an index object was expected | Use `PineconeVectorStore(index_name=..., embedding=...)` or `from_existing_index(...)` |
| Pinecone `401 Invalid API key` although `.env` was correct | An old `PINECONE_API_KEY` Windows variable took priority, since `load_dotenv()` does not override existing variables | `load_dotenv(override=True)` |
| `403: Gemini API free tier is not available for this project` | Account/project restriction on the provider, not a code bug | Switched the LLM to Groq |
| `NotFoundError` for a model name | The model was no longer offered by the provider | List the available models and use a current one |
| Page loads but chat fails with 404 | Page posted to a different URL and field name than the Flask route | Route `/api/chat`, field `query`, JSON reply |
| Duplicate chunks in retrieval | Ingestion ran more than once | Clear the index, then ingest once |
| Site not reachable on the EC2 public IP | Port 8080 not open, or app bound to `127.0.0.1` | Add the inbound rule for 8080 and use `app.run(host="0.0.0.0", port=8080)` |
| Deploy job stays "Waiting for a runner" | Self-hosted runner on EC2 is offline | Start the runner (`./run.sh` or the `svc.sh` service) |
| `docker: permission denied` in the deploy job | Runner user is not in the docker group | `sudo usermod -aG docker ubuntu`, then restart the runner |
| Container exits right after start | Missing or wrong `PINECONE_API_KEY` / `GROQ_API_KEY` | Check the GitHub secrets, then `docker logs medicalbot` |
| Container killed or very slow | Instance too small for the embedding model | Use a larger instance such as t2.medium |

## Limitations and Future Improvements

- Add source citations (store the page number as metadata and show it with each answer).
- Improve retrieval with reranking and tuned chunk size and overlap.
- Add conversation memory.
- Deploy with a production server (for example `waitress` on Windows or `gunicorn` on Linux) instead of the Flask development server.
- Add real lint and unit-test steps to the CI job (they are placeholders now).
- Add automated tests for retrieval quality.

## Disclaimer

This chatbot is an experimental information-retrieval project. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

## License

See the [LICENSE](LICENSE) file.

## Author

**Pratham Kumar Rana**
[GitHub: Kaizaki02](https://github.com/Kaizaki02)