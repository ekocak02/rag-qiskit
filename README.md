# ⚛️ Qiskit RAG Assistant

A robust, containerized **Retrieval-Augmented Generation (RAG)** system designed to answer questions about Quantum Computing and **Qiskit**, powered by **Google Gemini** and a local Vector Database (**ChromaDB**).

![CI Pipeline](https://github.com/ekocak02/rag-qiskit/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)

## 🚀 Key Features

*   **Advanced RAG Pipeline:** Ingests, chunks, and indexes Qiskit documentation to provide context-aware answers.
*   **Dual Interface:**
    *   **FastAPI Backend:** Robust REST API for programmatic access (`/query`).
    *   **Gradio UI:** User-friendly chat interface for interactive testing.
*   **Vector Search:** Uses **ChromaDB** for efficient similarity search with local embeddings.
*   **MLOps & DevOps:**
    *   **Dockerized:** Fully containerized services (API, UI, Worker, Database) with GPU support.
    *   **DVC Integration:** Data Version Control for managing datasets.
    *   **CI/CD:** Automated testing and build pipelines via GitHub Actions.
    *   **Code Quality:** Enforced via `flake8`, `black`, and `pytest`.

---

## 🏗️ Architecture

The system consists of four main services defined in `docker-compose.yml`:

1.  **`rag-api`**: The brain of the application. Handles user queries, retrieves context from ChromaDB, and calls the LLM (Gemini).
2.  **`rag-ui`**: A Gradio-based frontend communicating with the API.
3.  **`rag-worker`**: A dedicated service for the data pipeline (Ingestion -> Indexing -> Embedding).
4.  **`chromadb`**: The vector database server storing document embeddings.

---

## 🛠️ Installation & Setup

### Prerequisites
*   [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed.
*   (Optional) NVIDIA GPU with drivers installed for faster embedding generation.

### 1. Clone the Repository
```bash
git clone https://github.com/ekocak02/rag-qiskit.git
cd rag-qiskit
```

### 2. Configure Environment Variables
Copy the example configuration file:
```bash
cp .env.example .env
```
Open `.env` and fill in the required API keys:
*   `GOOGLE_API_KEY`: Your Gemini API key.
*   `HF_TOKEN`: HuggingFace token (if using gated models).

### 3. Run with Docker Compose
Start all services in the background:
```bash
docker compose up -d
```

Validating startup:
*   **API:** Visit `http://localhost:8000/docs` (Swagger UI).
*   **Frontend:** Visit `http://localhost:7860`.

---

## 🔄 Data Pipeline (Ingestion & Indexing)

To populate the database with Qiskit data, you need to run the pipeline. This process scrapes/loads data, chunks it, and updates ChromaDB.

You can trigger this via the `rag-worker` service:

```bash
# Run the full pipeline manually
docker compose run --rm rag-worker
```

Or execute the script directly if running locally:
```bash
./entrypoint_pipeline.sh
```

**Pipeline Steps:**
1.  **Ingestion:** Scrapes/loads raw files into `data/raw`.
2.  **Indexing:** Chunks text and saves to `data/merged/unified.jsonl`.
3.  **Embedding:** Computes embeddings and upserts to ChromaDB.

---

## 👩‍💻 Local Development

If you prefer to run the code locally (without Docker):

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Pre-commit Hooks (Recommended):**
    ```bash
    pre-commit install
    ```

4.  **Run Tests:**
    ```bash
    pytest
    ```

---

## 📂 Project Structure

```text
rag-qiskit/
├── .github/             # GitHub Actions (CI/CD)
├── .dvc/                # Data Version Control config
├── data/                # Data storage (Ignored by Git, managed by DVC)
├── src/
│   ├── api/             # FastAPI application
│   ├── ui/              # Gradio interface
│   ├── rag/             # Core RAG logic (Retriever, Generator)
│   ├── ingestion/       # Scrapers and data loaders
│   ├── indexing/        # Text chunking and processing
│   └── database/        # ChromaDB interaction
├── tests/               # Unit and integration tests
├── docker-compose.yml   # Multi-container setup
├── Dockerfile           # Unified Docker image definition
└── requirements.txt     # Python dependencies
```
