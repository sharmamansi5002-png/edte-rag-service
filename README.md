# Semantic GraphRAG Engine

A hybrid Retrieval-Augmented Generation (RAG) system that combines vector-based document retrieval with knowledge graph expansion to answer questions using document context and entity relationships.

## Features

- Ingest Markdown and PDF documents.
- Split documents into searchable chunks.
- Extract entities and relationships from documents.
- Store document and graph data in Neo4j.
- Retrieve relevant chunks and expand context through graph relationships.
- Generate answers using a language model.
- Expose functionality through a FastAPI REST API.
- Provide a Streamlit user interface for document uploads and questions.

## Technology Stack

- Python
- FastAPI
- Streamlit
- Neo4j
- Vector search
- Google Gemini
- Pytest
- Docker and Docker Compose

## Project Structure

```text
edte-rag-service/
├── docs/
├── src/
├── tests/
├── ui/
│   └── app.py
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11 or later
- Neo4j
- A Google Gemini API key, if required by your configured language-model provider
- Docker Desktop (for containerized deployment)

## Installation

Clone or open the project, then create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file in the project root using `.env.example` as a template.

Configure the Neo4j connection and the language-model credentials required by your application.

Do not commit `.env` or publish API keys and passwords.

## Running the Application

### 1. Start Neo4j

Start your Neo4j database and make sure it is accessible using the connection settings in `.env`.

### 2. Start the FastAPI backend

```powershell
python -m uvicorn src.api.main:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 3. Start the Streamlit interface

Open another terminal, activate the virtual environment, and run:

```powershell
python -m streamlit run ui/app.py
```

The interface is available at:

```text
http://localhost:8501
```

## Using the CLI

Ask a question using the command-line interface:

```powershell
python -m src.cli ask --query "What information is available about the Finance System?"
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check application and database health |
| POST | `/api/v1/rag/ingest` | Upload and index a document |
| POST | `/api/v1/rag/query` | Ask a question using GraphRAG |

## Running Tests

Run the test suite:

```powershell
python -m pytest -v
```

## Docker Deployment

Make sure Docker Desktop is running and the required environment variables are configured.

Build and start the services:

```powershell
docker compose up --build
```

Stop the services:

```powershell
docker compose down
```

To stop the services and remove the Neo4j data volume:

```powershell
docker compose down -v
```

**Warning:** The `-v` option deletes the persisted Neo4j database volume.

## System Overview

The application follows a hybrid retrieval workflow:

1. Documents are uploaded and processed.
2. Text is split into chunks.
3. Entities and relationships are extracted.
4. Document and graph information is stored in Neo4j.
5. Relevant chunks are retrieved for a user question.
6. Related graph information is added to the retrieval context.
7. The language model generates an answer using the retrieved context.
8. The answer and supporting sources are returned to the user.