# Rerankers API

This is a FastAPI-based REST API that provides reranking capabilities using multiple models from the [rerankers](https://github.com/AnswerDotAI/rerankers) Python library. The API allows users to rank documents based on a query and metadata, with support for different reranking models.

## Supported Rerankers

The API currently supports the following reranker models:

1. **FlashRank** (`flashrank`): 
   A fast ONNX-optimized reranker model that is lightweight and efficient, suitable for CPU-based inference.

2. **Colbert** (`colbert`):
   A powerful reranking model based on transformer models that can encode and rerank documents based on queries with high accuracy. 

By default, the API uses `flashrank` unless otherwise specified in the environment variable.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- `pip` for package management

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/rerankers-api.git
cd rerankers-api
```

### Step 2: Create and activate a virtual environment

Create a virtual environment to isolate your dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies

The dependencies are dynamically installed based on the reranker model you choose. You can set the environment variable `RANKER_MODEL` to configure which model you want to use.

- **For FlashRank** (default):
  
  No need to change anything, simply run the application.

- **For Colbert**:
  
  Set the environment variable before running the application:

```bash
  export RANKER_MODEL="colbert"
```

### Step 4: Start the API

To start the FastAPI server:

```bash
uvicorn rest.main:app --reload
```

## Configuration

The API behavior can be configured using environment variables:

- **RANKER_MODEL**: Specifies the reranker model to use. Supported values are:
  - `"flashrank"` (default)
  - `"colbert"`

Example of setting the environment variable for a specific model:

```bash
export RANKER_MODEL="colbert"
```

## Dependency Management

The project dynamically installs dependencies based on the reranker model configured via `RANKER_MODEL`. The dependencies for each model are:

- **flashrank**: Installs the `rerankers[flashrank]` dependency.
- **colbert**: Installs the `rerankers[transformers]` dependency.

If you're adding new rerankers to the project, ensure the dependencies are added in the code to be installed dynamically.

## Run with docker

You can start the API over docker using the following command:
```
docker run -p 8000:8000 rerankers-api:0.5.3
```
Notice, the image with the main version tag contains ALL dependencies for all rerankers which are supported. There are also images which do contain only the needed dependencies for one reranker type. See [https://hub.docker.com/r/gabauer/rerankers-api](https://hub.docker.com/r/gabauer/rerankers-api/tags) for further information. 

## Testing the API

You can test the API using `curl` or any HTTP client (such as Postman).

### Example `curl` Request

```bash
curl -X POST "http://127.0.0.1:8000/rerank" \
-H "Content-Type: application/json" \
-d '{
  "query": "best programming languages",
  "documents": [
    "Python is great for data science.",
    "JavaScript is great for web development.",
    "Java is widely used in enterprise applications."
  ],
  "metadata": [
    {"source": "article1"},
    {"source": "article2"},
    {"source": "article3"}
  ]
}'
