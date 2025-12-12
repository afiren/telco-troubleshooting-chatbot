# How the Main Program Works

This document explains how `app.py` works step by step.

## Overview

The program implements a **RAG (Retrieval-Augmented Generation)** system for telco network troubleshooting. It uses:
- **Local LLM** (Ollama with Mistral 7B) - no external API calls
- **Vector Store** (FAISS) - for semantic search over documents
- **LangChain** - for orchestrating the pipeline

## Program Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    app.py Execution Flow                    │
└─────────────────────────────────────────────────────────────┘

1. OPTIONAL: Data Validation (if --validate flag or VALIDATE_DATA=1)
   ├─ Check data files exist
   ├─ Validate CSV structure
   ├─ Validate text content
   ├─ Test document loading
   ├─ Test document splitting
   └─ Check data quality
   └─→ Exit if validation fails

2. Initialize LLM
   └─→ OllamaLLM(model="mistral:7b")
       (Connects to local Ollama service)

3. Load Documents
   ├─ CSVLoader('data/simulated_logs.csv')
   │   └─→ Loads network logs as documents
   └─ TextLoader('data/telco_manual.txt')
       └─→ Loads troubleshooting manual as documents

4. Split Documents
   └─→ CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
       (Breaks large documents into smaller chunks for better retrieval)

5. Create Embeddings & Vector Store
   ├─ OllamaEmbeddings(model="mistral:7b")
   │   └─→ Generates vector embeddings for each document chunk
   └─ FAISS.from_documents(split_docs, embeddings)
       └─→ Creates searchable vector index

6. Create Retriever
   └─→ vectorstore.as_retriever(search_kwargs={"k": 5})
       (Returns top 5 most relevant documents for any query)

7. Build RAG Chain
   └─→ LCEL (LangChain Expression Language) pipeline:
       Question → Retriever → Format Docs → Prompt → LLM → Response
```

## Detailed Step-by-Step Explanation

### Step 1: Optional Data Validation (Lines 17-41)

```python
VALIDATE_DATA = os.getenv('VALIDATE_DATA', '0') == '1' or '--validate' in sys.argv
```

- **Purpose**: Validate data before using expensive LLM resources
- **When**: Only runs if `VALIDATE_DATA=1` or `--validate` flag is used
- **What it checks**:
  - Files exist and are readable
  - CSV has correct structure
  - Documents can be loaded
  - Data quality is sufficient
- **Result**: Exits early if validation fails

### Step 2: Initialize LLM (Line 44)

```python
llm = OllamaLLM(model="mistral:7b")
```

- **Purpose**: Create the language model instance
- **How**: Connects to local Ollama service (must be running)
- **Model**: Uses Mistral 7B (7 billion parameters)
- **Note**: This is a local model, no API calls to external services

### Step 3: Load Documents (Lines 46-53)

```python
loaders = [
    CSVLoader('data/simulated_logs.csv'),
    TextLoader('data/telco_manual.txt', encoding='utf-8'),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
```

- **Purpose**: Load all knowledge base documents
- **CSVLoader**: Converts CSV rows into document objects
- **TextLoader**: Loads plain text file as documents
- **Result**: List of document objects, each with `page_content`

### Step 4: Split Documents (Lines 54-55)

```python
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)
```

- **Purpose**: Break large documents into smaller chunks
- **Why**: 
  - Better retrieval (smaller, focused chunks)
  - Respects token limits (Mistral has ~8K token limit)
  - More precise context matching
- **Parameters**:
  - `chunk_size=400`: Maximum characters per chunk
  - `chunk_overlap=50`: Overlap between chunks (prevents losing context at boundaries)

### Step 5: Create Vector Store (Lines 57-59)

```python
embeddings = OllamaEmbeddings(model="mistral:7b")
vectorstore = FAISS.from_documents(split_docs, embeddings)
```

- **Purpose**: Create a searchable index of document embeddings
- **Embeddings**: Convert text to numerical vectors (captures semantic meaning)
- **FAISS**: Facebook AI Similarity Search - fast vector database
- **How it works**:
  1. Each document chunk is converted to an embedding vector
  2. Vectors are stored in FAISS index
  3. Similar queries will find similar vectors (semantic search)

### Step 6: Create Retriever (Line 60)

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

- **Purpose**: Create a component that retrieves relevant documents
- **k=5**: Returns top 5 most relevant documents for any query
- **How**: Uses cosine similarity to find documents with similar embeddings to the query

### Step 7: Build RAG Chain (Lines 62-77)

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(
    "You are a telco AI assistant for network troubleshooting. "
    "Use this context:\n{context}\n\nQuestion: {question}\n"
    "Answer concisely, suggest root causes and actions."
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

- **Purpose**: Create the complete RAG pipeline
- **LCEL Pipeline** (LangChain Expression Language):
  
  ```
  User Question
      ↓
  {"context": retriever | format_docs, "question": RunnablePassthrough()}
      ↓ (retrieves top 5 docs and formats them)
  prompt
      ↓ (fills template with context + question)
  llm
      ↓ (generates response)
  StrOutputParser()
      ↓ (converts to string)
  Final Answer
  ```

- **What happens when you invoke it**:
  1. **Input**: User question (e.g., "What causes BGP peer down?")
  2. **Retrieval**: Finds top 5 most relevant document chunks
  3. **Formatting**: Combines chunks into context string
  4. **Prompt**: Creates prompt with context + question
  5. **LLM**: Generates answer based on context
  6. **Output**: Returns string response

## Usage Example

```python
from app import rag_chain

# Ask a question
response = rag_chain.invoke("What causes BGP peer down issues?")
print(response)
```

**What happens internally**:

1. Query: "What causes BGP peer down issues?"
2. Retriever finds 5 relevant chunks (e.g., chunks mentioning BGP, peer, neighbor, etc.)
3. Context is formatted: "chunk1\n\nchunk2\n\nchunk3..."
4. Prompt is created:
   ```
   You are a telco AI assistant for network troubleshooting.
   Use this context:
   [retrieved chunks]
   
   Question: What causes BGP peer down issues?
   Answer concisely, suggest root causes and actions.
   ```
5. LLM generates response using the context
6. Response is returned as a string

## Key Concepts

### RAG (Retrieval-Augmented Generation)
- **Retrieval**: Finds relevant information from knowledge base
- **Augmented**: Enhances LLM with retrieved context
- **Generation**: LLM generates answer using context

### Why RAG?
- LLMs have limited knowledge (training cutoff date)
- RAG allows using up-to-date, domain-specific information
- More accurate than LLM alone for specialized domains
- Can cite sources (retrieved documents)

### Vector Embeddings
- Convert text to numerical vectors
- Similar texts have similar vectors
- Enables semantic search (not just keyword matching)
- Example: "network issue" and "connectivity problem" have similar embeddings

## Performance Considerations

- **First run**: Slow (creates embeddings and vector store)
- **Subsequent runs**: Fast (vector store is in memory)
- **Query time**: ~1-5 seconds (depends on LLM response time)
- **Memory**: Stores all embeddings in RAM (FAISS)

## Dependencies

- **Ollama**: Must be running locally with `mistral:7b` model
- **Data files**: `data/simulated_logs.csv` and `data/telco_manual.txt` must exist
- **Python packages**: langchain-ollama, langchain-community, faiss-cpu


