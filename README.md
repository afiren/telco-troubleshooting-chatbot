# LLM Network Troubleshooting

A telco network troubleshooting assistant using RAG (Retrieval-Augmented Generation) with local LLM via Ollama.

## Features

- **Data Validation**: Comprehensive tests to validate data before using LLM resources
- **RAG Pipeline**: Uses local vector store (FAISS) with Ollama embeddings
- **Telco-Specific**: Includes telco jargon and network troubleshooting knowledge

## Setup

1. Install dependencies:
```bash
pip install langchain-ollama langchain-community langchain-core pandas faiss-cpu pytest streamlit
```

2. Generate sample data (if needed):
```bash
python data_generator.py
```

3. Ensure Ollama is running with the mistral:7b model:
```bash
ollama pull mistral:7b
```

## Data Validation

Before using the LLM, validate your data to avoid consuming resources with incorrect data.

### Option 1: Run standalone validation script
```bash
# Data validation only (fast, no LLM required)
python validate_data.py

# Include LLM integration tests (requires Ollama running)
python validate_data.py --test-llm
```

This will run all validation checks and provide a detailed report.

### Option 2: Run pytest tests
```bash
# Data validation tests only
pytest test_data_validation.py -v

# Include LLM integration tests (requires Ollama running)
pytest test_data_validation.py -v -m integration
```

### Option 3: Enable validation in app.py
Run the app with validation enabled:
```bash
# Using environment variable
VALIDATE_DATA=1 python app.py

# Or using command-line flag
python app.py --validate
```

## Validation Checks

### Data Validation (No LLM Required)

1. **Data Files**: Files exist and are readable
2. **CSV Structure**: Expected columns, data rows, timestamp format, severity values
3. **Text Content**: File has content and expected keywords
4. **Document Loading**: Documents can be loaded correctly
5. **Document Splitting**: Text splitting works and produces valid chunks
6. **Data Quality**: Overall content size and diversity

### LLM Integration Tests (Requires Ollama Running)

7. **LLM Initialization**: LLM can be initialized and responds to queries
8. **Embeddings**: Embeddings can be generated (single and batch)
9. **Vector Store & Retriever**: Vector store creation and document retrieval work
10. **RAG Chain**: Full RAG pipeline works with test queries

## Quick LLM Test

Test that the LLM works with a simple query:

```bash
python test_llm_simple.py
```

This will:
1. Test LLM initialization and basic response
2. Test the full RAG chain with a sample query

## Usage

### Option 1: Streamlit Web UI (Recommended)

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

This will:
- Open a web browser with the chat interface
- Allow you to ask questions interactively
- Show retrieved sources for each answer
- Maintain chat history during the session

**Features:**
- 💬 Interactive chat interface
- 📄 View retrieved document sources
- 🔍 Real-time network issue analysis
- 💡 Example questions in sidebar
- 🗑️ Clear chat history button

### Option 2: Python API

Use the RAG chain programmatically:

```python
from app import rag_chain

# Ask a question
response = rag_chain.invoke("What causes BGP peer down issues?")
print(response)
```

## Project Structure

```
.
├── app.py                    # Main application with RAG pipeline
├── streamlit_app.py          # Streamlit web UI for chatbot
├── data_generator.py         # Generates sample telco logs
├── validate_data.py          # Standalone validation script
├── test_data_validation.py   # Pytest test suite
├── test_llm_simple.py        # Quick LLM test script
├── data/
│   ├── simulated_logs.csv    # Sample network logs
│   └── telco_manual.txt      # Telco troubleshooting manual
└── README.md                 # This file
```

## Best Practices

- Always validate data before using LLM to save resources
- Use small, high-quality datasets to respect token limits
- Include telco-specific jargon (RCA, KPIs, NSOs) in your data
- Keep chunk sizes reasonable (400 chars with 50 overlap)

