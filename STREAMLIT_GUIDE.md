# Streamlit App Guide

## Quick Start

1. **Install Streamlit** (if not already installed):
   ```bash
   pip install streamlit
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser** - Streamlit will automatically open `http://localhost:8501`

## Features

### Main Chat Interface
- **Chat Input**: Type your questions at the bottom
- **Chat History**: All previous messages are displayed
- **Source Citations**: Click "View Retrieved Sources" to see the documents used for each answer

### Sidebar
- **Configuration**: Shows initialization status
- **About**: Information about the system
- **Example Questions**: Suggested questions to try
- **Clear Chat**: Button to clear conversation history

## Example Questions

Try these questions to test the system:

- "What causes BGP peer down issues?"
- "Why is connection slow?"
- "How to troubleshoot interface flaps?"
- "What are common root causes of network downtime?"
- "How do I fix ISIS neighbor drops?"

## How It Works

1. **Initialization**: On first load, the app:
   - Loads documents from CSV and text files
   - Creates embeddings and vector store
   - Initializes the RAG chain
   - This may take 30-60 seconds the first time

2. **Query Processing**:
   - Your question is sent to the retriever
   - Top 5 most relevant document chunks are found
   - Context is sent to the LLM with your question
   - Response is generated and displayed

3. **Source Display**:
   - Click the expander to see which documents were used
   - Helps verify the answer is based on your knowledge base

## Troubleshooting

### "RAG chain not initialized" Error
- Check that Ollama is running: `ollama list`
- Verify mistral:7b is installed: `ollama pull mistral:7b`
- Check the sidebar for specific error messages

### Slow Responses
- First query is slower (initialization)
- Subsequent queries should be faster
- Response time depends on LLM processing (typically 2-5 seconds)

### No Sources Displayed
- Sources are only shown if documents are retrieved
- Try rephrasing your question if no sources appear
- Check that data files exist in the `data/` directory

## Tips

- **Be Specific**: More specific questions get better results
- **Use Telco Terms**: Questions with terms like "BGP", "interface", "KPI" work well
- **Check Sources**: Always review sources to verify answer quality
- **Clear History**: Use the clear button if the conversation gets too long

## Keyboard Shortcuts

- `Ctrl+C` in terminal to stop Streamlit
- Browser refresh to restart (keeps chat history in session)


