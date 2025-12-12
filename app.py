# Best Practice (from Research): 
# Use small, high-quality datasets to fine-tune indirectly via prompts. 
# For telco, include jargon like "RCA", "KPIs", "NSOs" to train the model implicitly. 
# Avoid large files to respect token limits (e.g., Mistral's ~8K tokens)

# app.py
import os
import sys
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Local vector store
from langchain_community.embeddings import HuggingFaceEmbeddings

# Optional: Validate data before using LLM
# Set VALIDATE_DATA=1 environment variable or pass --validate flag to enable
VALIDATE_DATA = os.getenv('VALIDATE_DATA', '0') == '1' or '--validate' in sys.argv

if VALIDATE_DATA:
    print("Running data validation before LLM usage...")
    from validate_data import (
        validate_data_files, validate_csv_structure, validate_text_content,
        validate_document_loading, validate_document_splitting, validate_data_quality
    )
    
    validations = [
        validate_data_files(),
        validate_csv_structure(),
        validate_text_content(),
        validate_document_loading(),
        validate_document_splitting(),
        validate_data_quality(),
    ]
    
    if not all(validations):
        print("\n❌ Data validation failed! Please fix errors before using LLM.")
        print("Run 'python validate_data.py' for detailed validation report.")
        sys.exit(1)
    print("✅ All data validations passed. Proceeding with LLM setup...\n")

# Load LLM
llm = OllamaLLM(model="mistral:7b")

# Load and prepare data for RAG
loaders = [
    CSVLoader('data/simulated_logs.csv'),
    TextLoader('data/telco_manual.txt', encoding='utf-8'),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# Embed and index (local, no API)
# embeddings = OllamaEmbeddings(model="mistral:7b")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # Explicit for laptop
)
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG pipeline for troubleshooting (LCEL style)
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