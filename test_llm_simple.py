"""
Simple script to test LLM with a quick query.
This is a minimal test to verify LLM works before running full validation.
Usage: python test_llm_simple.py
"""
import sys

def test_llm_simple():
    """Test LLM with a simple query"""
    print("Testing LLM with simple query...")
    print("=" * 60)
    
    try:
        from langchain_ollama import OllamaLLM
        
        print("Initializing LLM...")
        llm = OllamaLLM(model="mistral:7b")
        print("✓ LLM initialized")
        
        print("\nSending test query: 'Say hello if you can read this.'")
        response = llm.invoke("Say hello if you can read this.")
        
        if response is None:
            print("❌ LLM returned None")
            return False
        elif len(response.strip()) == 0:
            print("❌ LLM returned empty response")
            return False
        else:
            print(f"✓ LLM responded successfully!")
            print(f"\nResponse: {response}")
            return True
            
    except ImportError as e:
        print(f"❌ Failed to import OllamaLLM: {e}")
        print("Install with: pip install langchain-ollama")
        return False
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running")
        print("  2. mistral:7b model is installed (run: ollama pull mistral:7b)")
        return False


def test_rag_chain_simple():
    """Test RAG chain with a simple query"""
    print("\n" + "=" * 60)
    print("Testing RAG chain with simple query...")
    print("=" * 60)
    
    try:
        # Import app components
        from langchain_ollama import OllamaLLM, OllamaEmbeddings
        from langchain_community.document_loaders import CSVLoader, TextLoader
        from langchain_text_splitters import CharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        
        print("Loading documents...")
        loaders = [
            CSVLoader('data/simulated_logs.csv'),
            TextLoader('data/telco_manual.txt', encoding='utf-8'),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        print(f"✓ Loaded {len(docs)} documents")
        
        print("Splitting documents...")
        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        split_docs = text_splitter.split_documents(docs)
        print(f"✓ Split into {len(split_docs)} chunks")
        
        print("Creating embeddings and vector store...")
        embeddings = OllamaEmbeddings(model="mistral:7b")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✓ Vector store created")
        
        print("Creating RAG chain...")
        llm = OllamaLLM(model="mistral:7b")
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        prompt = ChatPromptTemplate.from_template(
            "You are a telco AI assistant. Use this context:\n{context}\n\n"
            "Question: {question}\nAnswer briefly."
        )
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("✓ RAG chain created")
        
        print("\nTesting with query: 'What is network troubleshooting?'")
        response = rag_chain.invoke("What is network troubleshooting?")
        
        if response is None or len(response.strip()) == 0:
            print("❌ RAG chain returned empty response")
            return False
        else:
            print(f"✓ RAG chain works!")
            print(f"\nResponse:\n{response}")
            return True
            
    except Exception as e:
        print(f"❌ RAG chain test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SIMPLE LLM TEST")
    print("=" * 60 + "\n")
    
    # Test 1: Simple LLM
    llm_ok = test_llm_simple()
    
    if not llm_ok:
        print("\n❌ LLM test failed. Fix issues before testing RAG chain.")
        sys.exit(1)
    
    # Test 2: RAG chain (optional, can be skipped if data not ready)
    try:
        rag_ok = test_rag_chain_simple()
        if rag_ok:
            print("\n" + "=" * 60)
            print("✅ All tests passed! LLM is working correctly.")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n❌ RAG chain test failed.")
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n⚠️  RAG chain test skipped: {e}")
        print("Run data validation first: python validate_data.py")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ RAG chain test failed: {e}")
        sys.exit(1)

