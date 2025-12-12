"""
Data validation module to check data before using LLM.
Run this before app.py to ensure data is correct.
Usage: python validate_data.py [--test-llm]
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter


def validate_data_files():
    """Validate that data files exist and are readable"""
    print("=" * 60)
    print("STEP 1: Validating data files exist and are readable")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Check CSV file
    csv_path = Path('data/simulated_logs.csv')
    if not csv_path.exists():
        errors.append(f"CSV file not found: {csv_path}")
    else:
        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                errors.append("CSV file is empty")
            else:
                print(f"✓ CSV file found with {len(df)} rows")
        except Exception as e:
            errors.append(f"Failed to read CSV file: {e}")
    
    # Check text file
    text_path = Path('data/telco_manual.txt')
    if not text_path.exists():
        errors.append(f"Text file not found: {text_path}")
    else:
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if len(content) == 0:
                errors.append("Text file is empty")
            else:
                print(f"✓ Text file found with {len(content)} characters")
        except Exception as e:
            errors.append(f"Failed to read text file: {e}")
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✓ All data files validated\n")
    return True


def validate_csv_structure():
    """Validate CSV structure and content"""
    print("=" * 60)
    print("STEP 2: Validating CSV structure and content")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    try:
        df = pd.read_csv('data/simulated_logs.csv')
        
        # Check expected columns
        expected_columns = {
            'timestamp', 'pri', 'severity', 'facility', 'hostname',
            'alert_type', 'kpi_value', 'description', 'recommended_action'
        }
        actual_columns = set(df.columns)
        missing = expected_columns - actual_columns
        if missing:
            errors.append(f"Missing columns: {missing}")
        else:
            print(f"✓ All expected columns present: {len(expected_columns)} columns")
        
        # Check data rows
        if len(df) < 10:
            warnings.append(f"CSV has only {len(df)} rows, expected at least 10")
        else:
            print(f"✓ CSV has {len(df)} data rows")
        
        # Check critical fields are not empty
        critical_fields = ['timestamp', 'severity', 'hostname', 'alert_type', 'description']
        for field in critical_fields:
            if field not in df.columns:
                errors.append(f"Critical field '{field}' missing")
            else:
                empty_count = df[field].isna().sum() + (df[field] == '').sum()
                if empty_count > 0:
                    errors.append(f"Field '{field}' has {empty_count} empty/null values")
                else:
                    print(f"✓ Field '{field}' has no empty values")
        
        # Validate timestamp format
        try:
            pd.to_datetime(df['timestamp'], errors='raise')
            print("✓ Timestamp format is valid")
        except Exception as e:
            errors.append(f"Invalid timestamp format: {e}")
        
        # Validate severity values
        valid_severities = {
            'Emergency', 'Alert', 'Critical', 'Error', 
            'Warning', 'Notice', 'Info', 'Debug'
        }
        invalid = set(df['severity'].unique()) - valid_severities
        if invalid:
            errors.append(f"Invalid severity values: {invalid}")
        else:
            print(f"✓ All severity values are valid")
        
    except Exception as e:
        errors.append(f"Failed to validate CSV structure: {e}")
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✓ CSV structure validated\n")
    return True


def validate_text_content():
    """Validate text file content"""
    print("=" * 60)
    print("STEP 3: Validating text file content")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    try:
        with open('data/telco_manual.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content) < 100:
            warnings.append(f"Text file is short ({len(content)} chars)")
        else:
            print(f"✓ Text file has {len(content)} characters")
        
        # Check for expected keywords
        expected_keywords = ['troubleshooting', 'network', 'KPI', 'RCA']
        content_lower = content.lower()
        missing = [kw for kw in expected_keywords if kw.lower() not in content_lower]
        if missing:
            warnings.append(f"Missing expected keywords: {missing}")
        else:
            print("✓ Text file contains expected telco keywords")
        
    except Exception as e:
        errors.append(f"Failed to validate text content: {e}")
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✓ Text content validated\n")
    return True


def validate_document_loading():
    """Validate that documents can be loaded"""
    print("=" * 60)
    print("STEP 4: Validating document loading")
    print("=" * 60)
    
    errors = []
    
    try:
        loaders = [
            CSVLoader('data/simulated_logs.csv'),
            TextLoader('data/telco_manual.txt', encoding='utf-8'),
        ]
        docs = []
        for loader in loaders:
            loaded_docs = loader.load()
            if len(loaded_docs) == 0:
                errors.append(f"{loader.__class__.__name__} returned no documents")
            else:
                print(f"✓ {loader.__class__.__name__} loaded {len(loaded_docs)} documents")
                docs.extend(loaded_docs)
        
        if len(docs) == 0:
            errors.append("No documents loaded from any loader")
        else:
            print(f"✓ Total documents loaded: {len(docs)}")
        
        # Check document content
        empty_docs = [i for i, doc in enumerate(docs) if len(doc.page_content) == 0]
        if empty_docs:
            errors.append(f"Found {len(empty_docs)} documents with empty content")
        else:
            print("✓ All documents have content")
        
    except Exception as e:
        errors.append(f"Failed to load documents: {e}")
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✓ Document loading validated\n")
    return True


def validate_document_splitting():
    """Validate that document splitting works"""
    print("=" * 60)
    print("STEP 5: Validating document splitting")
    print("=" * 60)
    
    errors = []
    
    try:
        loaders = [
            CSVLoader('data/simulated_logs.csv'),
            TextLoader('data/telco_manual.txt', encoding='utf-8'),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        
        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        
        if len(split_docs) == 0:
            errors.append("Text splitter returned no chunks")
        else:
            print(f"✓ Documents split into {len(split_docs)} chunks")
        
        # Check chunk sizes
        max_chunk_size = max(len(doc.page_content) for doc in split_docs) if split_docs else 0
        if max_chunk_size > 600:
            errors.append(f"Chunks too large (max: {max_chunk_size}), expected <= 600")
        else:
            print(f"✓ Maximum chunk size: {max_chunk_size} characters")
        
        # Check that splitting occurred
        if len(split_docs) < len(docs):
            errors.append("Expected more chunks after splitting")
        else:
            print(f"✓ Splitting successful (from {len(docs)} to {len(split_docs)} chunks)")
        
    except Exception as e:
        errors.append(f"Failed to split documents: {e}")
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✓ Document splitting validated\n")
    return True


def validate_data_quality():
    """Validate overall data quality"""
    print("=" * 60)
    print("STEP 6: Validating overall data quality")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    try:
        loaders = [
            CSVLoader('data/simulated_logs.csv'),
            TextLoader('data/telco_manual.txt', encoding='utf-8'),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        
        total_chars = sum(len(doc.page_content) for doc in docs)
        if total_chars < 1000:
            warnings.append(f"Total content small ({total_chars} chars), may not be enough for RAG")
        else:
            print(f"✓ Total content size: {total_chars} characters")
        
        # Check data diversity
        csv_loader = CSVLoader('data/simulated_logs.csv')
        csv_docs = csv_loader.load()
        if len(csv_docs) > 1:
            descriptions = [doc.page_content for doc in csv_docs[:10]]
            unique_count = len(set(descriptions))
            if unique_count <= 1:
                warnings.append("Data lacks diversity - all descriptions are identical")
            else:
                print(f"✓ Data diversity: {unique_count} unique descriptions in sample")
        
    except Exception as e:
        errors.append(f"Failed to validate data quality: {e}")
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✓ Data quality validated\n")
    return True


def validate_llm_initialization():
    """Validate that LLM can be initialized and responds"""
    print("=" * 60)
    print("STEP 7: Validating LLM initialization")
    print("=" * 60)
    
    errors = []
    
    try:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model="mistral:7b")
        print("✓ LLM initialized")
        
        # Test with a simple query
        response = llm.invoke("Respond with just the word 'OK'")
        if response is None:
            errors.append("LLM returned None")
        elif len(response.strip()) == 0:
            errors.append("LLM returned empty response")
        else:
            print(f"✓ LLM responds correctly (response length: {len(response)} chars)")
            
    except ImportError as e:
        errors.append(f"Failed to import OllamaLLM: {e}. Install langchain-ollama")
    except Exception as e:
        errors.append(f"LLM initialization failed: {e}. Make sure Ollama is running with mistral:7b")
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✓ LLM initialization validated\n")
    return True


def validate_embeddings():
    """Validate that embeddings can be generated"""
    print("=" * 60)
    print("STEP 8: Validating embeddings generation")
    print("=" * 60)
    
    errors = []
    
    try:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model="mistral:7b")
        print("✓ Embeddings initialized")
        
        # Test single embedding
        test_text = "network troubleshooting"
        embedding = embeddings.embed_query(test_text)
        if embedding is None:
            errors.append("Embedding is None")
        elif len(embedding) == 0:
            errors.append("Embedding is empty")
        else:
            print(f"✓ Single embedding generated (dimension: {len(embedding)})")
        
        # Test batch embeddings
        test_texts = ["network", "troubleshooting", "telco"]
        embeddings_list = embeddings.embed_documents(test_texts)
        if len(embeddings_list) != len(test_texts):
            errors.append(f"Expected {len(test_texts)} embeddings, got {len(embeddings_list)}")
        else:
            print(f"✓ Batch embeddings generated ({len(embeddings_list)} embeddings)")
            
    except ImportError as e:
        errors.append(f"Failed to import OllamaEmbeddings: {e}. Install langchain-ollama")
    except Exception as e:
        errors.append(f"Embeddings generation failed: {e}. Make sure Ollama is running with mistral:7b")
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✓ Embeddings validated\n")
    return True


def validate_vectorstore_and_retriever():
    """Validate vector store creation and retriever functionality"""
    print("=" * 60)
    print("STEP 9: Validating vector store and retriever")
    print("=" * 60)
    
    errors = []
    
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS
        
        # Load and split documents
        loaders = [
            CSVLoader('data/simulated_logs.csv'),
            TextLoader('data/telco_manual.txt', encoding='utf-8'),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        
        # Create embeddings and vector store
        embeddings = OllamaEmbeddings(model="mistral:7b")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        print("✓ Vector store created")
        
        # Test search
        results = vectorstore.similarity_search("network", k=2)
        if len(results) == 0:
            errors.append("Vector store search returned no results")
        else:
            print(f"✓ Vector store search works (found {len(results)} results)")
        
        # Test retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        retrieved = retriever.invoke("BGP peer down")
        if len(retrieved) == 0:
            errors.append("Retriever returned no results")
        elif len(retrieved) > 5:
            errors.append(f"Retriever returned too many results: {len(retrieved)}")
        else:
            print(f"✓ Retriever works (retrieved {len(retrieved)} documents)")
            # Check document quality
            empty_docs = [i for i, doc in enumerate(retrieved) if len(doc.page_content) == 0]
            if empty_docs:
                errors.append(f"Retrieved {len(empty_docs)} documents with empty content")
            else:
                print("✓ Retrieved documents have content")
                
    except ImportError as e:
        errors.append(f"Failed to import required modules: {e}")
    except Exception as e:
        errors.append(f"Vector store/retriever validation failed: {e}")
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✓ Vector store and retriever validated\n")
    return True


def validate_rag_chain():
    """Validate that the full RAG chain works with a simple query"""
    print("=" * 60)
    print("STEP 10: Validating RAG chain with test query")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    try:
        from langchain_ollama import OllamaLLM, OllamaEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        
        # Load and prepare documents
        loaders = [
            CSVLoader('data/simulated_logs.csv'),
            TextLoader('data/telco_manual.txt', encoding='utf-8'),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        
        # Initialize LLM and embeddings
        llm = OllamaLLM(model="mistral:7b")
        embeddings = OllamaEmbeddings(model="mistral:7b")
        
        # Create vector store and retriever
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create RAG chain
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
        
        # Test with a simple query
        test_query = "What is network troubleshooting?"
        print(f"Testing with query: '{test_query}'")
        response = rag_chain.invoke(test_query)
        
        if response is None:
            errors.append("RAG chain returned None")
        elif len(response.strip()) == 0:
            errors.append("RAG chain returned empty response")
        elif len(response.strip()) < 10:
            warnings.append(f"RAG chain returned very short response: {len(response)} chars")
        else:
            print(f"✓ RAG chain works (response length: {len(response)} chars)")
            print(f"  Response preview: {response[:100]}...")
        
        # Test with telco-specific query
        telco_query = "What causes BGP peer down issues?"
        print(f"\nTesting with telco query: '{telco_query}'")
        telco_response = rag_chain.invoke(telco_query)
        
        if telco_response is None or len(telco_response.strip()) == 0:
            errors.append("RAG chain failed with telco query")
        else:
            response_lower = telco_response.lower()
            relevant_terms = ['bgp', 'peer', 'network', 'interface', 'neighbor', 'down']
            has_relevant_term = any(term in response_lower for term in relevant_terms)
            if has_relevant_term:
                print(f"✓ RAG chain works with telco query (response length: {len(telco_response)} chars)")
            else:
                warnings.append("Telco query response may not be relevant")
                
    except ImportError as e:
        errors.append(f"Failed to import required modules: {e}")
    except Exception as e:
        errors.append(f"RAG chain validation failed: {e}")
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✓ RAG chain validated\n")
    return True


def main():
    """Run all validation steps"""
    parser = argparse.ArgumentParser(description='Validate data and LLM before usage')
    parser.add_argument('--test-llm', action='store_true', 
                       help='Include LLM integration tests (requires Ollama running)')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("DATA VALIDATION - Checking data before LLM usage")
    print("=" * 60 + "\n")
    
    validations = [
        ("Data Files", validate_data_files),
        ("CSV Structure", validate_csv_structure),
        ("Text Content", validate_text_content),
        ("Document Loading", validate_document_loading),
        ("Document Splitting", validate_document_splitting),
        ("Data Quality", validate_data_quality),
    ]
    
    # Add LLM tests if requested
    if args.test_llm:
        validations.extend([
            ("LLM Initialization", validate_llm_initialization),
            ("Embeddings", validate_embeddings),
            ("Vector Store & Retriever", validate_vectorstore_and_retriever),
            ("RAG Chain", validate_rag_chain),
        ])
        print("ℹ️  LLM integration tests enabled (requires Ollama running)\n")
    
    results = []
    for name, validation_func in validations:
        try:
            result = validation_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Fatal error in {name}: {e}\n")
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} validations passed")
    
    if passed == total:
        print("\n✅ All validations passed! Data is ready for LLM usage.")
        return 0
    else:
        print(f"\n❌ {total - passed} validation(s) failed. Please fix errors before using LLM.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

