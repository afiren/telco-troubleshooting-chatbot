"""
Tests to validate data before using LLM to avoid consuming resources with incorrect data.
Run with: pytest test_data_validation.py -v
"""
import pytest
import os
import pandas as pd
from pathlib import Path
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter


class TestDataFiles:
    """Test that data files exist and are accessible"""
    
    def test_csv_file_exists(self):
        """Verify CSV file exists"""
        csv_path = Path('data/simulated_logs.csv')
        assert csv_path.exists(), f"CSV file not found at {csv_path}"
        assert csv_path.is_file(), f"{csv_path} is not a file"
    
    def test_text_file_exists(self):
        """Verify text file exists"""
        text_path = Path('data/telco_manual.txt')
        assert text_path.exists(), f"Text file not found at {text_path}"
        assert text_path.is_file(), f"{text_path} is not a file"
    
    def test_csv_file_readable(self):
        """Verify CSV file can be read"""
        csv_path = 'data/simulated_logs.csv'
        try:
            df = pd.read_csv(csv_path)
            assert len(df) > 0, "CSV file is empty"
        except Exception as e:
            pytest.fail(f"Failed to read CSV file: {e}")
    
    def test_text_file_readable(self):
        """Verify text file can be read"""
        text_path = 'data/telco_manual.txt'
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert len(content) > 0, "Text file is empty"
        except Exception as e:
            pytest.fail(f"Failed to read text file: {e}")


class TestCSVStructure:
    """Test CSV structure and content validity"""
    
    @pytest.fixture
    def df(self):
        """Load CSV as fixture"""
        return pd.read_csv('data/simulated_logs.csv')
    
    def test_csv_has_expected_columns(self, df):
        """Verify CSV has all expected columns"""
        expected_columns = {
            'timestamp', 'pri', 'severity', 'facility', 'hostname',
            'alert_type', 'kpi_value', 'description', 'recommended_action'
        }
        actual_columns = set(df.columns)
        assert expected_columns.issubset(actual_columns), \
            f"Missing columns. Expected: {expected_columns}, Got: {actual_columns}"
    
    def test_csv_has_data_rows(self, df):
        """Verify CSV has at least some data rows"""
        assert len(df) > 0, "CSV has no data rows"
        assert len(df) >= 10, f"CSV has too few rows ({len(df)}), expected at least 10"
    
    def test_csv_no_empty_critical_fields(self, df):
        """Verify critical fields are not empty"""
        critical_fields = ['timestamp', 'severity', 'hostname', 'alert_type', 'description']
        for field in critical_fields:
            assert field in df.columns, f"Critical field '{field}' missing"
            empty_count = df[field].isna().sum() + (df[field] == '').sum()
            assert empty_count == 0, \
                f"Field '{field}' has {empty_count} empty/null values"
    
    def test_csv_timestamp_format(self, df):
        """Verify timestamp format is valid"""
        # Try to parse timestamps
        try:
            pd.to_datetime(df['timestamp'], errors='raise')
        except Exception as e:
            pytest.fail(f"Invalid timestamp format: {e}")
    
    def test_csv_severity_values(self, df):
        """Verify severity values are valid"""
        valid_severities = {
            'Emergency', 'Alert', 'Critical', 'Error', 
            'Warning', 'Notice', 'Info', 'Debug'
        }
        invalid = set(df['severity'].unique()) - valid_severities
        assert len(invalid) == 0, \
            f"Invalid severity values found: {invalid}"


class TestTextFileContent:
    """Test text file content validity"""
    
    @pytest.fixture
    def text_content(self):
        """Load text file as fixture"""
        with open('data/telco_manual.txt', 'r', encoding='utf-8') as f:
            return f.read()
    
    def test_text_file_has_content(self, text_content):
        """Verify text file has meaningful content"""
        assert len(text_content) > 100, \
            f"Text file too short ({len(text_content)} chars), expected at least 100"
    
    def test_text_file_has_keywords(self, text_content):
        """Verify text file contains expected telco keywords"""
        expected_keywords = ['troubleshooting', 'network', 'KPI', 'RCA']
        content_lower = text_content.lower()
        missing = [kw for kw in expected_keywords if kw.lower() not in content_lower]
        assert len(missing) == 0, \
            f"Missing expected keywords: {missing}"


class TestDocumentLoading:
    """Test that documents load correctly"""
    
    def test_csv_loader_works(self):
        """Verify CSVLoader can load documents"""
        try:
            loader = CSVLoader('data/simulated_logs.csv')
            docs = loader.load()
            assert len(docs) > 0, "CSVLoader returned no documents"
            assert all(hasattr(doc, 'page_content') for doc in docs), \
                "Documents missing page_content attribute"
            assert all(len(doc.page_content) > 0 for doc in docs), \
                "Some documents have empty page_content"
        except Exception as e:
            pytest.fail(f"CSVLoader failed: {e}")
    
    def test_text_loader_works(self):
        """Verify TextLoader can load documents"""
        try:
            loader = TextLoader('data/telco_manual.txt', encoding='utf-8')
            docs = loader.load()
            assert len(docs) > 0, "TextLoader returned no documents"
            assert all(hasattr(doc, 'page_content') for doc in docs), \
                "Documents missing page_content attribute"
            assert all(len(doc.page_content) > 0 for doc in docs), \
                "Some documents have empty page_content"
        except Exception as e:
            pytest.fail(f"TextLoader failed: {e}")
    
    def test_all_loaders_combined(self):
        """Verify all loaders work together"""
        loaders = [
            CSVLoader('data/simulated_logs.csv'),
            TextLoader('data/telco_manual.txt', encoding='utf-8'),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        
        assert len(docs) > 0, "No documents loaded from any loader"
        assert all(len(doc.page_content) > 0 for doc in docs), \
            "Some documents have empty content"


class TestDocumentSplitting:
    """Test that document splitting works correctly"""
    
    @pytest.fixture
    def docs(self):
        """Load all documents as fixture"""
        loaders = [
            CSVLoader('data/simulated_logs.csv'),
            TextLoader('data/telco_manual.txt', encoding='utf-8'),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        return docs
    
    def test_text_splitter_works(self, docs):
        """Verify CharacterTextSplitter works"""
        try:
            text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            split_docs = text_splitter.split_documents(docs)
            assert len(split_docs) > 0, "Text splitter returned no chunks"
            assert all(len(doc.page_content) > 0 for doc in split_docs), \
                "Some split documents have empty content"
        except Exception as e:
            pytest.fail(f"Text splitter failed: {e}")
    
    def test_split_docs_have_reasonable_size(self, docs):
        """Verify split documents have reasonable chunk sizes"""
        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        
        # Check that chunks are not too large (should respect chunk_size)
        max_chunk_size = max(len(doc.page_content) for doc in split_docs)
        assert max_chunk_size <= 600, \
            f"Chunks too large (max: {max_chunk_size}), expected <= 600"
        
        # Check that we have multiple chunks (splitting occurred)
        assert len(split_docs) >= len(docs), \
            "Expected more chunks after splitting"


class TestDataQuality:
    """Test overall data quality metrics"""
    
    def test_total_content_size(self):
        """Verify we have enough content for meaningful RAG"""
        loaders = [
            CSVLoader('data/simulated_logs.csv'),
            TextLoader('data/telco_manual.txt', encoding='utf-8'),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        
        total_chars = sum(len(doc.page_content) for doc in docs)
        assert total_chars > 1000, \
            f"Total content too small ({total_chars} chars), expected at least 1000"
    
    def test_data_diversity(self):
        """Verify data has diversity (not all same content)"""
        loader = CSVLoader('data/simulated_logs.csv')
        docs = loader.load()
        
        # Check that descriptions are diverse
        descriptions = [doc.page_content for doc in docs[:10]]  # Sample first 10
        unique_descriptions = set(descriptions)
        assert len(unique_descriptions) > 1, \
            "Data lacks diversity - all descriptions are identical"


@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests for LLM, embeddings, and RAG chain.
    These tests require Ollama to be running with mistral:7b model.
    Marked as integration tests - run with: pytest -m integration
    """
    
    @pytest.fixture(scope="class")
    def llm(self):
        """Initialize LLM"""
        from langchain_ollama import OllamaLLM
        try:
            llm = OllamaLLM(model="mistral:7b")
            # Test with a simple query to verify it works
            response = llm.invoke("Say 'test' if you can read this.")
            assert response is not None
            return llm
        except Exception as e:
            pytest.skip(f"LLM not available: {e}. Make sure Ollama is running with mistral:7b")
    
    @pytest.fixture(scope="class")
    def embeddings(self):
        """Initialize embeddings"""
        from langchain_ollama import OllamaEmbeddings
        try:
            embeddings = OllamaEmbeddings(model="mistral:7b")
            # Test with a simple text
            result = embeddings.embed_query("test query")
            assert result is not None
            assert len(result) > 0, "Embeddings returned empty result"
            return embeddings
        except Exception as e:
            pytest.skip(f"Embeddings not available: {e}. Make sure Ollama is running with mistral:7b")
    
    @pytest.fixture(scope="class")
    def split_docs(self):
        """Load and split documents"""
        loaders = [
            CSVLoader('data/simulated_logs.csv'),
            TextLoader('data/telco_manual.txt', encoding='utf-8'),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        return text_splitter.split_documents(docs)
    
    def test_llm_initialization(self, llm):
        """Verify LLM can be initialized and responds"""
        assert llm is not None, "LLM is None"
        # Simple test query
        response = llm.invoke("Respond with just the word 'OK'")
        assert response is not None, "LLM returned None"
        assert len(response.strip()) > 0, "LLM returned empty response"
    
    def test_embeddings_generation(self, embeddings):
        """Verify embeddings can be generated"""
        test_text = "network troubleshooting"
        embedding = embeddings.embed_query(test_text)
        assert embedding is not None, "Embedding is None"
        assert len(embedding) > 0, "Embedding is empty"
        assert isinstance(embedding, list), "Embedding should be a list"
        assert all(isinstance(x, (int, float)) for x in embedding), \
            "Embedding should contain numbers"
    
    def test_embeddings_batch(self, embeddings):
        """Verify batch embedding generation works"""
        test_texts = ["network", "troubleshooting", "telco"]
        embeddings_list = embeddings.embed_documents(test_texts)
        assert len(embeddings_list) == len(test_texts), \
            f"Expected {len(test_texts)} embeddings, got {len(embeddings_list)}"
        assert all(len(emb) > 0 for emb in embeddings_list), \
            "Some embeddings are empty"
    
    def test_vectorstore_creation(self, embeddings, split_docs):
        """Verify vector store can be created from documents"""
        from langchain_community.vectorstores import FAISS
        try:
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            assert vectorstore is not None, "Vector store is None"
            # Verify we can search
            results = vectorstore.similarity_search("network", k=2)
            assert len(results) > 0, "Vector store search returned no results"
        except Exception as e:
            pytest.fail(f"Failed to create vector store: {e}")
    
    def test_retriever_functionality(self, embeddings, split_docs):
        """Verify retriever works correctly"""
        from langchain_community.vectorstores import FAISS
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Test retrieval
        results = retriever.invoke("BGP peer down")
        assert len(results) > 0, "Retriever returned no results"
        assert len(results) <= 5, f"Retriever returned too many results: {len(results)}"
        assert all(hasattr(doc, 'page_content') for doc in results), \
            "Retrieved documents missing page_content"
        assert all(len(doc.page_content) > 0 for doc in results), \
            "Some retrieved documents have empty content"
    
    def test_rag_chain_simple_query(self, llm, embeddings, split_docs):
        """Verify RAG chain works with a simple query"""
        from langchain_community.vectorstores import FAISS
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        
        # Setup vector store and retriever
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create prompt and chain
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
        
        # Test with a simple query
        test_query = "What is network troubleshooting?"
        try:
            response = rag_chain.invoke(test_query)
            assert response is not None, "RAG chain returned None"
            assert len(response.strip()) > 0, "RAG chain returned empty response"
            assert isinstance(response, str), "RAG chain should return a string"
        except Exception as e:
            pytest.fail(f"RAG chain failed: {e}")
    
    def test_rag_chain_telco_query(self, llm, embeddings, split_docs):
        """Verify RAG chain works with a telco-specific query"""
        from langchain_community.vectorstores import FAISS
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        
        # Setup vector store and retriever
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create prompt and chain
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
        
        # Test with a telco-specific query
        test_query = "What causes BGP peer down issues?"
        try:
            response = rag_chain.invoke(test_query)
            assert response is not None, "RAG chain returned None"
            assert len(response.strip()) > 10, \
                f"RAG chain returned too short response: {len(response)} chars"
            # Check that response contains some relevant keywords
            response_lower = response.lower()
            relevant_terms = ['bgp', 'peer', 'network', 'interface', 'neighbor', 'down']
            has_relevant_term = any(term in response_lower for term in relevant_terms)
            assert has_relevant_term, \
                f"Response doesn't seem relevant. Response: {response[:100]}..."
        except Exception as e:
            pytest.fail(f"RAG chain failed with telco query: {e}")

