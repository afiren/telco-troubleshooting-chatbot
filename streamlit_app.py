"""
Streamlit UI for Network Troubleshooting Chatbot
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Network Troubleshooting Chatbot",
    page_icon="🔧",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    st.session_state.retriever = None
    st.session_state.initialized = False

@st.cache_resource
def initialize_rag_chain():
    """Initialize the RAG chain (cached to avoid reloading on every interaction)"""
    try:
        # Suppress validation output when loading in Streamlit
        os.environ['VALIDATE_DATA'] = '0'
        
        # Import components from app.py
        from app import rag_chain, retriever
        return rag_chain, retriever, None
    except Exception as e:
        return None, None, str(e)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check if chain is initialized
    if not st.session_state.initialized:
        with st.spinner("Initializing RAG chain..."):
            chain, retriever, error = initialize_rag_chain()
            if error:
                st.error(f"Failed to initialize: {error}")
                st.session_state.rag_chain = None
                st.session_state.retriever = None
                st.session_state.initialized = True
            else:
                st.session_state.rag_chain = chain
                st.session_state.retriever = retriever
                st.session_state.initialized = True
                st.success("✅ RAG chain initialized!")
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    This chatbot helps troubleshoot network issues using:
    - **RAG (Retrieval-Augmented Generation)**
    - **Local LLM** (Ollama Mistral 7B)
    - **Network logs** and **troubleshooting manual**
    """)
    
    st.markdown("### 💡 Example Questions")
    st.markdown("""
    - What causes BGP peer down issues?
    - Why is connection slow?
    - How to troubleshoot interface flaps?
    - What are common root causes of network downtime?
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("🔧 Network Troubleshooting Chatbot")
st.markdown("Ask questions about network issues, troubleshooting, and root cause analysis.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show retrieved context if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(source[:200] + "..." if len(source) > 200 else source)

# User input
if prompt := st.chat_input("Ask about a network issue (e.g., 'Why is connection slow?')"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
        # Get response from chain
    if st.session_state.rag_chain is None:
        st.error("❌ RAG chain not initialized. Please check the sidebar for errors.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing network issue..."):
                try:
                    # Retrieve relevant documents for source citation
                    retrieved_docs = []
                    if st.session_state.retriever:
                        retrieved_docs = st.session_state.retriever.invoke(prompt)
                    
                    # Get response from RAG chain
                    response = st.session_state.rag_chain.invoke(prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Show sources if available
                    if retrieved_docs:
                        with st.expander("📄 View Retrieved Sources", expanded=False):
                            for i, doc in enumerate(retrieved_docs, 1):
                                st.markdown(f"**Source {i}:**")
                                # Truncate long documents
                                content = doc.page_content
                                if len(content) > 300:
                                    st.text(content[:300] + "...")
                                else:
                                    st.text(content)
                                st.markdown("---")
                    
                    # Store message with sources
                    sources = [doc.page_content for doc in retrieved_docs] if retrieved_docs else []
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>Powered by Ollama Mistral 7B • RAG with FAISS Vector Store</small>
    </div>
    """,
    unsafe_allow_html=True
)

