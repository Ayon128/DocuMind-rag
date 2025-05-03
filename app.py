import streamlit as st
from streamlit_option_menu import option_menu
import pdfplumber
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import base64
import json
import time
import re
from io import BytesIO
from datetime import datetime
from huggingface_hub import login
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration variables
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Log in to Hugging Face
login(token=HUGGINGFACE_API_TOKEN)

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_data = BytesIO(uploaded_file.read())
    total_pages = 0
    text = ""
   
    with pdfplumber.open(pdf_data) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
   
    return text, total_pages, pdf_data

# Function to create text chunks
def create_chunks(extracted_data):
    documents = [Document(page_content=extracted_data)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Function to get embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to load LLM
def load_llm(model_id=DEFAULT_MODEL, temperature=0.3, max_length=768):
    return HuggingFaceHub(
        repo_id=model_id,
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        model_kwargs={"temperature": temperature, "max_length": max_length}
    )

# Enhanced Custom Prompt Template - ANSWER ONLY
CUSTOM_PROMPT_TEMPLATE = """
You are DocuMind, an intelligent assistant specialized in extracting and explaining information from PDF documents. 

Instructions:
1. Answer questions based ONLY on the information provided in the context below.
2. If the answer cannot be found in the context, simply respond with "I don't have enough information in the document to answer this question" without making up information.
3. Be concise yet comprehensive in your responses.
4. Format your response for readability using bullet points or paragraphs as appropriate.
5. If asked about figures, tables, or images that might be in the document but not in the text context, mention that you cannot access visual elements directly.
6. IMPORTANT: Respond directly without mentioning 'based on the context', 'according to the document', or similar phrases. Never expose the raw context to the user.

Context (information from the document): 
{context}

Question: {question}

Answer:
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def setup_qa_chain(db, model_id=DEFAULT_MODEL, temperature=0.3):
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(model_id, temperature=temperature, max_length=768),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    return qa_chain

# Function to create PDF viewer
def display_pdf(pdf_data):
    base64_pdf = base64.b64encode(pdf_data.getvalue()).decode('utf-8')
    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" style="border: none;"></iframe>
    """
    return pdf_display

# Function to export chat
def export_chat_history(messages, format_type="json"):
    if format_type == "json":
        chat_export = json.dumps(messages, indent=2)
        filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:  # text format
        chat_export = ""
        for msg in messages:
            chat_export += f"{msg['role'].upper()}: {msg['content']}\n\n"
        filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
   
    return chat_export, filename

# Enhanced welcome message function
def get_welcome_message():
    return {
        "role": "assistant", 
        "content": "Welcome to DocuMind! I'm your intelligent document assistant. Upload any PDF document, and I'll help you extract insights through natural conversation. Just drag and drop your file to get started.", 
        "timestamp": datetime.now().strftime("%I:%M %p")
    }

# Improved PDF processing message
def get_pdf_processed_message(filename, total_pages):
    return {
        "role": "assistant",
        "content": f"✅ I've successfully processed '{filename}' ({total_pages} pages). This document is now ready for analysis! What would you like to know about its contents?",
        "timestamp": datetime.now().strftime("%I:%M %p")
    }

# Set page configuration - this must be the first Streamlit command
st.set_page_config(page_title="DocuMind", page_icon="📝", layout="wide")

# Apply custom CSS
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #5E5CEC;
        --primary-light: #F0F0FF;
        --text-color: #333333;
        --text-secondary: #667085;
        --background-color: #F9FAFB;
        --card-color: #FFFFFF;
        --border-color: #E5E7EB;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
    }

    /* General styles */
    body {
        font-family: 'Inter', sans-serif;
        background-color: var(--background-color);
        color: var(--text-color);
        margin: 0;
    }
    
    /* Remove padding from Streamlit container */
    .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Function to create processing UI
def create_processing_ui():
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem 0;">
            <div style="width: 40px; height: 40px; border: 3px solid rgba(94, 92, 236, 0.2); border-radius: 50%; border-top-color: #5E5CEC; animation: spin 1s ease-in-out infinite; margin-bottom: 1rem;"></div>
            <h2 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem;">Processing document...</h2>
            <p style="font-size: 0.875rem; color: #667085; margin-bottom: 1.5rem;">This may take a moment depending on the file size.</p>
        </div>
        
        <style>
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Main function
def main():
    # Load CSS
    load_css()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [get_welcome_message()]
    
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    
    if "pdf_data" not in st.session_state:
        st.session_state.pdf_data = None
    
    if "total_pages" not in st.session_state:
        st.session_state.total_pages = 0
    
    if "file_name" not in st.session_state:
        st.session_state.file_name = None
    
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "chat"
    
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    
    # Create header based on the current tab
    # We use a simplified version without JavaScript
    st.title("DocuMind")
    st.caption("Interactive PDF Intelligence")
    
    # Navigation tabs with Streamlit
    tabs = ["Chat", "Settings", "About"]
    selected_tab = option_menu(
        menu_title=None,
        options=tabs,
        icons=["chat", "gear", "info-circle"],
        default_index=tabs.index("Chat") if st.session_state.current_tab == "chat" else 
                     tabs.index("Settings") if st.session_state.current_tab == "settings" else 
                     tabs.index("About"),
        orientation="horizontal",
    )
    
    # Update session state based on selected tab
    if selected_tab == "Chat":
        st.session_state.current_tab = "chat"
    elif selected_tab == "Settings":
        st.session_state.current_tab = "settings"
    elif selected_tab == "About":
        st.session_state.current_tab = "about"
    
    # Display content based on current tab
    if st.session_state.current_tab == "chat":
        if st.session_state.pdf_data is None:
            if st.session_state.is_processing:
                create_processing_ui()
            else:
                # Use Streamlit's built-in file uploader
                uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
                st.markdown("<h3>Upload your PDF document</h3>", unsafe_allow_html=True)
                st.markdown("Drag and drop your file here, or click to browse your files")
                st.caption("Supported format: PDF (max 20MB)")
                
                if uploaded_file is not None and st.session_state.file_name != uploaded_file.name:
                    st.session_state.is_processing = True
                    st.session_state.file_name = uploaded_file.name
                    
                    with st.spinner("Processing document..."):
                        # Extract text from PDF
                        documents, total_pages, pdf_data = extract_text_from_pdf(uploaded_file)
                        
                        # Create chunks and vectorstore
                        text_chunks = create_chunks(documents)
                        embedding_model = get_embedding_model()
                        db = FAISS.from_documents(text_chunks, embedding_model)
                        
                        # Save vectorstore
                        DB_FAISS_PATH = "db_faiss"
                        db.save_local(DB_FAISS_PATH)
                        
                        # Setup QA chain with the improved prompt and parameters
                        qa_chain = setup_qa_chain(db)
                        
                        # Update session state with processed data
                        st.session_state.pdf_data = pdf_data
                        st.session_state.total_pages = total_pages
                        st.session_state.qa_chain = qa_chain
                        st.session_state.messages = [get_pdf_processed_message(uploaded_file.name, total_pages)]
                        st.session_state.is_processing = False
                        
                        st.rerun()
        else:
            # Chat interface using pure Streamlit components
            st.subheader(f"Chat with document: {st.session_state.file_name}")
            
            # Display chat messages
            for msg in st.session_state.messages:
                role = msg['role']
                content = msg['content']
                
                if role == "user":
                    st.chat_message("user").write(content)
                else:
                    st.chat_message("assistant").write(content)
            
            # Chat input
            user_input = st.chat_input("Ask a question about the document...")
            
            if user_input:
                st.chat_message("user").write(user_input)
                
                # Add user message to session state
                user_message = {"role": "user", "content": user_input, "timestamp": datetime.now().strftime("%I:%M %p")}
                st.session_state.messages.append(user_message)
                
                # Process the query
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.qa_chain({"query": user_input})
                        answer = response["result"]
                        
                        # Only show the answer, not the context
                        # Remove any prefixes like "Answer:" or "answer:"
                        if "Answer:" in answer:
                            answer = answer.split("Answer:", 1)[1].strip()
                        elif "answer:" in answer.lower():
                            answer = re.split(r'answer:', answer, flags=re.IGNORECASE)[1].strip()
                        
                        # Add assistant message to session state
                        assistant_message = {"role": "assistant", "content": answer, "timestamp": datetime.now().strftime("%I:%M %p")}
                        st.session_state.messages.append(assistant_message)
                        
                        # Display the assistant's response
                        st.chat_message("assistant").write(answer)
                    except Exception as e:
                        error_message = f"Sorry, I encountered an error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_message, "timestamp": datetime.now().strftime("%I:%M %p")})
                        st.chat_message("assistant").write(error_message)
            
            # Action buttons for chat
            col1, col2 = st.columns(2)
            with col1:
                if st.button("New Document", use_container_width=True):
                    st.session_state.messages = [get_welcome_message()]
                    st.session_state.qa_chain = None
                    st.session_state.pdf_data = None
                    st.session_state.total_pages = 0
                    st.session_state.file_name = None
                    st.rerun()
            
            with col2:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.messages = [get_pdf_processed_message(st.session_state.file_name, st.session_state.total_pages)]
                    st.rerun()
    
    elif st.session_state.current_tab == "settings":
        st.header("Settings")
        
        with st.expander("Appearance", expanded=True):
            st.toggle("Dark Mode", key="dark_mode", help="Switch between light and dark theme")
            st.toggle("Compact Mode", key="compact_mode", value=True, help="Display more content with tighter spacing")
            
            # Font size slider
            st.select_slider(
                "Font Size",
                options=["Small", "Medium", "Large"],
                value="Medium",
                key="font_size"
            )
        
        with st.expander("Answer Settings", expanded=True):
            st.toggle("Direct Answer Mode", key="direct_answer_mode", value=True, help="Show only answers without context")
            
            # Answer length selector
            st.selectbox(
                "Answer Length",
                ["Concise", "Balanced", "Detailed"],
                index=1,
                key="answer_length",
                help="Preferred response verbosity"
            )
    
    elif st.session_state.current_tab == "about":
        st.header("About DocuMind")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("📝")
        with col2:
            st.subheader("DocuMind")
            st.caption("Version 1.0.0")
        
        st.markdown("""
        DocuMind is an intelligent PDF assistant that uses advanced AI to help you interact with your documents through natural conversation.
        """)
        
        st.subheader("Key Features")
        features = [
            "Natural language interaction with PDF documents",
            "Direct answers without complex context",
            "Advanced semantic understanding of document content",
            "Clean, intuitive interface for document analysis",
            "Context-aware responses to your specific questions"
        ]
        for feature in features:
            st.markdown(f"- {feature}")
        
        st.subheader("Technology")
        st.markdown("""
        DocuMind uses state-of-the-art natural language processing and machine learning to analyze and understand document content, powered by Mistral-7B and Langchain technologies. The application is built with Streamlit, providing a responsive and interactive user interface for seamless document analysis.
        """)

if __name__ == "__main__":
    main()