import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from nltk.tokenize import sent_tokenize
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
import os
import google.generativeai as genai
import nltk

# Explicitly download NLTK data
nltk.download('punkt', quiet=True)

# Load environment variables from .env
load_dotenv()

# Configure Google AI credentials
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure Google AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the LLM and prompts
llm = ChatGroq(model='mixtral-8x7b-32768', temperature=0)

qa_prompt = ChatPromptTemplate.from_template(
    '''
As an expert in analyzing tender documents, you have a comprehensive understanding of the content within the uploaded PDFs and are skilled at extracting and interpreting details. Users will pose questions related to the tender document, seeking clear and concise answers.

If a question pertains to information not present in the document, kindly inform the user that the document does not contain the requested information.

In cases where the answer is not immediately clear from the document, respond with "I don't know" rather than offering potentially inaccurate information.

Below is a snippet of context from the relevant section of the document, which will not be shown to users: 
<context> 
Context: {context} 
Question: {input} 
</context> 

Your response should solely focus on providing useful information extracted from the document. Please be polite and enhance your response with specific details from the tender document where applicable.
'''
)

summary_prompt = ChatPromptTemplate.from_template(
    '''
As an expert tender analyst, please provide a concise overview summary of the tender document in approximately 100 words. Focus on the key aspects such as:
- Main purpose/scope of the tender
- Key requirements
- Important deadlines
- Significant evaluation criteria

Below is the content from the document:
<context>
{context}
</context>

Please provide a clear, professional summary that captures the essential information from this tender document in about 100 words.
'''
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""

# Function to generate document summary
def generate_document_summary(vector_store):
    try:
        # Create a chain specifically for summary generation
        summary_chain = create_stuff_documents_chain(llm, summary_prompt)
        
        # Get the most relevant chunks for summary
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        summary_retrieval_chain = create_retrieval_chain(retriever, summary_chain)
        
        # Generate the summary
        response = summary_retrieval_chain.invoke({'input': 'Generate a summary'})
        return response['answer']
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Function to update query input
def update_query():
    st.session_state.user_query = st.session_state.quick_question

# Page configuration
st.set_page_config(
    page_title="Tender Document Q&A System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .nav-link {
        display: inline-block;
        margin-right: 20px;
        font-size: 16px;
        color: #000;
        text-decoration: none;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .nav-link:hover {
        color: #007bff;
        background-color: #f8f9fa;
    }
    .st-emotion-cache-16idsys p {
        font-size: 18px;
        margin-bottom: 20px;
    }
    .sidebar-header {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for File Upload and Quick Questions
with st.sidebar:
    st.header("Upload Tender Document", divider='rainbow')
    file = st.file_uploader(label='Upload PDF', type='pdf')

    if file:
        # Process the document for embeddings
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != file.name:
            st.session_state.last_uploaded_file = file.name
            st.session_state.vector_store = None

        if st.session_state.vector_store is None:
            try:
                with st.spinner("Processing document..."):
                    pdf_reader = PdfReader(file)
                    text = ''
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                    # Initialize Google AI Embeddings with the configured API key
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=GOOGLE_API_KEY,
                    )

                    # Tokenize the document into sentences using NLTK
                    splitted_text = sent_tokenize(text)

                    # Create the vector store with the document embeddings
                    st.session_state.vector_store = Chroma.from_texts(texts=splitted_text, embedding=embeddings)
                    st.success('✨ Document processed successfully!')
            except Exception as e:
                st.error(f"An error occurred while processing the document: {str(e)}")
        else:
            st.info('💬 Using existing document from session')

    # Quick Questions Section
    st.subheader("Quick Questions", divider='rainbow')
    
    # Quick questions selection
    st.selectbox(
        'Select a question to ask:',
        [
            "What is the deadline for bid submission for this tender?",
            "What are the payment terms?",
            "Is there any penalty for late submission?",
            "What is the required Earnest Money Deposit (EMD) amount for this tender?",
            "Who is the point of contact for questions?",
            "Who are the contact persons for any tender-related queries?",
            "What are the required documents for submission?",
            "Are joint ventures allowed to participate in this tender?",
            "Are there any mandatory meetings?",
            "What is the scope of the contract?"
        ],
        key="quick_question",
        on_change=update_query
    )

# Main Content Area
st.title("📄 Tender Document Q&A System")
st.write("Upload your tender document and get instant answers to your questions!")

# Horizontal Navigation Bar
st.markdown(
    """
    <div style='margin-bottom: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>
        <a class="nav-link" href="#">📑 Summary</a>
        <a class="nav-link" href="#">📊 Projects</a>
        <a class="nav-link" href="#">✅ Task Lists</a>
        <a class="nav-link" href="#">❓ Queries</a>
    </div>
    """, 
    unsafe_allow_html=True
)

# Main content columns
col1, col2 = st.columns([2, 1])

with col1:
    # Generate Summary Section
    st.subheader("Document Summary", divider='rainbow')
    if st.button("Generate Summary 📝", use_container_width=True):
        if st.session_state.vector_store is not None:
            with st.spinner("Generating summary..."):
                summary = generate_document_summary(st.session_state.vector_store)
                st.success("✨ Summary generated successfully!")
                st.info(summary)
        else:
            st.warning("⚠️ Please upload a PDF document first to generate a summary.")

    # Query Section
    st.subheader("Ask Questions", divider='rainbow')
    user_prompt = st.text_input(
        'Enter your question about the tender document:',
        key="query_input",
        value=st.session_state.user_query,
        placeholder="Type your question here or select from quick questions..."
    )
    
    if user_prompt:
        if st.session_state.vector_store is not None:
            try:
                with st.spinner("Finding answer..."):
                    document_chain = create_stuff_documents_chain(llm, qa_prompt)
                    retriever = st.session_state.vector_store.as_retriever()
                    retriever_chain = create_retrieval_chain(retriever, document_chain)
                    response = retriever_chain.invoke({'input': user_prompt})
                    
                    st.write("**Your Question:**")
                    st.info(user_prompt)
                    st.write("**Answer:**")
                    st.success(response['answer'])
            except Exception as e:
                st.error(f"An error occurred while processing your query: {str(e)}")
        else:
            st.warning("⚠️ Please upload a PDF document first to ask questions.")

with col2:
    # Help Section
    st.subheader("Help & Tips", divider='rainbow')
    st.markdown("""
        ### 📝 How to use:
        1. Upload your tender document (PDF)
        2. Generate a summary to get an overview
        3. Ask questions using the input box
        4. Or select from quick questions

        ### 💡 Tips:
        - Be specific in your questions
        - Use quick questions for common queries
        - Check the summary first for key information
    """)

# Footer
st.markdown("""
    ---
    Made by © Muhammed Rasmil
""")