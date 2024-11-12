import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from nltk.tokenize import sent_tokenize
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import nltk
os.system("pip install sentence-transformers")

# Explicitly download NLTK data
nltk.download('punkt', quiet=True)

# Load environment variables from .env
load_dotenv()

# Configure Groq API key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

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
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Create document chain for summary
        summary_document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=summary_prompt
        )
        
        # Create retrieval chain
        summary_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=summary_document_chain
        )
        
        # Generate summary
        response = summary_chain.invoke({"input": "Generate a summary"})
        return response['answer']
    except Exception as e:
        return f"Error generating summary: {str(e)}"


# Function to update query input
def update_query():
    st.session_state.user_query = st.session_state.quick_question

# Function to save FAISS index
def save_vector_store(vector_store, index_name="tender_index"):
    try:
        vector_store.save_local("faiss_indexes/" + index_name)
        return True
    except Exception as e:
        st.error(f"Error saving index: {str(e)}")
        return False

# Function to load FAISS index
def load_vector_store(index_name="tender_index"):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if os.path.exists("faiss_indexes/" + index_name):
            return FAISS.load_local("faiss_indexes/" + index_name, embeddings)
        return None
    except Exception as e:
        st.error(f"Error loading index: {str(e)}")
        return None

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

                    # Initialize embeddings
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                    # Tokenize the document into sentences using NLTK
                    splitted_text = sent_tokenize(text)

                    # Create the FAISS vector store
                    st.session_state.vector_store = FAISS.from_texts(texts=splitted_text, embedding=embeddings)
                    
                    # Save the index
                    if not os.path.exists("faiss_indexes"):
                        os.makedirs("faiss_indexes")
                    save_vector_store(st.session_state.vector_store)
                    
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

