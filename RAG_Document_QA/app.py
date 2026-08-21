import streamlit as st
import os
import time
import uuid
import hmac
import logging
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional password gate: set APP_PASSWORD to require a shared password before
# the app can be used. This keeps unauthenticated visitors from running up
# usage on the paid Groq/OpenAI API keys configured on the server.
app_password = os.getenv("APP_PASSWORD")
if app_password:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        entered_password = st.text_input("Enter app password to continue", type="password")
        if entered_password:
            if hmac.compare_digest(entered_password, app_password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()

# Give each browser session its own isolated upload/working directory so that
# concurrent users on a shared deployment cannot see, overwrite, or delete
# each other's uploaded documents.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

BASE_UPLOAD_DIR = "research_papers"
SESSION_DIR = os.path.join(BASE_UPLOAD_DIR, st.session_state.session_id)


def safe_pdf_filename(name):
    """Strip any path components and reject anything that isn't a plain .pdf filename."""
    base_name = os.path.basename(name).strip()
    if not base_name or not base_name.lower().endswith(".pdf"):
        return None
    return base_name


# Check for environment variables and provide user-friendly error messages
try:
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.error("GROQ api key is not found in your environment variable. Please add it to your .env file")


except Exception as e:
    logger.exception("Error loading environment variable")
    st.error("Error loading environment variable. Please check the server logs.")

try:
    model = ChatGroq(model = 'llama-3.1-8b-instant')
except Exception as e:
    logger.exception("Error initializing Groq model")
    st.error("Error initializing Groq model. Please check your API key.")
    model = None

prompt = ChatPromptTemplate.from_template(
    """
        Using the {context} given, please provide the most accurate response for the {input} asked
        context : {context}
        question : {input}
    """
)

#Ensure the research_paper directory exit which will be used to model to answer question
if not os.path.exists(BASE_UPLOAD_DIR):
    os.makedirs(BASE_UPLOAD_DIR)




def create_vector_embedding():
    if "vector" not in st.session_state:
        #before creating embedding check if openai key is provided
        if not openai_key:
            st.error("Openai key is required for crreating embeddings")
        try:
            # Show a spinner while processing
            with st.spinner("Creating document embeddings. This may take a minute..."):
                # Initialize embedding model
                st.session_state.embeddings = OpenAIEmbeddings(api_key = openai_key)
                
                # Check if directory exists
                if not os.path.exists(SESSION_DIR) or len(os.listdir(SESSION_DIR)) == 0:
                    st.error("No uploaded PDF files found. Please upload your PDF files first.")
                    return False

                # Load PDF documents from a directory
                st.session_state.loader = PyPDFDirectoryLoader(SESSION_DIR)
                st.session_state.docs = st.session_state.loader.load()
                
                if not st.session_state.docs:
                    st.warning("No PDF documents found in the 'research_papers' directory. Please add some PDF files.")
                    return False
                
                # Split documents into manageable chunks
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                st.session_state.final_document = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
                
                # Create a FAISS vector store from the document chunks
                st.session_state.vector = FAISS.from_documents(st.session_state.final_document, st.session_state.embeddings)
                return True
                
        except Exception as e:
            logger.exception("Error creating vector database")
            st.error("Error creating vector database. Please check the server logs.")
            return False
    return True


st.title("RAG Document q&A with Groq and Gemma2")

openai_key = st.text_input("Enter your openai api key", type = 'password')

#add file uploader 
uploaded_files = st.file_uploader("upload your research papers (PDF)", type = 'pdf', accept_multiple_files = True)

if uploaded_files:
    # Clear this session's existing files to avoid duplicates (other sessions
    # are untouched since each session has its own SESSION_DIR)
    import shutil
    if os.path.exists(SESSION_DIR):
        shutil.rmtree(SESSION_DIR) #remove directory
    os.makedirs(SESSION_DIR) #make another directory

    skipped = 0
    for file in uploaded_files:
        # Sanitize the filename to prevent path traversal / writes outside SESSION_DIR
        safe_name = safe_pdf_filename(file.name)
        if not safe_name:
            skipped += 1
            continue
        # Save uploaded file to this session's directory
        with open(os.path.join(SESSION_DIR, safe_name), "wb") as f:
            f.write(file.getbuffer())

    if skipped:
        st.warning(f"Skipped {skipped} file(s) with invalid or non-PDF names.")
    st.success(f"Uploaded {len(uploaded_files) - skipped} PDF files successfully!")

# First explain what to do with clear instructions
st.write("1. Provide your openai key")
st.write("2. Upload your PDF research papers using the file uploader above")
st.write("3. Click 'Document Embedding' to index your research papers")
st.write("4. Then ask questions about the content of your papers")

# Document embedding button
if st.button("Document Embedding"):
    if not openai_key:
        st.error("Please provide your openai key first")
        
    if not uploaded_files and (not os.path.exists(SESSION_DIR) or len(os.listdir(SESSION_DIR)) == 0):
        st.error("Please upload PDF files first before creating embeddings.")
    else:
        success = create_vector_embedding()
        if success:
            st.success("Vector database is ready! You can now ask questions about your documents.")
        else:
            st.error("Failed to create vector database. Please check the error messages above.")

user_prompt = st.text_input("What is the question you want to ask from the research paper?")

if user_prompt:
    # Check if vector database has been created first
    if "vector" not in st.session_state:
        st.error("Please click 'Document Embedding' first to create the vector database")

    elif model is None:
        st.error("The Groq model could not be initialized. Please check your API key and try again.")
    
    else:
        try:
            with st.spinner("Searching for relevant information..."):
                #the crerate stuff documents chain combines all the documents as prompts and send it to the model
                document_chain = create_stuff_documents_chain(model, prompt)
                #make your vector a retiever that will be used to access the vector database
                retriever = st.session_state.vector.as_retriever()
                #create a retrieval chain using the document chain and the retriever
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                #calculate the time it takes (start time)
                start = time.process_time()
                #use the retriever chain to acess the vector database created above using the input from user as query
                response = retrieval_chain.invoke(
                    {
                        'input': user_prompt
                    }
                )

                #calculate the time it takes (end time)
                end = time.process_time()

                #find the time it takes to complete
                print(f"Response time : {end - start}")
                #see your answer from the query
                st.write(response['answer'])

                # Create an expandable UI section labeled "Document Similarity Search"
                # This creates a collapsible section that users can click to view or hide
                with st.expander("Document Similarity Search"):
                    # Loop through each document in the context returned from the retrieval
                    # enumerate() provides both the index (i) and the document object (doc)
                    for i, doc in enumerate(response['context']):
                        # Display the actual text content of the current document
                        st.write(doc.page_content)
                        
                        # Add a horizontal separator line between documents for better readability
                        # This helps users distinguish where one document ends and another begins
                        st.write("---------------------------------")

        except Exception as e:
            logger.exception("Error processing user request")
            st.error("Error processing your request. Please try again or check your API keys and document database.")
