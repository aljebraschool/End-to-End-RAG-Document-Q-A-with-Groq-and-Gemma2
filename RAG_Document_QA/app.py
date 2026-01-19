import streamlit as st
import os
import time
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


# Check for environment variables and provide user-friendly error messages
try:
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.error("GROQ api key is not found in your environment variable. Please add it to your .env file")
    
    
except Exception as e:
    st.error(f'Error loading environment variable str{e}')

try:
    model = ChatGroq(model = 'llama-3.1-8b-instant')
except Exception as e:
    st.error(f"Error initializing Groq model str{e}")
    model = None

prompt = ChatPromptTemplate.from_template(
    """
        Using the {context} given, please provide the most accurate response for the {input} asked
        context : {context}
        question : {input}
    """
)

#Ensure the research_paper directory exit which will be used to model to answer question
if not os.path.exists('research_papers'):
    os.makedirs("research_papers")




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
                if not os.path.exists("research_papers") or len(os.listdir("research_papers")) == 0:
                    st.error("Directory 'research_papers' not found. Please create this directory and add your PDF files.")
                    return False
                
                # Load PDF documents from a directory
                st.session_state.loader = PyPDFDirectoryLoader("research_papers")
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
            st.error(f"Error creating vector database: {str(e)}")
            return False
    return True


st.title("RAG Document q&A with Groq and Gemma2")

openai_key = st.text_input("Enter your openai api key", type = 'password')

#add file uploader 
uploaded_files = st.file_uploader("upload your research papers (PDF)", type = 'pdf', accept_multiple_files = True)

if uploaded_files:
    # Clear existing files to avoid duplicates
    import shutil
    if os.path.exists("research_papers"):
        shutil.rmtree("research_papers") #remove directory
    os.makedirs("research_papers") #make another directory

    for file in uploaded_files:
        # Save uploaded file to research_papers directory
        with open(os.path.join("research_papers", file.name), "wb") as f:
            f.write(file.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} PDF files successfully!")

# First explain what to do with clear instructions
st.write("1. Provide your openai key")
st.write("2. Upload your PDF research papers using the file uploader above")
st.write("3. Click 'Document Embedding' to index your research papers")
st.write("4. Then ask questions about the content of your papers")

# Document embedding button
if st.button("Document Embedding"):
    if not openai_key:
        st.error("Please provide your openai key first")
        
    if not uploaded_files and len(os.listdir("research_papers")) == 0:
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
            st.error(f"Error processing your request: {str(e)}")
            st.write("Please try again or check your API keys and document database.")
