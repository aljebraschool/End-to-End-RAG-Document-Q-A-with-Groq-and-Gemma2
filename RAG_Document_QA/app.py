import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
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

    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        st.error("OpenAI key is not found in your environment variable. Please add it to your .env file")
except Exception as e:
    st.error(f'Error loading environment variable str{e}')

try:
    model = ChatGroq(model = 'gemma2-9b-it')
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

def create_vector_embedding():
    if "vector" not in st.session_state:
        try:
            # Show a spinner while processing
            with st.spinner("Creating document embeddings. This may take a minute..."):
                # Initialize embedding model
                st.session_state.embeddings = OpenAIEmbeddings()
                
                # Check if directory exists
                if not os.path.exists("research_papers"):
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

# First explain what to do with clear instructions
st.write("1. Click 'Document Embedding' to index your research papers")
st.write("2. Then ask questions about the content of your papers")

if st.button("Document Embedding"):
    #call the function to create embedding for the document
    success = create_vector_embedding()
    if success:
        st.success("Vector database is ready! You can now ask questions about your documents.")
    else:
        st.write("Failed to create vector database, please cheack the error message above")

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