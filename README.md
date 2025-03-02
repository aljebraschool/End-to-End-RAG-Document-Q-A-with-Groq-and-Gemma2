# End-to-End RAG Document Q&A with Groq and Gemma2
A complete end-to-end Streamlit application that enables question-answering on research papers using Retrieval-Augmented Generation (RAG) powered by Groq's Gemma2 model. This project is ready to deploy on Streamlit Cloud for instant access anywhere.

📝 Description
This application allows users to upload any papers (PDF format) and ask natural language questions about their content. The entire pipeline—from document processing to answer generation—is implemented in a single, easy-to-use Streamlit app.

Using the RAG (Retrieval-Augmented Generation) approach, the system:

1. Extracts text from PDF files
2. Creates vector embeddings (via OpenAI Embeddings)
3. Stores these embeddings in a FAISS vector database
4. Retrieves relevant document chunks based on user queries
5. Generates answers using Groq's Gemma2 model

🌐 Live Demo
Access the live demo on Streamlit Cloud:
Your App Link [Here](https://end-to-end-rag-document-q-a-with-groq-and-gemma2-fjkivy2gnwufn.streamlit.app/)

![image](https://github.com/user-attachments/assets/e00a8d06-7b3f-4cc7-899a-265fcde825f7)

🚀 Features
  - Complete End-to-End Solution: From document ingestion to answer generation
  - PDF Document Processing: Automatic splitting and chunking of PDFs
  - Vector Embeddings & Semantic Search: Finds relevant sections of your documents
  - Integration with Groq's Gemma2: Powerful large language model for accurate responses
  - User-Friendly Interface: Streamlit-based UI for easy interaction
  - Expandable Document Similarity Results: Inspect the sources used for each answer
  - Error Handling & Logging: Better user experience and debugging

📂 Repository Structure
  ```bash
    End-to-End-RAG-Document-QA-with-Groq-and-Gemma2
    ├── research_papers/          # Directory where PDFs can be placed
    ├── app.py                    # Main Streamlit app
    ├── requirements.txt          # Python dependencies
    └── README.md                 # This file

  ```

🛠️ Installation
1. Clone the repository:
    ``` bash
 
      git clone https://github.com/YOUR-USERNAME/End-to-End-RAG-Document-QA-with-Groq-and-Gemma2.git
      cd End-to-End-RAG-Document-QA-with-Groq-and-Gemma2
    
    ```

3. Install required packages:
   ``` bash
   
        pip install -r requirements.txt
   
       ```
4. (Optional) Create and activate a virtual environment:
   
       ``` bash
        python -m venv venv
        source venv/bin/activate  # macOS/Linux
        venv\Scripts\activate     # Windows
       ```

  🔑 API Keys
  This application requires API keys from:
  - [Groq](https://console.groq.com/playground)
  - [OpenAI](https://platform.openai.com/docs/overview)

Create a .env file in the project’s root directory with the following variables:
    ``` bash
      GROQ_API_KEY=your_groq_api_key
      OPENAI_API_KEY=your_openai_api_key
    ```
(For Streamlit Cloud deployment, add these as secrets in your Streamlit Cloud dashboard instead.)

📋 Local Usage
  1. Add PDF files to the research_papers folder (or any folder you specify in the code).
  2. Run the Streamlit app:
      ``` bash
        streamlit run app.py
      ```
  3. Interact with the App
    - Upload your PDF(s) using the file uploader at the top of the interface.
    - Once your PDF file(s) have been successfully uploaded, click “Document Embedding” to index your research papers.
    - After embedding is complete, type your question in the “Ask questions about the content of your papers” text box.
  The app will generate an answer and display the relevant sources used to derive that answer.

🚀 Deploying to Streamlit Cloud
  1. Push your code to a public GitHub repository (be sure to ignore or exclude .env).
  2. Go to Streamlit Cloud and create a new app:
      - Select your GitHub repo and branch.
      - Enter app.py as the main file.
  3. Add your API keys as secrets in the Streamlit Cloud dashboard:
      - GROQ_API_KEY
      - OPENAI_API_KEY
  4. Deploy and share your app link!

🧩 How It Works
  1. Document Processing
    - PDF files are loaded and split into text chunks.
    - Each chunk is embedded using OpenAI Embeddings and stored in a FAISS vector database.
  2. Query Processing
    - The user’s question is embedded into the same vector space.
  3. Retrieval
    - The app searches the FAISS index for the most relevant chunks.
  4. Generation
    - Groq’s Gemma2 model generates a context-aware answer using the retrieved document chunks.

🔧 Technologies Used
- [Streamlit](https://streamlit.io/) - Web interface and easy deployment
- [LangChain](https://www.langchain.com/) - Orchestrating the RAG pipeline
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Groq](https://groq.com/) - Fast LLM inference for Gemma2
- Gemma2 - Underlying language model
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) - For document vectorization

📦 Requirements
  See requirements.txt for a complete list of dependencies.
    
🤝 Contributing
    Contributions, issues, and feature requests are welcome!
    Feel free to check the issues page or submit a pull request.

📄 License
    This project is licensed under the MIT License - see the LICENSE file for details.
    
🙏 Acknowledgements
    Groq for providing fast LLM inference
    Gemma2 for powering the question-answering capabilities
    LangChain for the RAG framework
    Streamlit for the user-friendly web application framework
          
