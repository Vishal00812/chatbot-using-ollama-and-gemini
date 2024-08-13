import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Ensure the Google API key is retrieved correctly
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it in your .env file.")
    st.stop()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8)

st.markdown(
    """
    <style>
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: white;
        padding: 10px 0;
        z-index: 100;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to create vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./pdf")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector Embeddings

# Define prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
""")

# Initialize session state for storing chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("Document Q&A with LangChain and FAISS")

# Button to embed documents
if st.button("Documents Embedding"):
    vector_embedding()
    st.success("Vector Store DB Is Ready")

# Display chat history
for message in st.session_state.messages:
    st.write(message)

# Move the input box to the bottom of the page
st.write("-----")  # Add a separator

# Input box for user to type in at the bottom
prompt1 = st.text_input("You: ", key="input_box", placeholder="Type your message here...")

st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# Handle user input and AI response
if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        response_time = time.process_time() - start
        
        # Display response and chat history
        st.session_state.messages.append(f"You: {prompt1}")
        st.session_state.messages.append(f"AI: {response['answer']}")
        st.write("Response Time:", response_time)
        st.write(response['answer'])

        # Clear the input box by resetting the text input
        st.session_state.input = ""

    else:
        st.error("Please embed the documents first by clicking 'Documents Embedding'.")

# Reset chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
