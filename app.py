import io
import streamlit as st
from lang_funcs import *
from langchain.llms import Ollama
from langchain_core.prompts import PromptTemplate  # Corrected import for PromptTemplate
from PyPDF2 import PdfReader
from langchain.schema import Document


# Function to initialize and load the model for question answering
def initialize_chain():
    llm = Ollama(model="orca-mini", temperature=0)
    return llm

# Set page configuration with book icon and title
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📚",  # Book icon (Unicode character)
)

# Title and introductory message
st.title("PDF Chatbot 📚")
st.markdown(
    """
    **Welcome to the PDF Chatbot!**  
    Upload a PDF, and ask me anything related to its content.  
    I will answer your questions based on the context from the PDF.
    """
)

# Sidebar for file upload
st.sidebar.header("Upload PDF File")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Variable to hold the QA chain and retriever
qa_chain = None
retriever = None

# Function to load and process PDF content from uploaded file
def load_pdf_from_uploaded_file(uploaded_file):
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


if uploaded_file is not None:
    try:
        # Load PDF Data from the uploaded file
        pdf_text = load_pdf_from_uploaded_file(uploaded_file)
        
        # Wrap the text into a Document object for LangChain
        documents = [Document(page_content=pdf_text)]

        # Split documents based on the text
        documents = split_docs(documents)

        # Load Embedding Model
        embed = load_embedding_model(model_path="all-MiniLM-L6-v2")

        # Create Embeddings and Vectorstore
        vectorstore = create_embeddings(documents, embed)

        # Convert Vectorstore to Retriever
        retriever = vectorstore.as_retriever()

        # Creating the prompt template for QA without history
        template = """
        ### System:
        You are a respectful and honest assistant. You have to answer the user's questions using only the context \
        provided to you. If you don't know the answer, just say you don't know. Don't try to make up an answer.

        ### Context:
        {context}

        ### User:
        {question}

        ### Response:
        """
        prompt = PromptTemplate.from_template(template)

        # Load the QA Chain
        qa_chain = load_qa_chain(retriever, initialize_chain(), prompt)

        st.success("PDF loaded and processed successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# UI to display ongoing conversation and ask questions
if qa_chain:
    st.markdown("### Ask me anything related to this PDF:")

    # Input field and a button for submitting new questions
    question = st.text_input("Your question:", "")
    
    if st.button("Ask"):
        if question:
            try:
                # For each question, respond based only on the PDF content
                response = qa_chain.invoke({'query': question, 'context': pdf_text})  # No history used

                # Display the new answer and conversation
                st.markdown(f"**Assistant:** {response['result']}")
            except Exception as e:
                st.error(f"An error occurred while processing the question: {e}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload a PDF to start asking questions.")
