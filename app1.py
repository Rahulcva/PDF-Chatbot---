import io
import streamlit as st
from lang_funcs1 import *
from langchain.llms import Ollama
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.schema import Document
import pdfplumber  # Make sure pdfplumber is imported

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

# Initialize conversation history if not already done
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = {}

# Function to load and process PDF content from uploaded file using pdfplumber
def load_pdf_from_uploaded_file(uploaded_file):
    try:
        # Using pdfplumber to handle different types of PDFs
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return None

# Function to update conversation history
def update_conversation(pdf_id, question, answer):
    if pdf_id not in st.session_state.conversation_history:
        st.session_state.conversation_history[pdf_id] = []
    st.session_state.conversation_history[pdf_id].append({"question": question, "answer": answer})

# Function to get accumulated conversation context
def get_accumulated_context(pdf_id):
    if pdf_id in st.session_state.conversation_history:
        context = "\n".join([f"User: {item['question']}\nAssistant: {item['answer']}" 
                             for item in st.session_state.conversation_history[pdf_id]])
        return context
    return ""

# Main logic for processing PDF and generating responses
if uploaded_file is not None:
    try:
        # Check the file type before processing
        if uploaded_file.type != "application/pdf":
            st.error("The uploaded file is not a valid PDF. Please upload a valid PDF file.")
        else:
            # Load PDF data and prepare documents
            pdf_text = load_pdf_from_uploaded_file(uploaded_file)
            if not pdf_text:
                st.error("Could not extract text from the PDF.")
            else:
                documents = [Document(page_content=pdf_text)]

                # Split documents into chunks
                documents = split_docs(documents)

                # Load embedding model
                embed = load_embedding_model(model_path="all-MiniLM-L6-v2")

                # Create embeddings and store them in a vectorstore
                vectorstore = create_embeddings(documents, embed)

                # Convert vectorstore to retriever
                retriever = vectorstore.as_retriever()

                # Create prompt template
                template = """
                ### System:
                You are a respectful and honest assistant. You have to answer the user's questions using only the context provided to you. If you don't know the answer, just say you don't know. Don't try to make up an answer.

                ### Context:
                {context}

                ### User:
                {question}

                ### Response:
                """
                prompt = PromptTemplate.from_template(template)

                # Load QA chain
                qa_chain = load_qa_chain(retriever, initialize_chain(), prompt)

                # Generate a unique identifier for the current PDF file
                pdf_id = hash(uploaded_file.read())  # Create a unique ID based on the PDF content

                # Clear previous conversation history if a new PDF is uploaded
                if pdf_id not in st.session_state.conversation_history:
                    st.session_state.conversation_history[pdf_id] = []

                st.success("PDF loaded and processed successfully!")

    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")

# Displaying the conversation history for the specific PDF
if uploaded_file is not None:
    pdf_id = hash(uploaded_file.read())  # Get unique ID for the uploaded file

    if pdf_id in st.session_state.conversation_history and len(st.session_state.conversation_history[pdf_id]) > 0:
        st.markdown("### Conversation History")
        for item in st.session_state.conversation_history[pdf_id]:
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            st.markdown("---")

# Create layout for better positioning of question box
if qa_chain:
    st.markdown("### Ask me anything related to this PDF:")

    # Create a column layout to place the question box at the bottom
    col1, col2 = st.columns([1, 2])
    
    with col2:
        if 'question_input' not in st.session_state:
            st.session_state.question_input = ""  # Initialize input state
        
        # Display the input field in the second column
        question = st.text_input("Your question:", st.session_state.question_input)  # Use session state value
    
    # Add some spacing for the layout
    st.markdown("<br>", unsafe_allow_html=True)

    with col2:
        if st.button("Ask", key="ask_button"):  # Provide a unique key for the button
            if question:
                try:
                    # Get response using accumulated context from memory
                    response = get_response_with_memory(question, qa_chain, get_accumulated_context(pdf_id))

                    # Update conversation history
                    update_conversation(pdf_id, question, response)

                    # Display the response
                    st.markdown(f"**Assistant:** {response}")

                    # Update session state with the question field reset
                    st.session_state.question_input = ""  # Resetting the input after question is answered

                except Exception as e:
                    st.error(f"An error occurred while processing the question: {e}")
            else:
                st.warning("Please enter a question.")
else:
    st.info("Please upload a PDF to start asking questions.")
