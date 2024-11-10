import streamlit as st
from dotenv import load_dotenv
import os
import uuid
from langchain_community.document_loaders import PDFMinerLoader, CSVLoader , Docx2txtLoader , TextLoader
import tempfile
import shutil
from io import BytesIO
from PIL import Image
import numpy as np
from database import VectorDatabase
from crew import SearchEngine
load_dotenv()


GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
# Set to track used UUIDs
used_uuids = set()

# Initialize the vector database
vector_db = VectorDatabase()

# Access the document and image stores
document_store = vector_db.get_vector_store()
image_store = vector_db.get_image_store()

search_engine = SearchEngine()


# Generate a unique list of UUIDs
def generate_unique_ids(num_ids):
    ids = []
    while len(ids) < num_ids:
        new_uuid = str(uuid.uuid4())
        if new_uuid not in used_uuids:  # Ensure no duplicates
            used_uuids.add(new_uuid)
            ids.append(new_uuid)
    return ids


def create_temp_file_from_uploaded_image(uploaded_file):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp() 
    # Create a temporary file path
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    # Write the image content to the temp file
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.getvalue())  # `uploaded_file.getvalue()` returns the content
    return temp_file_path

def get_file_suffix(uploaded_file):
    return os.path.splitext(uploaded_file.name)[1]

def reconstruct_image_from_array(image_array):
        # Check if image_array is a list, extract the first element if it's wrapped in a list
        if isinstance(image_array, list):
            image_array = image_array[0][0]  # Extract the first element (NumPy array) from the list
        
        # Ensure the array has the correct shape (H, W, C) for a color image (RGB)
        if image_array.ndim == 3 and image_array.shape[2] == 3:  # RGB image
            pil_image = Image.fromarray(image_array)
        elif image_array.ndim == 2:  # Grayscale image
            pil_image = Image.fromarray(image_array, mode='L')
        else:
            raise ValueError("Array dimensions are not valid for image reconstruction.")
        
        return pil_image

def pdf_file_loader(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.getbuffer())
        temp_filename = temp_file.name
    try:
        loader = PDFMinerLoader(temp_filename)
        documents = loader.load()
        if documents:
            st.success(f"Loaded {len(documents)} document(s) from the PDF.")
            return documents  
        else:
            st.error("No documents loaded from the PDF.")
            return None
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None

def csv_file_loader(csv_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(csv_file.getbuffer())
        temp_filename = temp_file.name
    try:
        loader = CSVLoader(temp_filename)
        documents = loader.load()
        if documents:
            st.success(f"Loaded {len(documents)} document(s) from the CSV.")
            return documents  
        else:
            st.error("No documents loaded from the CSV.")
            return None
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def docx_file_loader(docx_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
        temp_file.write(docx_file.getbuffer())
        temp_filename = temp_file.name
    try:
        loader = Docx2txtLoader(temp_filename)
        documents = loader.load()
        if documents:
            st.success(f"Loaded {len(documents)} document(s) from the docx.")
            return documents  
        else:
            st.error("No documents loaded from the docx.")
            return None
    except Exception as e:
        st.error(f"Error loading docx: {e}")
        return None
    
def text_file_loader(text_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(text_file.getbuffer())
        temp_filename = temp_file.name
    try:
        loader = TextLoader(temp_filename)
        documents = loader.load()
        if documents:
            st.success(f"Loaded {len(documents)} document(s) from the txt.")
            return documents  
        else:
            st.error("No documents loaded from the txt.")
            return None
    except Exception as e:
        st.error(f"Error loading txt: {e}")
        return None

# Set up Chroma client and vector store

# Streamlit app setup
st.set_page_config(page_title="Document_chatbot", layout="wide")
st.title("Document RAG")
st.sidebar.header("Upload Files/Documents")

if file := st.sidebar.file_uploader(label="file uploader", type=['pdf', 'jpg', 'png', 'csv', 'txt', 'docx', 'doc']):
    if st.sidebar.button("Submit and process"):
        file_suffix = get_file_suffix(file)
        if file_suffix == '.pdf':
            documents = pdf_file_loader(file)
            if documents:
                ids = generate_unique_ids(len(documents))  
                document_store.add_documents(ids=ids, documents=documents)
        elif file_suffix == '.csv':
            documents = csv_file_loader(file)
            if documents:
                ids = generate_unique_ids(len(documents))  
                document_store.add_documents(ids=ids, documents=documents)
        elif file_suffix == '.docx':
            documents = docx_file_loader(file)
            if documents:
                ids = generate_unique_ids(len(documents))  
                document_store.add_documents(ids=ids, documents=documents)  
        elif file_suffix == '.txt':
            documents = text_file_loader(file)
            if documents:
                ids = generate_unique_ids(len(documents))  
                document_store.add_documents(ids=ids, documents=documents)
        elif file_suffix == '.jpg' or 'png' or 'jpeg':
            document = create_temp_file_from_uploaded_image(file)
            if document:
                image_store.add(
                    ids = generate_unique_ids(1),
                    uris = [document]
                )

def handle_submit():
    # Get the user query
    query = st.session_state.get("user_query", "")

    if query:
        # Perform search using the pipeline's search method
        result = search_engine.perform_search(query=query)
        
        # Display the result
        st.write("Search Results:", result)
    else:
        st.write("Please enter a query.")
    
                
with st.form(key="user_input", clear_on_submit=True):
    # Text input for the query
    query_input = st.text_input("Enter your query", key="user_query")

    # Submit button with handle_submit as the callback
    submit_button = st.form_submit_button(label="Submit", on_click=handle_submit)