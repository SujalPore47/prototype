# vector_database.py

import chromadb
import os
import uuid
from dotenv import load_dotenv
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env
load_dotenv()

class VectorDatabase:
    def __init__(self, document_store_name="Document_store", image_store_name="img_db",
                 document_db_dir="./chroma_langchain_db", image_db_dir="image_db"):
        self.document_store_name = document_store_name
        self.image_store_name = image_store_name
        self.document_db_dir = document_db_dir
        self.image_db_dir = image_db_dir

        # Initialize the embedding function for documents
        self.document_embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
        
        # Initialize the embedding function for images
        self.image_embedding_function = OpenCLIPEmbeddingFunction()
        
        # Initialize Chroma vector store for documents
        self.vector_store = self._initialize_document_store()
        
        # Initialize Chroma persistent client for images
        self.image_store = self._initialize_image_store()

    def _initialize_document_store(self):
        """Initialize and return a Chroma vector store for document embeddings."""
        return Chroma(
            collection_name=self.document_store_name,
            embedding_function=self.document_embedding_function,
            persist_directory=self.document_db_dir
        )

    def _initialize_image_store(self):
        """Initialize and return a persistent client for image embeddings."""
        persistent_client = chromadb.PersistentClient(path=self.image_db_dir)
        img_loader = ImageLoader()
        return persistent_client.get_or_create_collection(
            name=self.image_store_name,
            embedding_function=self.image_embedding_function,
            data_loader=img_loader
        )

    def get_vector_store(self):
        """Return the document vector store instance."""
        return self.vector_store

    def get_image_store(self):
        """Return the image store instance."""
        return self.image_store
