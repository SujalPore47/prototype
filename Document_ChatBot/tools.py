from crewai_tools import tool 
from database import VectorDatabase

vector_db = VectorDatabase()

# Access the document and image stores
document_store = vector_db.get_vector_store()
image_store = vector_db.get_image_store()


@tool("VectorStoreSearchTool")
def VectorStoreSearchTool(topic: str) -> str:
    """
    Retrieves relevant context from a vector store based on similarity scores
    and generates a response that can be used by the parent language model (LLM).
    """
    # Retrieve context from vector store using similarity scores
    context = document_store.similarity_search_with_score(topic),
    return context[0][0][0].page_content

#answer = image_store.query(query_texts ="birds", n_results=1 , include=['data'])
#image = reconstruct_image_from_array(answer['data'])