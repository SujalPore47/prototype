from crewai import Task
from agents import vectorstore_researcher
from tools import VectorStoreSearchTool

document_retrieval = Task(
    agent=vectorstore_researcher,
    tools=[VectorStoreSearchTool],
    description="Retrieve relevant content from the vector store for document analysis on the topic: {topic}.",
    expected_output="""
        - Concise document summary.
        - Key insights and major points on {topic}.
        - Source links or identifiers, if applicable.
        - Additional contextual info if relevant.
    """
)
