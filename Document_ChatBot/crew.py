from crewai.process import Process
from crewai import Crew
from agents import vectorstore_researcher
from tasks import document_retrieval
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class SearchEngine:
    def __init__(self, cache=True, verbose=True):
        """Initialize the Crew instance with specific agents, tasks, and configurations."""
        self.crew = Crew(
            agents=[vectorstore_researcher],
            tasks=[document_retrieval],
            process=Process.sequential,
            cache=cache,
            verbose=verbose,
        )

    def perform_search(self, query):
        """Perform a search with the specified query and return the result."""
        result = self.crew.kickoff(inputs={'topic': query})
        return result.raw
