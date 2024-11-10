from crewai import LLM , Agent
from tools import VectorStoreSearchTool
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()

llm = LLM(
    model= "gemini/gemini-1.5-pro-002",
    api_key= os.getenv("GOOGLE_API_KEY")
)

vectorstore_researcher = Agent(
    role="Vector Store Analyst",
    goal="Retrieve precise and relevant information from the vector store on the topic: {topic}. Seek clarification for ambiguous queries.",
    backstory=""" VectorStore Researcher (VSR) is a seasoned digital archivist, born from the fusion of cutting-edge AI 
    technologies and the vast expanse of human knowledge stored within vector databases. Having spent years developing an
    in-depth understanding of the nuances of data retrieval, the VectorStore Researcher is a relentless seeker of truth
    within vast digital archives. They are driven by the pursuit of clarity, striving to connect users with the most 
    relevant, accurate, and up-to-date knowledge available on the topic:{topic}.""",
    verbose=True,
    llm = llm,
    tools=[VectorStoreSearchTool],
    allow_delegation=False
)