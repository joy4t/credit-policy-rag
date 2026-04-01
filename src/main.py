import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load API key from .env file
load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# First test call
response = llm.invoke("What is a Non-Performing Asset in Indian banking? Answer in 2 sentences.")

print(response.content)