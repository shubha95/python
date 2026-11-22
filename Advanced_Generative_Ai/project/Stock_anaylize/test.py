import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

#launching-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI

# load api key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

# initialize the model with the API key

llm = ChatGoogleGenerativeAI(
    api_key=api_key,
    model="gemini-3.8-flash",
    temperature=0.7,
 )

print(llm.invoke("Say hello").content[0].text)
