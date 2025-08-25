from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Delhi is tha capital of India",
    "Kolkata is a port city",
    "Bangalore is a Silicon Valley"
]

## result = embedding.embed_query("delhi is the capital city of India") # used for generating embedding for a Query
result = embedding.embed_documents(documents)

print(str(result))