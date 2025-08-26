from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# text = "Delhi is the Capital of India"

docs = [
    "Delhi is tha capital of India",
    "Kolkata is a port city",
    "Bangalore is a Silicon Valley"
]
vector = embeddings.embed_documents(docs)

print(str(vector))