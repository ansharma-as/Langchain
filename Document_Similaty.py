from langchain_huggingface import HuggingFaceEmbeddings
from torch.nn.functional import cosine_similarity

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

docs = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "who is virat kohli"

doc_embeddings = embeddings.embed_documents(docs)

query_embeddings = embeddings.embed_query(query)

# print(doc_embeddings)
# print(query_embeddings)
similarities = cosine_similarity(query_embeddings.unsqueeze(0), doc_embeddings)

# print(cosine_similarity([query_embeddings], doc_embeddings))
