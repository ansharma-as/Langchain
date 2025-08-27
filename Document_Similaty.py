from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

docs = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "who is Bumrah"

doc_embeddings = embeddings.embed_documents(docs)
query_embeddings = embeddings.embed_query(query)

# print(doc_embeddings)
# print(query_embeddings)
# print(cosine_similarity([query_embeddings], doc_embeddings)[0])

result = cosine_similarity([query_embeddings], doc_embeddings)[0]
# print(sorted(list(enumerate(result)), key=lambda x: x[1])[-1])
# by doing this we are giving index to each similarity score bcz they come randomly after cosine_similarity

index, score = sorted(list(enumerate(result)), key=lambda x: x[1])[-1]

print(query)
print(docs[index])

print("Similarity Score is: ", score)

'''

    what we are doing in this Document Similarity ?
  
    we are doing semantic search (similarity search)  based on scores in the docs array 
    and finding which document is most similar to the query based on scores (This process is called retrieval Process)
   
   This is the basic RAG flow
'''