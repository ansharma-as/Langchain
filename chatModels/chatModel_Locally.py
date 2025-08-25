from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="/Users/strontium/Documents/llama models/Llama3.2-3B/",
    temperature=0.5,
    max_tokens=200,
    n_ctx=4096
)

print(llm.invoke("What is the capital of India?"))
