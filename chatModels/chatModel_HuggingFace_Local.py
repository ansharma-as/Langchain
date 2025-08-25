from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
import os

import os
os.environ["HF_HOME"] = "/Users/strontium/Desktop/Langchain/models"

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )   # pipeline keyword arguments
)

model = ChatHuggingFace(llm= llm)

result = model.invoke("what is the capital of India")

print(result)

print("-------------------------------------------------------------------------------")

print()
print(result.content)