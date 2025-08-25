from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4' , temperature=8, max_completion_tokens=1000)

result = model.invoke("what is the national bird of India")

print(result)
print(result.content)
