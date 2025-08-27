# long term context through database or text files by loading previous chats with MessagePlaceHolder

from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'), # Adding previous chat context
    ('human', '{query}')
])

chat_history = []
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

prompt = chat_template.invoke({'chat_history': chat_history, 'query':"where is my refund" })

result = model.invoke(prompt)

print(prompt)

print(result)