from langchain_huggingface import ChatHuggingFace ,  HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "quit" or user_input == "exit":
        break
    else:
        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))
        print("AI: ", result.content)

print(chat_history)