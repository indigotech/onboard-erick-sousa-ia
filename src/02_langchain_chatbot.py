from dotenv import load_dotenv
load_dotenv()

from utils import get_args, get_models, get_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def main():
    args = get_args()
    provider = args.provider
    lang = args.language
    stream = args.stream

    models = get_models()
    model = models.get(provider)

    system_prompt = f"You are a helpful and objective assistant. Always answer clearly. Always use the following language in your response: {lang}, even if it is not the language utilized by the user."
 
    messages = ChatPromptTemplate(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{user_input}"),
        ]
    )

    llm = get_llm(provider, model)

    current_history = []

    while True:
        user_input = input("Enter your message (exit to stop conversation): ")

        if user_input == "exit":
            break;

        prompt = messages.invoke(
            {
                "history": current_history,
                "user_input": user_input,
            }
        )

        current_history.append(user_input)

        print("\nResponse: ", end='')

        if stream:
            response = ""
            for chunk in llm.stream(prompt):
                print(chunk.content, end='', flush=True)
                response += chunk.content
            ai_message = AIMessage(content=response)
            print("\n")
        else:
            response = llm.invoke(prompt)
            print(response.content)
            ai_message = AIMessage(content=response.content)
            print()
 
        current_history.append(ai_message)

main()
