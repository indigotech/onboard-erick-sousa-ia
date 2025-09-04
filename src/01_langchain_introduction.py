from dotenv import load_dotenv
load_dotenv()

from utils import get_args, get_models, get_llm
from langchain_core.messages import SystemMessage, HumanMessage


def main():
    args = get_args()
    provider = args.provider
    lang = args.language
    stream = args.stream

    models = get_models()
    model = models.get(provider)

    system_prompt = f"You are a helpful and objective assistant. Always answer clearly. Always use the following language in your response: {lang}, even if it is not the language utilized by the user."

    user_input = input("Enter your message: ")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    llm = get_llm(provider, model)

    print("\nResponse: \n")
 
    if stream:
        for chunk in llm.stream(messages):
            print(chunk.content, end='', flush=True)
    else:
        response = llm.invoke(messages)
        print(response.content)
 
    print()

main()
