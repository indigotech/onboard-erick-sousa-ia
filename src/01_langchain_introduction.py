import os
import argparse
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--provider", choices=["openai", "deepseek", "meta-llama"],
                    help="Chooses the LLM Vendor.", required=True)
parser.add_argument("-l", "--language", choices=["portuguese", "english", "japanese", "indonesian", "german", "spanish"],
                    help="Chooses the response's language.", required=True)
parser.add_argument(
    "-s", "--stream", help="Generate the response in real time.", action="store_true")

args = parser.parse_args()
provider = args.provider
lang = args.language
stream = args.stream

openrouter_url = os.getenv("OPENROUTER_URL")
models = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek/deepseek-r1:free",
    "meta-llama": "meta-llama/llama-4-maverick:free",
}

model = models.get(provider)

system_prompt = f"You are a helpful and objective assistant. Always answer clearly. Always use the following language in your response: {lang}, even if it is not the language utilized by the user."
user_input = input("Enter your message: ")

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_input)
]

if provider == "openai":
    llm = init_chat_model(model, model_provider=provider)
else:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    llm = ChatOpenAI(
        api_key=openrouter_api_key,
        base_url=openrouter_url,
        model=model,
    )

print("\nResponse: \n")

if stream:
    for chunk in llm.stream(messages):
        print(chunk.content, end='', flush=True)
else:
    response = llm.invoke(messages)
    print(response.content)

print()
