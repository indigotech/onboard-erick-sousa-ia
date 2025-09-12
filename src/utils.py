import argparse
import os
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI


def get_models() -> dict:
    return {
        "openai": "gpt-4.1",
        "deepseek": "deepseek/deepseek-r1:free",
        "meta-llama": "meta-llama/llama-4-maverick:free",
    }


def get_llm(provider: str, model: str) -> ChatOpenAI:
    openrouter_url = os.getenv("OPENROUTER_URL")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if provider == "openai":
        llm = init_chat_model(model, model_provider=provider)
    else:
        llm = ChatOpenAI(
            api_key=openrouter_api_key,
            base_url=openrouter_url,
            model=model,
        )

    return llm


def get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--provider",
        choices=["openai", "deepseek", "meta-llama"],
        help="Chooses the LLM Vendor.",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--language",
        choices=[
            "portuguese",
            "english",
            "japanese",
            "indonesian",
            "german",
            "spanish",
        ],
        help="Chooses the response's language.",
        required=True,
    )
    parser.add_argument(
        "-c", "--chat_id", help="Specifies an already existing conversation"
    )
    parser.add_argument(
        "-s",
        "--stream",
        help="Generate the response in real time.",
        action="store_true",
    )

    return parser.parse_args()
