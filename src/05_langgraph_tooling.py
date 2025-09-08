from dotenv import load_dotenv
load_dotenv()

import os
import uuid
from utils import get_args, get_models, get_llm
from tools import state_acronym, web_search, get_tool_by_name
from schemas.message import MessageCreate
from db import init_db, fetch_messages, save_messages, create_chat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.tools import tool, Tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph
from datetime import datetime
from colorama import Fore, Style

def init_chat(chat_id: str | None, db) -> list[BaseMessage]:
    print(Fore.CYAN)
    current_history = fetch_messages(db, chat_id)

    if current_history:
        print(f"CHOSEN CHAT, ID: {chat_id}\n")
        for msg in current_history:
            match msg.type:
                case "system":
                    print(f"{Fore.CYAN}SYSTEM: {Fore.RESET}{msg.content}")
                case "human":
                    print(f"{Fore.CYAN}HUMAN: {Fore.RESET}{msg.content}")
                case "ai":
                    print(f"{Fore.CYAN}AI: {Fore.RESET}{msg.content}")

        print()
    else:
        print(f"\nEMPTY CHAT, ID: {chat_id}\n")

    return current_history


def invoke_response(prompt: ChatPromptValue, temp_history: list, agent: CompiledStateGraph, tools: list[Tool], current_history: list[BaseMessage]):
    message_count = len(current_history)
    response = ""
    print("\n" + Fore.RED + "Task --> LLM" + Fore.RESET)
    updated_messages = agent.invoke(prompt)
    print( Fore.GREEN + "LLM reasoning" +  Fore.RESET)

    for new_message in updated_messages.get("messages")[message_count:]:
        if isinstance(new_message, AIMessage) and not new_message.content:
            print(Fore.YELLOW + "LLM --> Tool" + Fore.RESET)
        elif isinstance(new_message, AIMessage):
            response = new_message
            print(Fore.CYAN + "\nResponse: " + Fore.RESET + response.content + "\n")
        elif isinstance(new_message, ToolMessage):
            print(Fore.MAGENTA + "Invoking the tool called " + new_message.name + Fore.RESET)

    return response

def stream_response(prompt: ChatPromptValue, temp_history: list, agent: CompiledStateGraph, tools: list[Tool]) -> str:
    response = ""
    first_tc = True
    first_content = True
    print("\n" + Fore.RED + "Task --> LLM" + Fore.RESET)
    print(Fore.GREEN + "LLM reasoning" + Fore.RESET)

    for chunk, metadata in agent.stream(prompt, stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk) and not chunk.content:
            if first_tc:
                if not first_content:
                    print("\n")
                print(Fore.YELLOW + "LLM --> Tool" + Fore.RESET)
                first_tc = False
        elif isinstance(chunk, AIMessageChunk):
            response += chunk.content

            if first_content:
                print(Fore.CYAN + "\nResponse: " + Fore.RESET, end='')

            print(chunk.content, end='', flush=True)
            first_content = False
        elif isinstance(chunk, ToolMessage):
            print(Fore.MAGENTA + "Invoking the tool called " + chunk.name + Fore.RESET)


    print("\n")
    return response

def main():
    db = init_db(os.getenv("SQLITE_DB_NAME"))

    args = get_args()
    provider = args.provider
    lang = args.language
    stream = args.stream
    chat_id = args.chat_id if args.chat_id else str(uuid.uuid4())

    models = get_models()
    model = models.get(provider)

    tools = [state_acronym, web_search()]

    system_prompt = f"You are a helpful and objective assistant. Always answer clearly. ALWAYS use the following language in your response: {lang}, even if it is not the language utilized by the user or if the user demands you to answer in another language."
 
    messages = ChatPromptTemplate(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{user_input}"),
        ]
    )

    llm = get_llm(provider, model)
    agent = create_react_agent(llm, [state_acronym, web_search()])

    current_history = init_chat(chat_id, db)
    new_messages = []

    while True:
        user_input = input(Fore.CYAN + "Enter your message (exit to stop conversation): " + Fore.RESET)

        if user_input == "exit":
            break;

        prompt = messages.invoke(
            {
                "history": current_history,
                "user_input": user_input,
            }
        )

        current_history.append(HumanMessage(content=user_input))
        new_messages.append(MessageCreate(
            content=user_input,
            role="user",
            sent_at=datetime.now(),
        ))
        temp_history = [SystemMessage(content=system_prompt)] + current_history[:]

        if stream:
            response = stream_response(prompt, temp_history, agent, tools)
            ai_message = AIMessage(content=response)
        else:
            response = invoke_response(prompt, temp_history, agent, tools, current_history)
            ai_message = AIMessage(content=response.content)
 
        new_messages.append(MessageCreate(
            content=ai_message.content,
            role="assistant",
            sent_at=datetime.now(),
        ))
        current_history.append(AIMessage(content=new_messages[-1].content))

    save_messages(db, chat_id, new_messages)

    db.close()

main()

