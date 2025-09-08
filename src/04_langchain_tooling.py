from dotenv import load_dotenv
load_dotenv()

import os
import uuid
from utils import get_args, get_models, get_llm
from tools import state_acronym, web_search, get_tool_by_name
from schemas.message import MessageCreate
from db import init_db, fetch_messages, save_messages, create_chat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.tools import tool, Tool
from langchain_openai import ChatOpenAI
from datetime import datetime
from colorama import Fore, Style, Back

def init_chat(chat_id: str, db) -> list[BaseMessage]:
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

def invoke_response(prompt: ChatPromptValue, temp_history: list, llm: ChatOpenAI, tools: list[Tool]):
    print("\n" +  Fore.RED + "STEP 1: Task --> LLM" +  Fore.RESET)
    response = llm.invoke(prompt)
    print( Fore.GREEN + "STEP 2: LLM reasoning" +  Fore.RESET)

    if response.tool_calls:
        print( Fore.YELLOW + "STEP 3: LLM --> Tool" +  Fore.RESET)
        temp_history.append(response)
        for tool_call in response.tool_calls:
            selected_tool = get_tool_by_name(tools, tool_call["name"])
            tool_result = selected_tool.invoke(tool_call)
            print( Fore.BLUE + "STEP 4: Action. Invoking the tool called " + tool_call["name"] +  Fore.RESET)
            temp_history.append(tool_result)
            print( Fore.MAGENTA + "STEP 5: Result. Tool call completed." +  Fore.RESET)

        print( Fore.CYAN + "STEP 6: Tool --> LLM")
        response = llm.invoke(temp_history)
        print(Back.WHITE + Fore.BLACK + "STEP 7: LLM --> Response" + Back.RESET + Fore.RESET)
 
    print(Fore.CYAN + "\nResponse: " + Fore.RESET + response.content)
    print()
    return response

def stream_response(prompt: ChatPromptValue, temp_history: list, llm: ChatOpenAI, tools: list[Tool]) -> str:
    response = ""
    first_tc = True
    first_content = True
    print("\n" +  Fore.RED + "STEP 1: Task --> LLM" +  Fore.RESET)

    for chunk in llm.stream(prompt):
        if first_tc:
            gathered = chunk
            first_tc = False
        else:
            gathered = gathered + chunk

        if chunk.content:
            response += chunk.content

            if first_content:
                print(Fore.CYAN + "\nResponse: " + Fore.RESET, end='')

            print(chunk.content, end='', flush=True)
            first_content = False

    print()

    print( Fore.GREEN + "STEP 2: LLM reasoning" +  Fore.RESET)

    if gathered.tool_calls:
        print( Fore.YELLOW + "STEP 3: LLM --> Tool" +  Fore.RESET)
        temp_history.append(gathered)
        for tool_call in gathered.tool_calls:
            selected_tool = get_tool_by_name(tools, tool_call["name"])
            tool_result = selected_tool.invoke(tool_call)
            print( Fore.BLUE + "STEP 4: Action. Invoking the tool called " + tool_call["name"] +  Fore.RESET)
            temp_history.append(tool_result)
            print( Fore.MAGENTA + "STEP 5: Result. Tool call completed." +  Fore.RESET)

        print( Fore.CYAN + "STEP 6: Tool --> LLM")
        print(Back.WHITE + Fore.BLACK + "STEP 7: LLM --> Response" + Back.RESET + Fore.RESET)
        print(Fore.CYAN + "\nResponse: " + Fore.RESET, end='')

        for chunk in llm.stream(temp_history):
            response += chunk.content
            print(chunk.content, end='', flush=True)

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

    llm = llm.bind_tools(tools)

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
            response = stream_response(prompt, temp_history, llm, tools)
            ai_message = AIMessage(content=response)
        else:
            response = invoke_response(prompt, temp_history, llm, tools)
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
