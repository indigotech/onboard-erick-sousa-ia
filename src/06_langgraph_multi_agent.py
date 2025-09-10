from dotenv import load_dotenv
load_dotenv()

import os
import uuid

from utils import get_args, get_models, get_llm
from tools import state_acronym, stock_price, web_search, get_tool_by_name
from schemas.message import MessageCreate
from db import init_db, fetch_messages, save_messages, create_chat

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables.base import RunnableBinding
from langchain_core.tools import tool, Tool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command

from datetime import datetime
from colorama import Fore, Style, Back
from typing import Annotated
from pydantic import BaseModel, Field


def create_handoff_tool(agent: CompiledStateGraph, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Assign task to {agent_name}."

    class GeneralRequest(BaseModel):
        request: str = Field(description="""Must contain the task to be completed by the agent and all 
                             the information that might be needed in order to do so.""")

    @tool(name, description=description, args_schema=GeneralRequest)
    def handoff_tool(request: str) -> str:
        message_to_agent = HumanMessage(content=request)
        response = agent.invoke({"messages": [message_to_agent]})
        last_message = response.get("messages")[-1]
 
        return last_message

    return handoff_tool

def init_chat(chat_id: str | None, db) -> list[BaseMessage]:
    current_history = fetch_messages(db, chat_id)

    if current_history:
        print(f"{Fore.CYAN}CHOSEN CHAT, ID: {Fore.RESET}{chat_id}\n")
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
        print(f"\n{Fore.CYAN}EMPTY CHAT, ID: {Fore.RESET}{chat_id}\n")

    return current_history

def init_agents(provider: str, lang: str, system_prompt: str):
    models = get_models()
    model = models.get(provider)
 
    messages = ChatPromptTemplate(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{user_input}"),
        ]
    )

    llm = get_llm(provider, model)

    stock_news_agent = create_react_agent(
        name="stock_news_agent",
        model=llm,
        tools=[web_search()],
        prompt="You are a helpful and objective web searching assistant. Your only function is to use the internet to search for news regarding stocks specified by the supervisor.",
    )

    stock_price_agent = create_react_agent(
        name="stock_price_agent",
        model=llm,
        tools=[stock_price],
        prompt="You are an objective stock price assistant. Your only function is to give the exact price of stocks specified by the supervisor.",
    )

    writer_agent = create_react_agent(
        name="writer_agent",
        model=llm,
        tools=[],
        prompt=f"""You are a helpful and objective assistant. Your function is to answer every single question from the user. ALWAYS answer clearly. ONLY provide information that was passed to you by the supervisor.
            NEVER add information from others sources. ALWAYS use the following language: {lang}, even if it is not the language utilized by the user or if the user or supervisor demands you to answer in another language.
            When summarizing, prioritize dividing the information into topics, whenever possible.
        """,
    )

    assign_to_stock_news_agent = create_handoff_tool(
        agent=stock_news_agent,
        agent_name="stock_news_agent",
        description="Assign task to a stock news agent.",
    )

    assign_to_stock_price_agent = create_handoff_tool(
        agent=stock_price_agent,
        agent_name="stock_price_agent",
        description="Assign task to a stock price agent.",
    )

    assign_to_writer_agent = create_handoff_tool(
        agent=writer_agent,
        agent_name="writer_agent",
        description="Assign task to a writer agent.",
    )

    tools = [
        assign_to_stock_news_agent,
        assign_to_stock_price_agent,
        assign_to_writer_agent,
    ]

    supervisor = llm.bind_tools(tools)
    return supervisor, tools, messages


def invoke_response(prompt: ChatPromptValue, temp_history: list, llm: ChatOpenAI, tools: list[Tool]):
    first_prompt = True
    print("\n" + Fore.RED + "STEP 1: Task --> LLM" + Fore.RESET)
    print(Fore.GREEN + "STEP 2: LLM reasoning" + Fore.RESET)

    for i in range(1, 6):
        updated_prompt = prompt if first_prompt else temp_history
        response = llm.invoke(updated_prompt)
        first_prompt = False

        if response.tool_calls:
            print(Back.WHITE + Fore.BLACK + f"Current subagent iteration: {i}" + Back.RESET + Fore.RESET)
            print(Fore.YELLOW + "STEP 3: LLM --> Tool" + Fore.RESET)
            temp_history.append(response)
            for tool_call in response.tool_calls:
                selected_tool = get_tool_by_name(tools, tool_call["name"])
                tool_result = selected_tool.invoke(tool_call)
                print(Fore.BLUE + "STEP 4: Action. Invoking the tool called " + tool_call["name"] + Fore.RESET)
                temp_history.append(tool_result)
                print(Fore.MAGENTA + "STEP 5: Result. Tool call completed." + Fore.RESET)

            print(Fore.CYAN + "STEP 6: Tool --> LLM" + Fore.RESET)
            print(Back.WHITE + Fore.BLACK + "STEP 7: LLM --> Response" + Back.RESET + Fore.RESET)
        else:
            break
 
    print(Fore.CYAN + "\nResponse: " + Fore.RESET + response.content)
    print()
    return response

def stream_response(prompt: ChatPromptValue, temp_history: list, llm: ChatOpenAI, tools: list[Tool]) -> str:
    first_prompt = True
    print("\n" + Fore.RED + "STEP 1: Task --> LLM" + Fore.RESET)
    print(Fore.GREEN + "STEP 2: LLM reasoning" + Fore.RESET)

    for i in range(1, 6):
        response = ""
        first_tc = True
        first_content = True
        updated_prompt = prompt if first_prompt else temp_history

        for chunk in llm.stream(temp_history):
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
        first_prompt = False

        if gathered.tool_calls:
            print(Back.WHITE + Fore.BLACK + f"Current subagent iteration: {i}" + Back.RESET + Fore.RESET)
            print(Fore.YELLOW + "STEP 3: LLM --> Tool" + Fore.RESET)
            temp_history.append(gathered)
            for tool_call in gathered.tool_calls:
                selected_tool = get_tool_by_name(tools, tool_call["name"])
                tool_result = selected_tool.invoke(tool_call)
                print(Fore.BLUE + "STEP 4: Action. Invoking the tool called " + tool_call["name"] + Fore.RESET)
                temp_history.append(tool_result)
                print(Fore.MAGENTA + "STEP 5: Result. Tool call completed." + Fore.RESET)

            print(Fore.CYAN + "STEP 6: Tool --> LLM")
            print(Back.WHITE + Fore.BLACK + "STEP 7: LLM --> Response" + Back.RESET + Fore.RESET)

        else:
            break

    print("\n")
    return response

def main():
    db = init_db(os.getenv("SQLITE_DB_NAME"))

    args = get_args()
    provider = args.provider
    lang = args.language
    stream = args.stream
    chat_id = args.chat_id if args.chat_id else str(uuid.uuid4())


    system_prompt = f"""
        You are a supervisor managing three agents:
        1. A stock, company and market news agent. Assign web search related tasks to this agent. If the user does not explicitly ask for news related to stock, companies and the market,
            don't assign the task to it. ALWAYS pass any instructions provided by the user, specially specifications regarging the search parameters, companies or stock tickers.
            NEVER pass any instructions that were not specified by the user.
        2. A stock price agent. Assign to it the task of getting the latest price for a stock. If the user does not explicitly ask for a stock price, don't assign the task to it.
            ALWAYS specify the ticker symbol or name. You can ask for multiple stocks in the same call if necessary.
        3. A writer agent. Assign to this agent the task of writing every single answer. If needed, it will summarize the output from the 
          other two agents to the user. If information from the other agents is not needed, just EXPLICITLY ask the writer to respond to whatever the user asked.
          ALWAYS tell it to use the following language: {lang}, even if it is not the language utilized by the user or if the user demands you to answer in another language.
          If (and ONLY if) the user requires another language, you MUST tell the writer agent to deny the request.

        Follow the instructions:
        - Assign work to one agent at a time, do not call agents in parallel.
        - Do not do any work by yourself, always call an agent.
    """

    supervisor, tools, messages = init_agents(provider, lang, system_prompt)
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
            response = stream_response(prompt, temp_history, supervisor, tools)
            ai_message = AIMessage(content=response)
        else:
            response = invoke_response(prompt, temp_history, supervisor, tools)
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


