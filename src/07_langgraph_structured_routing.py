from dotenv import load_dotenv

load_dotenv()

import os
import uuid

from utils import get_args, get_models, get_llm
from tools import stock_price, web_search, get_tool_by_name
from schemas.message import MessageCreate
from db import init_db, fetch_messages, save_messages, create_chat

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables.base import RunnableBinding
from langchain_core.tools import tool, Tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, MessagesState

from colorama import Fore, Style, Back
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class ChosenTeam(Enum):
    STOCK = "stock_graph"
    POEM = "poem_graph"
    NONE = "__end__"


class Router(BaseModel):
    """
    Team to route to next.
    """

    chosen_team: ChosenTeam


def router_func(llm: ChatOpenAI, fallback: str, state: MessagesState) -> str:
    response = llm.with_structured_output(Router).invoke([state["messages"][-1]])
    choice = response.chosen_team.value

    if choice == ChosenTeam.NONE.value:
        print(Fore.CYAN + "\nResponse: " + Fore.RESET + fallback + "\n")
    else:
        print(
            f"\n{Fore.RED}--- Chosen graph: {Fore.YELLOW}{choice}{Fore.RED} ---{Fore.RESET}\n"
        )

    return choice


def create_handoff_tool(
    agent: CompiledStateGraph, agent_name: str, description: str | None = None
) -> Tool:
    name = f"transfer_to_{agent_name}"
    description = description or f"Assign task to {agent_name}."

    class GeneralRequest(BaseModel):
        request: str = Field(
            description="""Must contain the task to be completed by the agent and all 
                             the information that might be needed in order to do so."""
        )

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


def init_graph(llm: ChatOpenAI, lang: str, fallback: str) -> CompiledStateGraph:
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
            If another language is demanded by the user, respond by telling that you can you use the language {lang}. When summarizing, prioritize dividing the information into topics, whenever possible.
        """,
    )

    poem_agent = create_react_agent(
        name="poem_agent",
        model=llm,
        tools=[],
        prompt=f"""You are a poem writer agent. Your function is to write poems based on provided intructions.
            ALWAYS use the following language: {lang}, even if it is not the language utilized by the user or if the user or supervisor demands you to answer in another language.
            If another language is demanded by the user, respond by telling that you can you use the language {lang}
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

    assign_to_poem_agent = create_handoff_tool(
        agent=poem_agent,
        agent_name="poem_agent",
        description="Assign task to a poem writer agent.",
    )

    stock_team_tools = [
        assign_to_stock_news_agent,
        assign_to_stock_price_agent,
        assign_to_writer_agent,
    ]

    poem_team_tools = [
        assign_to_poem_agent,
    ]

    stock_team_supervisor = create_react_agent(
        name="stock_team_supervisor",
        model=llm,
        tools=stock_team_tools,
        prompt=f"""
            You are a supervisor managing three agents:
            1. A stock, company and market news agent. Assign web search related tasks to this agent. If the user does not explicitly ask for news related to stock, companies and the market,
                don't assign the task to it. ALWAYS pass any instructions provided by the user, specially specifications regarging the search parameters, companies or stock tickers.
                NEVER pass any instructions that were not specified by the user.
            2. A stock price agent. Assign to it the task of getting the latest price for a stock. If the user does not explicitly ask for a stock price, don't assign the task to it.
                ALWAYS specify the ticker symbol or name. You can ask for multiple stocks in the same call if necessary.
            3. A writer agent. Assign to this agent the task summarizing the output of the other agents. The writer agent's task is to summarize the information given by the stock news and 
                price agents in a clear and structured way. Never do the summarization yourself, always send to the writer a clear, specific message that includes most of the content from ther other agent's responses. Never send 
                generic requests such as 'summarize this' without giving the information obtained from the other agents. 

            Follow the instructions:
            - Assign work to one agent at a time, do not call agents in parallel.
            - Do not do any work by yourself, always call an agent.
            - The typical flow should be to first call the stock news and or the stock price agent, then gather all the obtained information as an input for the writer agent.
            - ALWAYS use the following language: {lang}, even if it is not the language utilized by the user or if the user demands you to answer in another language.
            - If another language is demanded by the user, respond by telling that you can you use the language {lang}
        """,
    )

    poem_team_supervisor = create_react_agent(
        name="poem_team_supervisor",
        model=llm,
        tools=poem_team_tools,
        prompt=f"""
            You are a supervisor managing one agent:
            1. A poem writer agent. Assign to this agent the task of writing a poem.

            Follow the instructions:
            - Always use the tool "transfer_to_poem_agent"
            - Do not do any work by yourself, always call the poem writer agent.
            - ALWAYS use the following language: {lang}, even if it is not the language utilized by the user or if the user demands you to answer in another language.
            - If another language is demanded by the user, respond by telling that you can you use the language {lang}
        """,
    )

    stock_builder = StateGraph(MessagesState)
    stock_builder.add_node("stock_team_supervisor", stock_team_supervisor)
    stock_builder.add_node("stock_news_agent", stock_news_agent)
    stock_builder.add_node("stock_price_agent", stock_price_agent)
    stock_builder.add_node("writer_agent", writer_agent)
    stock_builder.add_edge(START, "stock_team_supervisor")
    stock_graph = stock_builder.compile()

    poem_builder = StateGraph(MessagesState)
    poem_builder.add_node("poem_team_supervisor", poem_team_supervisor)
    poem_builder.add_node("poem_agent", poem_agent)
    poem_builder.add_edge(START, "poem_team_supervisor")
    poem_graph = poem_builder.compile()

    super_builder = StateGraph(MessagesState)
    super_builder.add_node("stock_graph", stock_graph)
    super_builder.add_node("poem_graph", poem_graph)
    super_builder.add_conditional_edges(
        START, lambda state: router_func(llm, fallback, state)
    )
    super_graph = super_builder.compile()

    return super_graph


def invoke_response(
    prompt: ChatPromptValue,
    current_history: list,
    graph: CompiledStateGraph,
) -> str | None:
    response = graph.invoke(prompt)
    last_message = response["messages"][-1]

    if isinstance(last_message, HumanMessage):
        return None

    current_history.append(last_message)
    print(Fore.CYAN + "\nResponse: " + Fore.RESET + last_message.content + "\n")
    return last_message.content


def stream_response(
    prompt: ChatPromptValue,
    current_history: list,
    graph: CompiledStateGraph,
) -> str:
    response = graph.invoke(prompt)
    last_message = response["messages"][-1]

    if isinstance(last_message, HumanMessage):
        return None

    current_history.append(last_message)
    print(Fore.CYAN + "\nResponse: " + Fore.RESET + last_message.content + "\n")
    return last_message.content


def main():
    db = init_db(os.getenv("SQLITE_DB_NAME"))

    args = get_args()
    chat_id = args.chat_id if args.chat_id else str(uuid.uuid4())

    model = get_models().get(args.provider)
    llm = get_llm(args.provider, model)

    fallback = "Please ask about poems or stocks."
    graph = init_graph(llm, args.language, fallback)
    current_history = init_chat(chat_id, db)
    messages = ChatPromptTemplate(
        [
            MessagesPlaceholder("history"),
            ("human", "{user_input}"),
        ]
    )
    new_messages = []

    while True:
        user_input = input(
            Fore.CYAN + "Enter your message (exit to stop conversation): " + Fore.RESET
        )

        if user_input == "exit":
            break

        prompt = {"messages": current_history + [HumanMessage(content=user_input)]}

        current_history.append(HumanMessage(content=user_input))
        new_messages.append(
            MessageCreate(
                content=user_input,
                role="user",
                sent_at=datetime.now(),
            )
        )

        if args.stream:
            response = stream_response(
                prompt, current_history, graph
            )  # DOES THE SAME AS invoke_response
        else:
            response = invoke_response(prompt, current_history, graph)

        if not response:
            response = fallback

        new_messages.append(
            MessageCreate(
                content=response,
                role="assistant",
                sent_at=datetime.now(),
            )
        )

    save_messages(db, chat_id, new_messages)
    db.close()


main()
