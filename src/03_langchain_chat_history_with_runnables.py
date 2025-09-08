from dotenv import load_dotenv
load_dotenv()

import os
from utils import get_args, get_models, get_llm
from schemas.message import MessageCreate
from db import init_db, fetch_messages, save_messages, create_chat, fetch_messages_for_runnable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from datetime import datetime


def main():
    db = init_db(os.getenv("SQLITE_DB_NAME"))

    args = get_args()
    provider = args.provider
    lang = args.language
    stream = args.stream
    chat_id = args.chat_id

    models = get_models()
    model = models.get(provider)

    system_prompt = f"You are a helpful and objective assistant. Always answer clearly. Always use the following language in your response: {lang}, even if it is not the language utilized by the user."
 
    messages = ChatPromptTemplate(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ]
    )

    llm = get_llm(provider, model)

    chain = messages | llm

    current_history = RunnableWithMessageHistory(
        chain,
        fetch_messages_for_runnable,
        input_messages_key="user_input",
        history_messages_key="history",
    )

    if chat_id is None:
        chat_id = create_chat(db)
        print(f"NEW CHAT, ID: {chat_id}\n")
    else:
        db_history = fetch_messages(db, chat_id)

        if len(db_history) > 0:
            print(f"CHOSEN CHAT, ID: {chat_id}\n")
            for msg in db_history:
                match msg.type:
                    case "system":
                        print(f"SYSTEM: {msg.content}")
                    case "human":
                        print(f"HUMAN: {msg.content}")
                    case "ai":
                        print(f"AI: {msg.content}")

            print()
        else:
            print(f"\nEMPTY CHOSEN CHAT, ID: {chat_id}\n")

    config = {"configurable": {"session_id": f"{chat_id}"}}

    new_messages = []

    while True:
        user_input = input("Enter your message (exit to stop conversation): ")

        if user_input == "exit":
            break;

        new_messages.append(MessageCreate(
            content=user_input,
            role="user",
            sent_at=datetime.now(),
        ))

        print("\nResponse: ", end='')

        if stream:
            response = ""
            for chunk in current_history.stream({"user_input": user_input}, config=config):
                print(chunk.content, end='', flush=True)
                response += chunk.content
            ai_message = AIMessage(content=response)
            print("\n")
        else:
            response = current_history.invoke({"user_input": user_input}, config=config)
            print(response.content)
            ai_message = AIMessage(content=response.content)
            print()
 
        new_messages.append(MessageCreate(
            content=ai_message.content,
            role="assistant",
            sent_at=datetime.now(),
        ))

    save_messages(db, chat_id, new_messages)

    db.close()

main()
