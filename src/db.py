import sqlite3
import uuid
import os
from schemas.message import MessageCreate
from schemas.chat import ChatPublic
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from typing import List


def init_db(name: str) -> sqlite3.Connection:
    db_path = os.path.join(os.getcwd(), "src", "data", name)

    db = sqlite3.connect(db_path)
    cursor = db.cursor()

    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                context TEXT
            );
        """
    )

    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT,
                sent_at TIMESTAMP,
                role TEXT,
                content TEXT,
                FOREIGN KEY(chat_id) REFERENCES chats(id)
            );
        """
    )

    db.commit()
    return db


def create_chat(db: sqlite3.Connection, chat_id: str | None) -> str:
    id = chat_id or str(uuid.uuid4())

    statement = """INSERT INTO chats (id,context) VALUES (?,?)"""

    cursor = db.cursor()
    db_chat = cursor.execute(
        statement,
        (
            chat_id,
            "",
        ),
    )
    db.commit()

    return chat_id


def save_messages(db: sqlite3.Connection, chat_id: str, messages: List[MessageCreate]):
    cursor = db.cursor()
    statement = """INSERT INTO messages (id, chat_id, sent_at, role, content) VALUES (?, ?, ?, ?, ?)"""
    for msg in messages:
        cursor.execute(
            statement, (str(uuid.uuid4()), chat_id, msg.sent_at, msg.role, msg.content)
        )

    db.commit()


def fetch_messages(db: sqlite3.Connection, chat_id: str) -> List[BaseMessage]:
    cursor = db.cursor()
    statement = """SELECT * FROM messages WHERE chat_id = (?) ORDER BY sent_at ASC"""
    cursor.execute(statement, (chat_id,))

    messages = []

    for id, chat_id, sent_at, role, content in cursor.fetchall():
        match role:
            case "system":
                messages.append(SystemMessage(content=content))
            case "user":
                messages.append(HumanMessage(content=content))
            case "assistant":
                messages.append(AIMessage(content=content))

    return messages


# for script with RunnableWithMessageHistory
def fetch_messages_for_runnable(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id, connection=f"sqlite:///./src/data/{os.getenv('SQLITE_DB_NAME')}"
    )


def fetch_chat(db: sqlite3.Connection, chat_id: str) -> ChatPublic:
    cursor = db.cursor()
    messages_statement = (
        """SELECT * FROM messages WHERE chat_id = (?) ORDER BY sent_at ASC"""
    )
    context_statement = """SELECT context FROM chats WHERE id = (?)"""
    cursor.execute(messages_statement, (chat_id,))

    messages = []

    for id, chat_id, sent_at, role, content in cursor.fetchall():
        match role:
            case "system":
                messages.append(SystemMessage(content=content))
            case "user":
                messages.append(HumanMessage(content=content))
            case "assistant":
                messages.append(AIMessage(content=content))

    cursor.execute(context_statement, (chat_id,))
    context = cursor.fetchone()

    return ChatPublic(messages=messages, context=str(context))


def update_context(db: sqlite3.Connection, chat_id: str, context: str):
    cursor = db.cursor()
    statement = """UPDATE chats SET context = ? WHERE id = ?"""
    cursor.execute(
        statement,
        (context, chat_id),
    )
    db.commit()
