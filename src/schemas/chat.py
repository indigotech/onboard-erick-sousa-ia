from datetime import datetime
from pydantic import BaseModel
from typing import List
from langchain_core.messages import BaseMessage


class ChatPublic(BaseModel):
    messages: List[BaseMessage]
    context: str
