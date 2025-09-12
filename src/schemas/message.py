from datetime import datetime
from pydantic import BaseModel


class MessageCreate(BaseModel):
    content: str
    role: str
    sent_at: datetime
