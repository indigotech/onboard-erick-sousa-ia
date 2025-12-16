from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List

from langgraph.graph import MessagesState


class ChosenTeam(Enum):
    STOCK = "stock_graph"
    POEM = "poem_graph"
    NONE = "context_manager_agent"


class Router(BaseModel):
    """
    Team to route to next.
    """

    chosen_team: ChosenTeam = Field(
        default=ChosenTeam.NONE, description="Team to route to next."
    )


class StockInfo(BaseModel):
    company_name: str
    ticker: str
    price: Optional[str]
    news_summary: Optional[str]


class Context(BaseModel):
    """
    Context of the whole conversation, containing relevant information based on previous context and messages.
    """

    summary: str = Field(
        description="Contains relevant information based on previous context and messages.",
    )
    stock_info: Optional[List[StockInfo]] = Field(
        description="Contains information about every stock the user has asked about.",
    )
    user_name: Optional[str] = Field(
        description="The name of the user.",
    )


class State(MessagesState):
    context: Context
