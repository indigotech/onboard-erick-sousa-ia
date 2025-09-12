from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, Tool
from data.brazillian_states import state_acronyms
from pydantic import BaseModel, Field
import yfinance as yf


class StateAcronymInput(BaseModel):
    state_name: str = Field(
        description="The name of the state in brazillian portuguese, with beggining capital letters and all the accents needed. If the user input cannot be parsed as a BRAZILLIAN state, it CANNOT be used in this tool."
    )


@tool(args_schema=StateAcronymInput)
def state_acronym(state_name: str) -> str:
    """Get brazillian state acronym"""

    return state_acronyms[state_name]


class GetStockPriceInput(BaseModel):
    ticker_symbols: list[str] = Field(
        description="A list of the unique series of characters that identify a stock, as an abbreviation."
    )


@tool(args_schema=GetStockPriceInput)
def stock_price(ticker_symbols: list[str]) -> list[dict]:
    """Get the price of a stock"""
    all_prices = []

    for ts in ticker_symbols:
        ticker = yf.Ticker(ts)
        all_prices.append(ticker.history(period="1d", auto_adjust=False))

    return all_prices


def web_search() -> Tool:
    return DuckDuckGoSearchRun()


def get_tool_by_name(tools: list[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
