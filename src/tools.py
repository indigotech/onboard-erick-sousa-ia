from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, Tool
from data.brazillian_states import state_acronyms
from pydantic import BaseModel, Field

class StateAcronymInput(BaseModel):
    state_name: str = Field(description="The name of the state in brazillian portuguese, with beggining capital letters and all the accents needed")

@tool(args_schema=StateAcronymInput)
def state_acronym(state_name: str) -> str:
    """ Get brazillian state acronymm """

    return state_acronyms[state_name]


def web_search() -> Tool:
    return DuckDuckGoSearchRun()

def get_tool_by_name(tools: list[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
