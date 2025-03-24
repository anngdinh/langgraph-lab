from dotenv import load_dotenv, find_dotenv
import os
_ = load_dotenv(override=True)  # read local .env file

################################################################
# ResearchTeam tools

from typing import Annotated, List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

tavily_tool = TavilySearchResults(max_results=5)

from langchain_community.tools import ShellTool
shell_tool = ShellTool()
    
################################################################
# Document writing team tools

from typing_extensions import TypedDict


################################################################
# Helper Utilities

from typing import List, Optional, Literal
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, trim_messages


class State(MessagesState):
    next: str


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node

################################################################
# Research Team

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o")

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0
)

search_agent = create_react_agent(llm, tools=[tavily_tool])
def search_node(state: State) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

terminal_agent = create_react_agent(llm, tools=[shell_tool])
def terminal_node(state: State) -> Command[Literal["supervisor"]]:
    result = terminal_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="terminal")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

research_supervisor_node = make_supervisor_node(llm, ["search", "terminal"])

################################################################
# print out the graph

research_builder = StateGraph(State)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("search", search_node)
research_builder.add_node("terminal", terminal_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()

from IPython.display import Image, display

display(Image(research_graph.get_graph().draw_png()))

################################################################
# Testing the graph

for s in research_graph.stream(
    {"messages": [("user", "when is Taylor Swift's next tour?")]},
    # {"messages": [("user", "what is current EPL table?")]},
    {"recursion_limit": 100},
):
    print(s)
    print("---")
