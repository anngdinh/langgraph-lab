from dotenv import load_dotenv, find_dotenv
import os
_ = load_dotenv(override=True)  # read local .env file

################################################################
# Tools

from langchain_core.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults
tavily_tool = TavilySearchResults(max_results=5)

from langchain_community.tools import ShellTool
shell_tool = ShellTool()


from typing import Annotated
from langchain_experimental.utilities import PythonREPL
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    
    print(f"Executing code: {code}")
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str

################################################################
# LLM

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0,
)

# from langchain_ollama import ChatOllama
# llm = ChatOllama(
#     model="llama3.2",
#     # api_key=os.getenv('OLLAMA_API_KEY'),
#     temperature=0,
# )

# from langchain_deepseek import ChatDeepSeek
# llm = ChatDeepSeek(
#     model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
#     api_key=os.getenv('DEEPSEEK_API_KEY'),
#     api_base="https://model-000.ai-gateway.vngcloud.tech/deepseek/chat/completions",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )
    
################################################################
# Supervisor

from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import MessagesState, END
from langgraph.types import Command

members = ["flight_assistant", "driving_assistant"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH. NEVER attempt to answer questions outside of the scope of the workers, just return `I don't know`."
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]
    
class State(MessagesState):
    next: str


def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages, {"recursion_limit": 5})
    print("-----------------", response)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})

################################################################
# Graph

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

prompt_template = "You are a smart assistant. ONLY answer questions if {topic}, otherwise return 'that question is out of my scope.'."

flight_agent = create_react_agent(
    llm, tools=[tavily_tool], prompt=prompt_template.format(topic="input is about flying, flights, airlines, or travel by air")
)


def flight_node(state: State) -> Command[Literal["supervisor"]]:
    result = flight_agent.invoke(state)
    # print("aaaaaaa", result)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="flight_assistant")
            ]
        },
        goto="supervisor",
    )


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
driving_agent = create_react_agent(
    llm, tools=[tavily_tool], prompt=prompt_template.format(topic="input is about driving")
)


def driving_node(state: State) -> Command[Literal["supervisor"]]:
    result = driving_agent.invoke(state)
    # print("aaaaaaa", result)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="driving_assistant")
            ]
        },
        goto="supervisor",
    )


builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("flight_assistant", flight_node)
builder.add_node("driving_assistant", driving_node)
graph = builder.compile()

from IPython.display import display, Image
display(Image(graph.get_graph().draw_png()))


################################################################

for s in graph.stream(
    {"messages": [("user",
                # "How many flights are there from Hanoi to Ho Chi Minh City?",
                # "How many highways are there from Hanoi to Ho Chi Minh City?",
                # "compare time to drive from Hanoi to Ho Chi Minh City and Da Nang to Ho Chi Minh City",
                "how far between Hanoi and Ho Chi Minh City when flying? How about driving?",
                # "how is GDP of Vietnam compared to Thailand?",
                # "how long does it take to take a trip around the world in a boat?",
            )]}, subgraphs=True
):
    print("bbbbbbb", s)
    print("----")