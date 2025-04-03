"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import SEARCH_TOOLS, EXECUTE_TOOLS
from react_agent.utils import load_chat_model

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langgraph.prebuilt import create_react_agent


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next and reason. Each worker will perform a"
        " task and respond with their results and status. You must check if the result satisfies the question or not. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]
        reason: str

    def supervisor_node(state: State, config: RunnableConfig) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        configuration = Configuration.from_runnable_config(config)
        # messages = [
        #     {"role": "system", "content": system_prompt},
        # ] + state.messages
        # print(f" *********** Supervisor messages: {messages}")
        messages = [
            {"role": "system", "content": system_prompt}, *state.messages]
        # print(f" *********** Supervisor messages: {messages}")
        response = llm.with_structured_output(Router).invoke(messages, config)
        print(f" *********** Supervisor response: {response}")
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0
)

supervisor_node = make_supervisor_node(llm, ["search_node"])


def search_node(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = "You are a smart research assistant. Use the tools to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, \
you are allowed to do that!"

    model = llm.bind_tools(SEARCH_TOOLS)

    # Get the model's response
    response = cast(
        AIMessage,
        model.invoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node("supervisor_node", supervisor_node)
builder.add_node(search_node)
builder.add_node("search_tools", ToolNode(SEARCH_TOOLS))

builder.add_edge("__start__", "supervisor_node")


def route_model_output(state: State) -> Literal["supervisor_node", "search_tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("supervisor_node" or "search_tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "supervisor_node"
    # Otherwise we execute the requested actions
    return "search_tools"


# Add a conditional edge to determine the next step after `search_node`
builder.add_conditional_edges(
    "search_node",
    route_model_output,
)

builder.add_edge("search_tools", "search_node")

# Compile the builder into an executable graph
# You can customize this by adding interrupt points for state updates
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "ReAct Agent"  # This customizes the name in LangSmith
