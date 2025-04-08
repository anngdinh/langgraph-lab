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


def make_supervisor_node(llm: BaseChatModel, workers: list[dict[str, str]]) -> str:
    members = [f"{worker['name']}" for worker in workers]
    options = ["FINISH"] + members
    system_prompt = f"""You are an intelligent task coordinator managing a team of specialized AI agents. Your team consists of: {members}

Worker Specializations:
{'\n'.join([f"- {worker['name']}: Goal: {worker['goal']}. Backstory: {worker['backstory']}" for worker in workers])}

Core Responsibilities:
1. Task Analysis & Delegation
   - Analyze user requests to determine required expertise
   - Match tasks to the most suitable worker based on their specialization
   - Ensure efficient task routing and coordination

2. Quality Control
   - Verify that worker responses fully address user needs
   - Request clarification when responses are incomplete
   - Ensure information accuracy and completeness

3. Workflow Management
   - If more information is needed: Ask user specific questions and FINISH
   - If task requires multiple workers: Coordinate sequential actions
   - When task is complete: Respond with FINISH

Decision Making Process:
1. Evaluate if the request needs clarification
   - If unclear: Ask specific questions to get necessary details
   - If ambiguous: Request clarification on specific points
   - If incomplete: Ask for missing information
2. Identify which worker's expertise best matches the task
3. Consider if multiple workers need to collaborate
4. Determine if the current response satisfies the user's needs

Response Format:
When asking for clarification:
- Next Action: FINISH
- Reasoning: [List specific questions that need answers]

When delegating to a worker:
- Next Action: [Worker Name]
- Reasoning: [Clear explanation of your decision]
- Additional Context: [Any relevant information for the chosen worker]

When completing a task:
- Next Action: FINISH
- Reasoning: [Brief summary of what was accomplished]

IMPORTANT: You must ALWAYS provide a response, even when asking for clarification. Never return an empty response.
"""

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
        update_messages = cast(AIMessage, response["reason"])
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto, "messages": [update_messages]})

    return supervisor_node


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0
)

workers = [
    {
        "name": "search_node",
        "goal": "Find information using search tools",
        "backstory": "Expert in searching and retrieving information from various sources, tools, and databases"
    },
    {
        "name": "terminal_node",
        "goal": "Execute commands and retrieve information from the terminal",
        "backstory": "Expert in executing commands and retrieving information from the terminal"
    },
]
supervisor_node = make_supervisor_node(llm, workers)


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
    system_message_format = """You are an expert information retrieval specialist. Your role is to find accurate, relevant, and up-to-date information to answer user queries.

Search Strategy Guidelines:
1. Query Formulation
   - Break down complex questions into specific search terms
   - Use precise, targeted keywords
   - Consider multiple search angles

2. Source Evaluation
   - Prioritize authoritative sources
   - Cross-reference information across multiple sources
   - Consider the recency and relevance of information

3. Information Synthesis
   - Combine information from multiple sources
   - Highlight conflicting information
   - Provide context for the information found

4. Response Format
   - Present information clearly and concisely
   - Include source attribution when possible
   - Indicate confidence level in the information

Remember:
- Make multiple searches if needed to verify information
- Ask for clarification if the search query is ambiguous
- If information seems incomplete, try alternative search terms
- If you're unsure about the search strategy, ask the supervisor

System time: {system_time}"""

    system_message = system_message_format.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

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


def terminal_node(
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
    system_message_format = """You are a system operations specialist with expertise in Kubernetes cluster management and system operations. Your primary role is to interact with the Kubernetes cluster using kubectl commands and execute system operations safely.

Core Responsibilities:
1. Kubernetes Operations
   - Execute kubectl commands to manage cluster resources
   - Monitor cluster health and resource status
   - Troubleshoot cluster issues
   - Deploy and manage applications

2. Command Execution
   - Execute commands with proper error handling
   - Verify command safety before execution
   - Provide clear explanations of command purposes

3. Safety Protocols
   - Never execute potentially harmful commands
   - Always verify kubectl commands before execution
   - Use --dry-run flag when appropriate
   - Report any security concerns to the supervisor

4. Response Format
   - Command executed: [command]
   - Purpose: [explanation]
   - Output: [formatted result]
   - Status: [success/error]
   - Next steps: [recommendations]

Remember:
- Always explain what you're doing and why
- If a command seems unsafe, ask for confirmation
- If you encounter errors, provide clear error messages
- If you're unsure about a command, ask the supervisor
- Format output for readability
- Be cautious with kubectl delete and apply commands

System time: {system_time}"""

    system_message = system_message_format.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    model = llm.bind_tools(EXECUTE_TOOLS)

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
builder.add_node(terminal_node)
builder.add_node("search_tools", ToolNode(SEARCH_TOOLS))
builder.add_node("terminal_tools", ToolNode(EXECUTE_TOOLS))

builder.add_edge("__start__", "supervisor_node")


def route_search_node(state: State) -> Literal["supervisor_node", "search_tools"]:
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


def route_terminal_node(state: State) -> Literal["supervisor_node", "terminal_tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("supervisor_node" or "terminal_tools").
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
    return "terminal_tools"


# Add a conditional edge to determine the next step after `search_node`
builder.add_conditional_edges(
    "search_node",
    route_search_node,
)
builder.add_edge("search_tools", "search_node")

builder.add_conditional_edges(
    "terminal_node",
    route_terminal_node,
)
builder.add_edge("terminal_tools", "terminal_node")


# Compile the builder into an executable graph
# You can customize this by adding interrupt points for state updates
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "ReAct Agent"  # This customizes the name in LangSmith
