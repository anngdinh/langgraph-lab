from langchain_core.prompts import ChatPromptTemplate
import asyncio
from IPython.display import Image, display
from langgraph.graph import StateGraph, START
from langgraph.graph import END
from typing import Literal
from typing import Union
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Annotated, List, Tuple
import operator
from langgraph.prebuilt import create_react_agent
import llms
import tools
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from dotenv import load_dotenv, find_dotenv
import os
_ = load_dotenv(override=True)  # read local .env file

################################################################
################################################################
################################################################

tools = [
    tools.tavily_tool,
    # tools.duckduckgo_tool,
    tools.shell_tool,
    tools.search_weather,
    tools.find_city_abbr,
    tools.find_city_gdp,
    tools.generate_access_token,
    tools.get_balance,
    tools.generate_sql,
    tools.run_sql_by_id,
    # tools.document_search,
]


agent_executor = create_react_agent(llms.New_Gemini_Pro(), tools,
                                    prompt="You are a smart research assistant. Use the tools to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, \
you are allowed to do that!")

################################################################
# Define the State


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


################################################################
# Planning Step


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | llms.New_Gemini_Pro().with_structured_output(Plan)

# mess = planner.invoke(
#     {
#         "messages": [
#             ("user", "what is the hometown of the current Australia open winner?")
#         ]
#     }
# )
# print(mess)

################################################################
# Re-Plan Step


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, first check if the goal was achieved in the previous steps, return the empty plan. \
If the goal was not achieved, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps. \
Do not use alternative pronouns like 'ones', 'them', 'they', 'it' etc. You must give clear information for each step.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

With the information gathered in the previous steps, update the planning information for the next steps to be more consistent and clear, avoiding dependencies on previous steps. \
You can add notes to the next steps to indicate that the goal was achieved in the previous steps. \
If no further steps are needed and you can return to the user, respond with that information. \
Otherwise, fill in the plan. Only add steps to the plan that still NEED to be done. \
Do not return previously performed steps as part of the plan."""
)


replanner = replanner_prompt | llms.New_Gemini_Pro().with_structured_output(Act)

################################################################
# Create the Graph


async def execute_step(state: PlanExecute):
    plan = state["plan"]
    if len(plan) == 0:
        return {"response": state["past_steps"][-1][-1]}
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    print("Replanning..........")
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

# Add a replan node
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")

# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["agent", END],
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
# app = workflow.compile()

################################################################

# display(Image(app.get_graph(xray=True).draw_mermaid_png()))

################################################################


config = {"recursion_limit": 50, "configurable": {"thread_id": "1234"}}
inputs = {"input":
          #   "what is the weather in san francisco?"
          #   "Compare the weather of San Francisco and London. Write a polite answer."
          # "what is the average GDP of San Francisco and London? Write a polite answer."
          # "The gap temperatures between San Francisco and London. Write a polite answer."
          #   "what is the hometown of the mens 2021 Australia open winner's parents?"
          #   "get a new access token. What tool do you use to get it? Show me the code of the tool."
          #   "get current balance"
          #   "How many customers are using vserver?"
          #   "percentage of customers using vServer?"
          #   "who are using vServer? Use sql tool to find the answer."
          #   "Khách hàng nào xóa vServer nhiều hơn tạo?"
          # "Find all customer deleted more vServer than created? Use sql tool to find the answer."
          #   "Which customer have the most vServer? Give me detail about their usage. Use sql tool to find the answer."
          #   "Information about customer have user_id=53539 usage vServer. Use sql tool to find the answer."
          # "Customer have the most vServer with status CREATED. Use sql tool to find the answer."
          #   "How to date crush. Use document search tool to find the answer."
          #   "ask the document search tool to date crush. Can you use the document search tool to find the answer? Can you read the response from this tool?"
          #   "Find the parents of the man who scored in Arsenal's last match"
          "Find the last match of Arsenal, find who scored in that match, and find the parents of that person."
          # "What is LangGraph"
          }


async def main():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        app = workflow.compile(checkpointer=memory)
        async for event in app.astream(inputs, config=config):
            for k, v in event.items():
                # if k != "__end__":
                print(v)

# Run the async function using an event loop
asyncio.run(main())

################################################################
################################################################
################################################################
################################################################
################################################################
