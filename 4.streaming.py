import asyncio
from datetime import datetime
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from IPython.display import Image
from dotenv import load_dotenv, find_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
import os
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0
)
_ = load_dotenv(override=True)  # read local .env file

tool = TavilySearchResults(
    max_results=5, include_answer=True, include_raw_content=True, include_images=True,
    api_key=os.getenv('TAVILY_API_KEY')
)


# memory = SqliteSaver.from_conn_string(":memory:")
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"{datetime.now()} - Calling tool {t}")
            result = self.tools[t["name"]].invoke(t["args"])
            # print(f"Result from tool: {result}")
            # for item in result:
            #     print(item['url'])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print(f"{datetime.now()} - back to model")
        return {"messages": results}

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0


# abot = Agent(model, [tool], system=prompt)
# Image(abot.graph.get_graph().draw_png())

# messages = [HumanMessage(content="who was the the before the last soccer world cup winner and what is the GDP of that country in 2024?")]
# result = abot.graph.invoke({'messages': messages})
# print(result['messages'][-1].content)

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, \
you are allowed to do that!
"""
# model = ChatOpenAI(model="gpt-4-turbo")

# with SqliteSaver.from_conn_string(":memory:") as memory:
#     abot = Agent(model, [tool], system=prompt, checkpointer=memory)
#     messages = HumanMessage(
#         content="who won the soccer world cup 4 years before the last world cup and what was that country population 4 year ago?"
#     )
#     thread = {"configurable": {"thread_id": "1234"}}
#     for event in abot.graph.stream({"messages": [messages]}, thread):
#         for v in event.values():
#             print(v["messages"])

    # messages = HumanMessage(content="what is that country's population?")
    # thread = {"configurable": {"thread_id": "1234"}}
    # for event in abot.graph.stream({'messages': [messages]}, thread):
    #     for v in event.values():
    #         print(v['messages'])


async def main():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(model, [tool], system=prompt, checkpointer=memory)
        messages = HumanMessage(
            content="who won the second last soccer world cup and what is their population?"
        )
        thread = {"configurable": {"thread_id": "1234"}}
        async for event in abot.graph.astream({"messages": [messages]}, thread):
            if "llm" in event:
                messages = event["llm"]["messages"]
                for message in messages:
                    content = message.content
                    if content:
                        if hasattr(message, "response_metadata"):
                            token_usage = message.response_metadata.get(
                                "token_usage", {}
                            )
                            content += f" | Token usage: {token_usage}"
                        print(content, end="|")


asyncio.run(main())