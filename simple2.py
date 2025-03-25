from dotenv import load_dotenv, find_dotenv
import os
_ = load_dotenv(override=True)  # read local .env file


from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0
)

# This code depends on pip install langchain[anthropic]
from langgraph.prebuilt import create_react_agent


from typing import Annotated
from langchain_core.tools import tool
# note that this can also be changed to return a list instead (for example, you can return a list of values raised to the power p in the example below)

@tool
def sum_exponents(
    p: Annotated[int, "Exponent"],
    values: Annotated[dict[str, int], "Mapping from letters to numbers"]
) -> float:
    """Use this to sum exponents of values, keeping only values for vowels."""
    return sum(pow(v, p) for k, v in values.items() if k in {'a', 'e', 'i', 'o', 'u'})

tools = [sum_exponents]
agent = create_react_agent(llm, tools)

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
mess = agent.invoke({"messages": [("human", "find sum of exponents for these values {'a': 2', 'b': 3, 'e': 4}, use power of 2")]})
# print(mess["messages"][-1].content)
for m in mess["messages"]:
    if isinstance(m, AIMessage):
        print(type(m), ":", m.content, ", ", m.additional_kwargs)
    else:
        print(type(m), ":", m.content)
