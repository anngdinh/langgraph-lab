from dotenv import load_dotenv, find_dotenv
import os
_ = load_dotenv(override=True)  # read local .env file


from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0,
    verbose=True
)

# This code depends on pip install langchain[anthropic]
from langgraph.prebuilt import create_react_agent

def find_city_abbr(city: str):
    """Call to find the city abbreviation."""
    print("finding city abbreviation forrrrrrrrrr:", city)
    if "san francisco" in city.lower():
        return "sf"
    if "london" in city.lower():
        return "ldcc"
    return "The city abbreviation could not be found. Please provide the correct city name and try again."

def search(query: str):
    """Call to find the weather."""
    print("searching forrrrrrrrrrrr:", query)
    if "sf" in query.lower() :
    # or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    if "ldcc" in query.lower() :
    # or "london" in query.lower():
        return "It's 50 degrees and rainy."
    # return "It's 90 degrees and sunny."
    return "The query must include the first letters of the city name, for example, to find New York, the owner needs to pass 'weather in NY'. Please provide the correct format and try again."

agent = create_react_agent(
    llm, 
    tools=[search, find_city_abbr],
    prompt="You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, \
you are allowed to do that! You MUST log what you are doing and why.",
    debug=True
)

mess = agent.invoke(
    {"messages": [{"role": "user", "content": 
        # "what is the weather in San Francisco now? return a polite answer."
        "what is the weather in London now? return a polite answer."
        }]}
)

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
for m in mess["messages"]:
    print(m)
    # if isinstance(m, AIMessage):
    #     print(type(m), ":", m.content, ", ", m.tool_calls)
    # else:
    #     print(type(m), ":", m.content)
