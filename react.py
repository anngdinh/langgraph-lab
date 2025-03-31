from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(override=True)  # read local .env file

################################################################
import tools

################################################################
import llms

################################################################

tools = [
    # tools.tavily_tool,
    tools.duckduckgo_tool,
    tools.shell_tool,
    tools.search_weather,
    tools.find_city_abbr,
    tools.find_city_gdp,
    tools.generate_access_token,
    tools.get_balance,
]

# Define the graph

from langgraph.prebuilt import create_react_agent

graph = create_react_agent(
    llms.New_Gemini_Pro(),
    tools=tools,
    prompt="You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, \
you are allowed to do that!",
)

################################################################
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
################################################################
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {
    "messages": [
        (
            "user",
            # "what is the weather in san francisco?"
            # "what is the hometown of the mens 2024 Australia open winner's parents?"
            # "what is the average GDP of San Francisco and London? Write a polite answer."
            # "Compare the weather of San Francisco and London. Write a polite answer."
            "The gap temperatures between San Francisco and London. Write a polite answer."
            # "get current balance"
            # "get a new access token"
        )
    ]
}
print_stream(graph.stream(inputs, stream_mode="values"))
