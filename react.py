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
    # tools.shell_tool,
    # tools.search_weather,
    # tools.find_city_abbr,
    # tools.find_city_gdp,
    # tools.generate_access_token,
    # tools.get_balance,
    # tools.generate_sql,
    # tools.run_sql_by_id,
    # tools.document_search,
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
    # debug=True,
)

################################################################
from IPython.display import Image, display

display(Image(graph.get_graph().draw_png()))
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
            # "The gap temperatures between San Francisco and London. Write a polite answer."
            # "get current balance"
            # "get a new access token. What tool do you use to get it? Show me the code of the tool."
            # "How many customers are using vserver?"
            # "How many customers are using vks? Use sql tool to find the answer."
            # "Khách hàng nào xóa vServer nhiều hơn tạo?"
            # "Which customer deleted more vServer than created? Use sql tool to find the answer."
            # "Which customer have the most vServer? Give me detail about their usage. Use sql tool to find the answer."
            # "Information about customer have user_id=53539 usage vServer. Use sql tool to find the answer."
            # "Customer have the most vServer with status CREATED. Use sql tool to find the answer."
            # "How to date crush. Use document search tool to find the answer."
            # "ask the document search tool to date crush. Can you use the document search tool to find the answer? Can you read the response from this tool?"
            # "find information about Bukayo Saka's parents"
            # "Find the parents of the man who scored in Arsenal's last match"
            "Find the last match of Arsenal, find who scored in that match, and find the parents of that person."
            # "What is Langgraph?"
        )
    ]
}
print_stream(graph.stream(inputs, stream_mode="values"))
