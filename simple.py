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

def search(query: str):
    """Call to surf the web."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

agent = create_react_agent(llm, tools=[search])
mess = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf ? write a romantic answer."}]}
)
print(mess["messages"][-1].content)