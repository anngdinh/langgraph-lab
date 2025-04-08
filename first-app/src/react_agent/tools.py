"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from colorama import Fore, Back, Style
from langchain_community.tools import ShellTool
from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated
from langchain_core.tools import tool
import os
from colorama import Fore, Back, Style

from react_agent.configuration import Configuration


async def search_tavily(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


async def search_duckduckgo(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Duckduckgo search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    print(Fore.GREEN + "[tool] search_duckduckgo()" + Style.RESET_ALL)
    configuration = Configuration.from_runnable_config(config)
    wrapped = DuckDuckGoSearchRun(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


async def search_llm_async(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the LLM search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    # call an api which return in stream
    import httpx
    aggregated_response = []
    timeout: float = 30.0
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream(
                "POST",
                os.getenv("CUONGDM3_ENDPOINT"),
                json={"query": query, "stream": True, "conversation_id": 1},
                headers={"Accept": "text/event-stream",
                         "Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    aggregated_response.append(chunk)
        except httpx.TimeoutException:
            raise TimeoutError(f"Stream exceeded {timeout} seconds")

    return "".join(aggregated_response)


def search_llm_sync(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the LLM search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    import httpx
    aggregated_response = []
    timeout: float = 30.0

    with httpx.Client(timeout=timeout) as client:
        try:
            with client.stream(
                "POST",
                os.getenv("CUONGDM3_ENDPOINT"),
                json={"query": query, "stream": True,
                      "conversation_id": "aaa"},
                headers={"Accept": "text/event-stream",
                         "Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_text():
                    aggregated_response.append(chunk)
        except httpx.TimeoutException:
            raise TimeoutError(f"Stream exceeded {timeout} seconds")

    return "".join(aggregated_response)

# shell_tool = ShellTool()


def shell_tool(command: str) -> str:
    """Run the command on shell. You MUST provide the command."""
    print(Fore.GREEN + "[tool] shell_tool_run(): " + command + Style.RESET_ALL)
    wrapped = ShellTool()
    result = wrapped.invoke({"commands": command})
    print(Fore.GREEN + "[tool] shell_tool_run(): " +
          str(result) + Style.RESET_ALL)
    return result


def duckduckgo_tool_func(query: str) -> str:
    """Search for the information. You MUST provide the query."""
    print(Fore.GREEN + "[tool] duckduckgo_tool_func(): " +
          query + Style.RESET_ALL)
    wrapped = TavilySearchResults()
    result = wrapped.invoke({"query": query})
    # print(Fore.GREEN + "[tool] duckduckgo_tool_func(): " + str(result) + Style.RESET_ALL)
    return result
################################################################


@tool
def generate_sql(question: str) -> str:
    """Generate SQL query from question. It will return struct like this:`{"id": "", "text": "", "type": ""}`"""
    print("Generating SQL query from question: ", question)
    import requests
    url = 'https://hcm-03.console.vngcloud.tech/genai-platform/text-to-sql/api/v0/generate_sql'
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'authorization': f'Basic {os.getenv("VINHPH2_TOKEN")}',
        'priority': 'u=1, i',
        'referer': 'https://hcm-03.console.vngcloud.tech/genai-platform/text-to-sql',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
    }
    params = {
        'question': question
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


@tool
def run_sql_by_id(id: str) -> str:
    """Run SQL query from id."""
    print("Running SQL query from id: ", id)
    import requests
    url = 'https://hcm-03.console.vngcloud.tech/genai-platform/text-to-sql/api/v0/run_sql'
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'authorization': f'Basic {os.getenv("VINHPH2_TOKEN")}',
        'priority': 'u=1, i',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
    }
    params = {
        'id': id
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()
################################################################


SEARCH_TOOLS: List[Callable[..., Any]] = [
    # search_llm_sync,
    duckduckgo_tool_func,
    # search_duckduckgo,
    generate_sql,
    run_sql_by_id,
]

EXECUTE_TOOLS: List[Callable[..., Any]] = [
    shell_tool,
]
