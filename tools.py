from dotenv import load_dotenv
import os
_ = load_dotenv(override=True)  # read local .env file

################################################################
# ResearchTeam tools

from typing import Annotated, List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

################################################################
from colorama import Fore, Back, Style

@tool
def generate_sql(question: str) -> str:
    """Generate SQL query to answer question. You ONLY provide normal question (human language only). It will return struct like this:`{"id": "", "text": "", "type": ""}`"""
    print(Fore.GREEN + "[tool] generate_sql(): " + question + Style.RESET_ALL)
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

################################################################
@tool
def run_sql_by_id(id: str) -> str:
    """Run SQL query from id."""
    print(Fore.GREEN + "[tool] run_sql_by_id(): " + id + Style.RESET_ALL)
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
from langchain_community.tools.tavily_search import TavilySearchResults
# tavily_tool = TavilySearchResults(max_results=5)
def tavily_tool(query: str)-> str:
    """Search for the information. You MUST provide the query."""
    print(Fore.GREEN + "[tool] tavily_tool(): " + query + Style.RESET_ALL)
    wrapped = TavilySearchResults()
    result = wrapped.invoke({"query": query})
    print(Fore.GREEN + "[tool] tavily_tool(): " + str(result) + Style.RESET_ALL)
    return result
    # return TavilySearchResults(**kwargs)

from langchain_community.tools import ShellTool
# shell_tool = ShellTool()
def shell_tool(command: str)-> str:
    """Run the command on shell. You MUST provide the command."""
    print(Fore.GREEN + "[tool] shell_tool_run(): " + command + Style.RESET_ALL)
    wrapped = ShellTool()
    result = wrapped.invoke({"commands": command})
    print(Fore.GREEN + "[tool] shell_tool_run(): " + str(result) + Style.RESET_ALL)
    return result

from langchain_community.tools import DuckDuckGoSearchRun
def duckduckgo_tool(query: str)-> str:
    """Search for the information from internet ONLY. You MUST provide the query."""
    print(Fore.GREEN + "[tool] duckduckgo_tool(): " + query + Style.RESET_ALL)
    wrapped = DuckDuckGoSearchRun()
    result = wrapped.invoke({"query": query})
    print(Fore.GREEN + "[tool] duckduckgo_tool(): " + str(result) + Style.RESET_ALL)
    return result
################################################################

def find_city_gdp(city: str):
    """Find the city GDP. You MUST provide the city abbreviation."""
    print(Fore.GREEN + "[tool] find_city_gdp(): " + city + Style.RESET_ALL)
    if "sf" in city.lower():
        return "$500 billion."
    if "ldcc" in city.lower():
        return "$700 billion."
    return "Please provide the correct city abbreviation and try again."

def find_city_abbr(city: str):
    """Find the city abbreviation by city name."""
    print(Fore.GREEN + "[tool] find_city_abbr(): " + city + Style.RESET_ALL)
    if "san francisco" in city.lower():
        return "sf"
    if "london" in city.lower():
        return "ldcc"
    return "The city abbreviation could not be found. Please provide the correct city name and try again."

def search_weather(query: str):
    """Search for the weather of city. You MUST provide the city abbreviation."""
    print(Fore.GREEN + "[tool] search_weather(): " + query + Style.RESET_ALL)
    if "sf" in query.lower() :
    # or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    if "ldcc" in query.lower() :
    # or "london" in query.lower():
        return "It's 50 degrees and rainy."
    # return "It's 90 degrees and sunny."
    return "The query must include the abbreviation of the city name. Please provide the correct format and try again."

################################################################
@tool
def generate_access_token() -> str:
    """
    Generate a new access token. This function handles all necessary steps 
    (including refresh tokens if needed) internally. When the token expires, 
    call this function directly instead of managing refresh tokens manually.
    """
    print(Fore.GREEN + "[tool] generate_access_token()" + Style.RESET_ALL)
    import requests
    url = 'https://iamapis.vngcloud.vn/accounts-api/v1/auth/token'
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'authorization': f'Basic {os.getenv("IAM_TOKEN")}',
        'content-type': 'application/json',
    }
    data = {
        'grantType': 'client_credentials'
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    access_token = result['accessToken']
    print(Fore.GREEN + "[tool] generate_access_token(): " + access_token + Style.RESET_ALL)
    return access_token

@tool
def get_balance(token: str) -> str:
    """Fetch the current account balance. **Requires a valid access token.**"""
    print(Fore.GREEN + "[tool] get_balance(): " + Style.RESET_ALL)
    import requests
    url = 'https://hcm-3.console.vngcloud.vn/vserver/iam-billing-gateway/v1/users/info'
    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'en',
        'authorization': f'Bearer {token}',
        'content-type': 'application/json',
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################