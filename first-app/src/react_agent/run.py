from graph import graph


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
            # "last game of Arsenal"
            "Find the most recent Arsenal match, find the players who scored in that match and find the parents of those players."
            # "use terminal to list all pods"
            # "find different between LangGraph and LangChain then save the report as a file using terminal. "
            # "different revert and forward proxy, use tools"
            # "find the latest cve related to k8s, find and run command to check if my cluster is vulnerable (using terminal tools)"
            # "find the latest cve related to k8s, save it to a file using terminal. "
            # "use terminal to list all pod in all ns"
            # "what is vks?"
            # "use search tool \"give me full yaml to create a NLB in my VKS kubernetes cluster\""
        )
    ]
}
print_stream(graph.stream(inputs, stream_mode="values"))
