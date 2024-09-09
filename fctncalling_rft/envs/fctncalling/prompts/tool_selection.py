
TOOL_LIST_SCHEMA = '''{name}: {description}\n'''
ONE_SHOT_NAME = '''\nQuestion: {query}\nAnswer: {ToolName}\n'''
TOOL_LIST_SCHEMA_PARAM = '''Tool name: {interface}\nParameter format: {parameters}'''
ONE_SHOT_PARAM = '''\nTool name: {ToolName}\nParameter format: {params}\nQuestion: {query}\nAnswer: ```\nToolName: {ToolName}\nParameters: {label}\n```\n'''

RAG_LORA_NAME_USER_PROMPT = """你是一个工具选择器，你的任务是根据用户的请求，精准调用相应的工具。

你能调用的工具如下：
{tool_list}

请从以上工具中选出一个合适的工具。当以上工具无法完成用户的请求时，输出<irrelevant_function>。

输出格式：
仅输出需要调用的工具 Tool

例如：
{few_shots}

现在开始！
请严格按照输出格式回答下面的用户问题，完成了问题之后立刻停止输出：
Question:{task}
Answer:"""






SELECT_TOOL_EN = """You are a tool selector, and your task is to accurately call the appropriate tool based on the user's request.

The available tools you can call are as follows:
{tool_list}

Please select suitable tools from the above tools. If none of the above tools can fulfill the user's query, output <irrelevant_function>.

For example, given:
```
Query: the user's query.
Tools to be called:
```
Your response can be like this (without ```):
```
toolname1, toolname2, toolname3
```

Remember, do not directly answer the query and no need for explanation, only output suitable tools to be called! And stop generating right after finishing outputing the suitable tools.

Let's begin!

Query: {query}
Tools to be called: """



SELECT_MORE_TOOL_EN = """You are a tool selector, and your task is to accurately call the appropriate tool based on the user's request.

In the last round, for the query ```{query}```, you selected the following tools to solve the query:
{selected_tools}.

But these tools are not enough to fulfill the user's query. Please select more suitable tools from the below tools:
{tool_list}

For the output format, for example, given:
```
Query: the user's query.
Tools to be called:
```
Your response can be like this (without ```):
```
toolname1, toolname2, toolname3
```

Remember, do not directly answer the query and no need for explanation, only output the tool to be called! And stop generating right after finishing outputing the suitable tools.

Let's begin!

Query: {query}
Tools to be called:"""



SELECT_TOOL_WRONG_EN = """You are a tool selector, and your task is to accurately call the appropriate tool based on the user's request.

In the last round, for the query ```{query}```, you selected the following tools to solve the query:
{selected_tools}.

But some of the selected tools are not right for solving the query:
{wrong_tools}.

So, you need to re-select suitable tools for the query from the tool list below:
{tool_list}

For the output format, for example, given:
```
Query: the user's query.
Tools to be called:
```
Your response can be like this (without ```):
```
toolname1, toolname2, toolname3
```

Remember, do not directly answer the query and no need for explanation, only output the tool to be called! And stop generating right after finishing outputing the suitable tools.

Let's begin!

Query: {query}
Tools to be called:"""



SELECT_TOOL_HALLUCINATED_EN = """You are a tool selector, and your task is to accurately call the appropriate tool based on the user's request.

In the last round, for the query ```{query}```, you selected the following tools to solve the query:
{selected_tools}.

But some of the selected tools are not from the tool list:
{hallucinated_tools}.

The reason these tools are considered a hallucination is either it self is not in the tool list or you gave the tool name without following the established format.

So, you need to re-select suitable tools for the query only from the tool list below:
{tool_list}

For the output format, for example, given:
```
Query: the user's query.
Tools to be called:
```
Your response can be like this (without ```):
```
toolname1, toolname2, toolname3
```

Remember, do not directly answer the query and no need for explanation, only output the tool to be called! And stop generating right after finishing outputing the suitable tools.

Let's begin!

Query: {query}
Tools to be called:"""



SELECT_TOOL_WRONG_HALLUCINATED_EN = """You are a tool selector, and your task is to accurately call the appropriate tool based on the user's request.

In the last round, for the query ```{query}```, you selected the following tools to solve the query:
{selected_tools}

But some of the selected tools are not right for solving the query and some are not in the tool list:
Wrong tools: {wrong_tools}.
Halucinated tools: {hallucinated_tools}.

The reason these tools are considered a hallucination is either it self is not in the tool list or you gave the tool name without following the established format.

So, you need to re-select suitable tools for the query only from the tool list below:
{tool_list}

For the output format, for example, given:
```
Query: the user's query.
Tools to be called:
```
Your response can be like this (without ```):
```
toolname1, toolname2, toolname3
```

Remember, do not directly answer the query and no need for explanation, only output the tool to be called! And stop generating right after finishing outputing the suitable tools.

Let's begin!

Query: {query}
Tools to be called:"""