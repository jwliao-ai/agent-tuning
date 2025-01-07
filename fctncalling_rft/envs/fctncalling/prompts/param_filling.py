RAG_LORA_PARAM_USER_PROMPT = """你是一个参数填充器，你的任务是按照给定的格式合理的进行参数填充。

给到工具名ToolName，以及相关功能描述和包含key、value对的参数格式
按以下格式输出：
```
ToolName:调用的工具名
Parameters:相应的参数调用对应的JSON格式
```
例如：
{few_shots}

现在开始！
请严格按照输出格式回答下面的用户问题，完成了问题之后立刻停止输出：\n

{tool_list}
Question: {task}
Answer:"""





FILL_PARAM_EN = """You are a parameter filler, and your task is to reasonably fill in parameters according to the given format.

Given the tool name, available parameters in key-value pair format and the query, you should be able to generate the parameter completion for the tool from the query. If the tool should be called more than one time, please generate a list of reasonable parameter completions.

For example, when given:
```
ToolName: the name of the tool.
Available Parameters: the parameters of the tool.
Query: the query to be completed.
Parameter completion:
```

you can output like below (without ```):
```
a list of corresponding parameter completions in JSON format and for each item in each param completion, the key is the parameter name and the value is the parameter value.
```

Let's begin!
Please strictly follow the output format to generate the parameter completions below without explanation and stop generating right after your parameter completions.

ToolName: {tool_name}.
Parameters: {parameters}.
Query: {query}.
Parameter completion:
"""



FILL_MORE_PARAM_EN = """You are a parameter filler, and your task is to reasonably fill in parameters according to the given format.

Given the tool name, available parameters in key-value pair format and the query, you should be able to generate the parameter completion for the tool from the query. If the tool should be called more than one time, please generate a list of reasonable parameter completions.

In the last round, for the query ```{query}``` and selected tool ```{tool_name}```, you generated the following parameter completions of the tool to solve the query:
{last_param_completions}.

But to solve the query, the tool should be called more times with more parameter completions. So please generate a list of more reasonable parameter completions for the tool to solve the query.

For the output format, for example, when given:
```
ToolName: the name of the tool.
Available Parameters: the parameters of the tool.
Query: the query to be completed.
Parameter completion:
```

you can output like below (without ```):
```
a list of corresponding parameter completions in JSON format and for each item in each param completion, the key is the parameter name and the value is the parameter value.
```

Let's begin!
Please strictly follow the output format to generate the parameter completions below and no need to explain why.

ToolName: {tool_name}.
Parameters: {parameters}.
Query: {query}.
Parameter completion:
"""



FILL_PARAM_WRONG_EN = """You are a parameter filler, and your task is to generate reasonable parameter completions according to the given format.

Given the tool name, available parameters in key-value pair format and the query, you should be able to generate the parameter completion for the tool from the query. If the tool should be called more than one time, please generate a list of reasonable parameter completions.

In the last round, for the query ```{query}``` and selected tool ```{tool_name}```, you generated the following quoted parameter completions of the tool to solve the query:
```{last_param_completions}```.

But to solve the query, your parameter completions may be incorrect. The reason may either you generated too many completions or some of the completions are not suitable. So please re-generate a list of more reasonable parameter completions for the tool to solve the query.

For the output format, for example, when given:
```
ToolName: the name of the tool.
Available Parameters: the parameters of the tool.
Query: the query to be completed.
Parameter completion:
```

you can output like below (without ```):
```
a list of corresponding parameter completions in JSON format and for each item in each param completion, the key is the parameter name and the value is the parameter value.
```

Let's begin!
Please strictly follow the output format to generate the parameter completions below and no need to explain why.

ToolName: {tool_name}.
Parameters: {parameters}.
Query: {query}.
Parameter completion:
"""



FILL_PARAM_FORMAT_WRONG_EN = """You are a parameter filler, and your task is to generate reasonable parameter completions according to the given format.

Given the tool name, available parameters in key-value pair format and the query, you should be able to generate the parameter completion for the tool from the query. If the tool should be called more than one time, please generate a list of reasonable parameter completions.

For the output format, for example, when given:
```
ToolName: the name of the tool.
Parameters: the parameters of the tool.
Query: the query to be completed.
Parameter completion:
```

you can output like below (without ```):
```
a list of corresponding parameter completions in JSON format and for each item in each param completion, the key is the parameter name and the value is the parameter value.
```

In the last round, for the query ```{query}``` and selected tool ```{tool_name}```, you generated the following quoted parameter completions of the tool to solve the query:
```{last_param_completions}```.

But the format is totally incorrect. Your response should be a complete JSON format response!

Please strictly follow the output format to generate the parameter completions below and no need to explain why.

ToolName: {tool_name}.
Parameters: {parameters}.
Query: {query}.
Parameter completion:
"""