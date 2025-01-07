INTENT_FIX_SHOT_USER_PROMPT = """你是一个意图分类。你的任务是根据用户的请求，精准判断用户的意图。

用户的可能意图如下：
{intent_list}

请从以上意图中选出一个合适的意图。如果以上意图与用户请求不相符、或用户意图为闲聊时，请输出“闲聊”。

如果用户在请求中，指定了应用（APP）名称，请输出应用名称；否则输出空字符串。

请使用如下格式进行输出：
intent: "用户意图", app: "应用名称"

###
例如：

用户请求：使用微信给孙总发条消息，今天晚上6点开会，讨论下阶段广告投放策略。
输出：
intent: "即时通讯", app: "微信"

用户请求：查一下今日要闻。
输出：
intent: "新闻", app: ""

###
现在，请严格按照输出格式回答用户的如下请求，完成请求之后立刻停止输出：

用户请求：{query}
输出：
"""





INTENT_FIX_SHOT_USER_PROMPT_EN = """
You are an intent classifier. Your task is to accurately determine the user's intent based on their request.

The possible user intents are as follows:
{intent_list}

Please select a suitable intent from the above list. If none of the intents match the user's request, or if the user's intent is casual conversation, please output "casual conversation".

If the user specifies an application name in their request, please output the application name, otherwise, output 'null'.

For the output, your generation should have exactly the following format and have exactly 4 lines (without ```):
```
intent:
user intent,
app:
application name
```

###
For example:

For user request: Use WeChat to send a message to Mr. Sun, the meeting is at 6 PM tonight to discuss the next phase of the advertising strategy.
Your response can be like this:
```
intent:
instant messaging
app:
WeChat
```

For user request: Check today's news.
Your response can be like this:
```
intent:
news
app:
null
```

###
Now, please strictly follow the output format to respond to the user's request below:

User request: {query}
"""