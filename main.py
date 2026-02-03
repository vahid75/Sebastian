from openai import OpenAI
import json


client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string"
                    }
                },
                "required": ["city"]
            }
        }
    }
]


SYSTEM_PROMPT = """
You are a helpful assistant.
Use tools when needed to answer the user's question.
"""


def calculator(expression: str) -> str:
    return str(eval(expression))


def get_weather(city: str) -> str:
    data = {
        "helsinki": "Cold and snowy ❄️",
        "tehran": "Sunny ☀️"
    }
    return data.get(city.lower(), "Unknown")




def react_agent(user_input: str, max_iters = 5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for _iter in range(max_iters):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # ✅ Final answer
        if msg.content:
            return msg.content

        # ✅ Tool call (structured, no parsing)
        tool_call = msg.tool_calls[0]
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if name == "calculator":
            result = calculator(**args)
        elif name == "get_weather":
            result = get_weather(**args)
        else:
            result = "Unknown tool"

        messages.append(msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })


res = react_agent("Whats the weather in Tehran")
print(res)