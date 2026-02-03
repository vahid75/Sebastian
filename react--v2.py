# pip install -U langchain-openai langchain-core
# export OPENAI_API_KEY="..."

import re
from typing import Callable, Dict, Tuple, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import dotenv


dotenv.load_dotenv('.env')

# -------------------------
# 1) Tools (your executor)
# -------------------------

def calculate(expression: str) -> str:
    """Basic calculator tool (demo)."""
    if not re.fullmatch(r"[0-9+\-*/(). \t]+", expression):
        return "ERROR: unsupported characters in expression"
    try:
        return str(eval(expression, {"builtins": {}}))
    except Exception as e:
        return f"ERROR: {e}"

_FAQ = {
    "planner vs executor": "Planner decides next action; executor runs tools and returns observations.",
    "react": "ReAct = Reason + Act loop: decide an action, run it, observe, repeat.",
}

def lookup(topic: str) -> str:
    """Tiny lookup tool (demo)."""
    return _FAQ.get(topic.strip().lower(), f"No entry for '{topic}'")

TOOLS: Dict[str, Callable[[str], str]] = {
    "calculate": calculate,
    "lookup": lookup,
}


# ---------------------------------
# 2) ReAct prompt + output parser
# ---------------------------------

SYSTEM = SystemMessage(
    content=(
        "You are a ReAct agent.\n"
        "You must follow this format EXACTLY and only:\n\n"
        "Thought: <brief>\n"
        "Action: <one of: calculate, lookup, finish>\n"
        "Action Input: <string>\n\n"
        "If Action is finish, put the final user-facing answer in Action Input.\n"
        "Never invent tool outputs. Only use observations given to you."
    )
)

ACTION_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\nAction:\s*(?P<action>.*?)\nAction Input:\s*(?P<input>.*)",
    re.DOTALL,
)

def parse_react(text: str) -> Tuple[str, str, str]:
    m = ACTION_RE.search(text.strip())
    if not m:
        # Fail loudly so you can see when the model deviates from the format
        raise ValueError(f"Model output not in ReAct format:\n{text}")
    return m.group("thought").strip(), m.group("action").strip(), m.group("input").strip()


# -----------------------------
# 3) The explicit agent loop
# -----------------------------

def run_react_agent(user_question: str, max_iters: int = 8) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # This "scratchpad" is what LangChain often builds internally.
    scratchpad = ""

    for step_idx in range(1, max_iters + 1):
        prompt = (
            f"User question: {user_question}\n\n"
            f"Previous steps:\n{scratchpad}\n"
            "Now decide the next step."
        )

        resp = llm.invoke([SYSTEM, HumanMessage(content=prompt)])
        text = resp.content

        thought, action, action_input = parse_react(text)

        print(f"\n--- ITERATION {step_idx} ---")
        print("MODEL THOUGHT:", thought)
        print("MODEL ACTION :", action)
        print("ACTION INPUT :", action_input)

        if action == "finish":
            print("\n[done]")
            return action_input  # final answer

        tool = TOOLS.get(action)
        if tool is None:
            observation = f"ERROR: unknown tool '{action}'. Available: {list(TOOLS)}"
        else:
            observation = tool(action_input)

        print("OBSERVATION :", observation)

        # Append to scratchpad so the model can use it next iteration
        scratchpad += (
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Action Input: {action_input}\n"
            f"Observation: {observation}\n\n"
        )

    return "Stopped: reached max_iters without finishing."


# if "__name__" == "__main__":
question = (
    "Explain planner vs executor in one sentence using lookup, "
    "then calculate (12.5 * 4) + 3, then give me both."
)
final = run_react_agent(question)
print("\nFINAL ANSWER:\n", final)