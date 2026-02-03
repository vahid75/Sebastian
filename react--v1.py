
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import re

# ----------------------------
# "Tools" the executor can use
# ----------------------------

def tool_calculate(expression: str) -> str:
    """Very small calculator tool (safe-ish demo). Supports digits, + - * / ( ) . and spaces."""
    if not re.fullmatch(r"[0-9+\-*/(). \t]+", expression):
        return "ERROR: expression contains unsupported characters"
    try:
        result = eval(expression, {"builtins": {}})
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"

def tool_web_search(query: str) -> str:
    """Fake search tool for demonstration."""
    fake_index = {
        "helsinki population": "Helsinki population is about 650,000 (city proper) in recent years.",
        "finland capital": "The capital of Finland is Helsinki.",
        "react agent": "ReAct agents interleave reasoning and tool actions in a loop: think -> act -> observe."
    }
    return fake_index.get(query.lower(), f"No results for '{query}' (fake search).")


# ----------------------------
# Planner / Executor Contracts
# ----------------------------

@dataclass
class Step:
    kind: str               # "tool" or "finish"
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    message: Optional[str] = None

@dataclass
class AgentState:
    goal: str
    memory: Dict[str, Any] = field(default_factory=dict)
    history: List[Tuple[str, str]] = field(default_factory=list)  # (step description, observation)

class Planner:
    """
    Planner decides what to do next based on the goal and what we already know.
    This is where "strategy" lives.
    """
    def next_step(self, state: AgentState) -> Step:
        goal = state.goal.lower()

        # If goal involves math, plan to use calculator tool.
        if ("calculate" in goal or "what is" in goal) and "calc_result" not in state.memory:
            # Extract a math-ish expression from the goal (toy heuristic)
            expr = re.sub(r"[^0-9+\-*/(). ]", " ", state.goal)
            expr = re.sub(r"\s+", " ", expr).strip()
            return Step(kind="tool", tool_name="calculate", tool_input=expr)

        # If goal involves "look up" / "search", plan to use web_search.
        if ("look up" in goal or "search" in goal) and "search_result" not in state.memory:
            # Toy heuristic: take everything after "search" or "look up"
            m = re.search(r"(?:search|look up)\s+(.*)", state.goal, re.IGNORECASE)
            query = (m.group(1) if m else state.goal).strip()
            return Step(kind="tool", tool_name="web_search", tool_input=query)

        # Otherwise, if we have enough info, finish with a composed response.
        return Step(kind="finish", message=self._compose_answer(state))

    def _compose_answer(self, state: AgentState) -> str:
        parts = []
        if "calc_result" in state.memory:
            parts.append(f"Result: {state.memory['calc_result']}")
        if "search_result" in state.memory:
            parts.append(f"Found: {state.memory['search_result']}")
        if not parts:
            parts.append("I didn't need any tools; here's a direct response (demo).")
        return "\n".join(parts)


class Executor:
    """
    Executor performs the planner's chosen step using tools.
    This is where "doing" lives.
    """
    def run_step(self, step: Step, state: AgentState) -> str:
        if step.kind != "tool":
            return "No tool execution needed."

        if step.tool_name == "calculate":
            obs = tool_calculate(step.tool_input or "")
            state.memory["calc_result"] = obs
            return obs

        if step.tool_name == "web_search":
            obs = tool_web_search(step.tool_input or "")
            state.memory["search_result"] = obs
            return obs

        return f"ERROR: unknown tool '{step.tool_name}'"


# ----------------------------
# The ReAct-ish control loop
# ----------------------------

def run_agent(goal: str, max_iters: int = 5) -> str:
    state = AgentState(goal=goal)
    planner = Planner()
    executor = Executor()

    for i in range(max_iters):
        step = planner.next_step(state)

        if step.kind == "finish":
            state.history.append((f"finish", step.message or ""))
            return step.message or ""

        # Execute tool step
        observation = executor.run_step(step, state)
        state.history.append((f"tool:{step.tool_name}({step.tool_input})", observation))

    return "Stopped: reached max_iters without finishing."


# ----------------------------
# Demo runs
# ----------------------------
if "__name__" == "__main__":
    print("=== Demo 1: Math goal ===")
    answer = run_agent("Calculate (12.5 * 4) + 3")
    print(answer)

    print("\n=== Demo 2: Search goal ===")
    answer = run_agent("Search Finland capital")
    print(answer)

    print("\n=== Demo 3: Mixed (planner could be expanded) ===")
    answer = run_agent("Look up Helsinki population")
    print(answer)

