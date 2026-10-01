import re
import asyncio
from rag.generation.llm import LLMService
from rag.agent.tools import SearchTool, CryptoPriceTool
from rag.agent.mcp_tools import MCPToolRegistry

REACT_SYSTEM_PROMPT = """
You are a smart research assistant.
You have access to these local tools:

1. SearchTool: Use this to find facts from the knowledge base. Input should be a specific search query.
2. CryptoPriceTool: Use this to get live cryptocurrency prices. Input should be the full name (e.g., bitcoin).

You also have these MCP tools:
{mcp_tools}

Use the following format:

Question: the input question
Thought: you should always think about what to do
Action: exact tool name from the list above
Action Input: JSON object matching that tool's input schema. For SearchTool and CryptoPriceTool, use a plain string.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""

class AgentRunner:
    def __init__(self):
        self.llm = LLMService()
        self.search_tool = SearchTool()
        self.crypto_tool = CryptoPriceTool()
        self.max_steps = 5

    async def _run(self, query: str) -> str:
        async with MCPToolRegistry() as mcp_tools:
            system_prompt = REACT_SYSTEM_PROMPT.format(
                mcp_tools=mcp_tools.prompt_description() or "No MCP tools are available."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {query}"},
            ]

            for step in range(self.max_steps):
                response = self.llm.chat(messages)
                messages.append({"role": "assistant", "content": response})
                print(f"--- Step {step} ---\n{response}\n")

                if "Final Answer:" in response:
                    return response.split("Final Answer:")[-1].strip()

                action_match = re.search(r"Action:\s*([\w.]+)", response)
                input_match = re.search(r"Action Input:\s*(.+)", response, re.DOTALL)
                if action_match and input_match:
                    action = action_match.group(1).strip()
                    action_input = input_match.group(1).strip()
                    if action == "SearchTool":
                        observation = self.search_tool.search(action_input.strip('"'))
                    elif action == "CryptoPriceTool":
                        observation = self.crypto_tool.get_price(action_input.strip('"'))
                    else:
                        observation = await mcp_tools.call(action, action_input)
                    print(f"Observation: {observation[:100]}...")
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                else:
                    messages.append({
                        "role": "user",
                        "content": "Continue. Use the exact Action / Action Input format or give a Final Answer.",
                    })

            return "I could not find the answer after multiple steps."

    def run(self, query: str) -> str:
        """Synchronous entry point retained for the existing scripts and callers."""
        return asyncio.run(self._run(query))
