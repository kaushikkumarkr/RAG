from typing import List, Dict
import re
from rag.generation.llm import LLMService
from rag.agent.tools import SearchTool

REACT_SYSTEM_PROMPT = """
You are a smart research assistant.
You have access to the following tools:

SearchTool: Use this to find facts. Input should be a specific search query.

Use the following format:

Question: the input question
Thought: you should always think about what to do
Action: the action to take, should be one of [SearchTool]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""

class AgentRunner:
    def __init__(self):
        self.llm = LLMService()
        self.tool = SearchTool()
        self.max_steps = 5

    def run(self, query: str) -> str:
        messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": f"Question: {query}"})
        
        for step in range(self.max_steps):
            # 1. LLM Generate Thought & Action
            response = self.llm.chat(messages)
            messages.append({"role": "assistant", "content": response})
            print(f"--- Step {step} ---\n{response}\n")

            # 2. Check for Final Answer
            if "Final Answer:" in response:
                return response.split("Final Answer:")[-1].strip()

            # 3. Parse Action
            action_match = re.search(r"Action: (\w+)", response)
            input_match = re.search(r"Action Input: (.*)", response)
            
            if action_match and input_match:
                action = action_match.group(1)
                action_input = input_match.group(1).strip()
                
                # 4. Execute Action
                observation = "Error: Tool not found."
                if action == "SearchTool":
                    observation = self.tool.search(action_input)
                
                print(f"Observation: {observation[:100]}...")

                # 5. Feed Observation back
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                # If LLM didn't follow format, ask it to try again
                messages.append({"role": "user", "content": "Please continue. If you have the answer, state 'Final Answer:'."})
        
        return "I could not find the answer after multiple steps."
