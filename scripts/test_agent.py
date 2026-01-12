from rag.agent.runner import AgentRunner

def main():
    agent = AgentRunner()
    
    # Complex question requiring retention of multiple facts
    query = "Compare the year AI was founded as an academic discipline with the year Deep Blue defeated Kasparov."
    
    print(f"Goal: {query}\n")
    answer = agent.run(query)
    
    print(f"\nFinal Answer: {answer}")

if __name__ == "__main__":
    main()
