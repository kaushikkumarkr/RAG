from rag.agent.runner import AgentRunner
import sys

def main():
    print("Initializing Agent with Live Tools...")
    runner = AgentRunner()
    
    query = "What is the current price of bitcoin?"
    print(f"\nQuery: {query}")
    
    answer = runner.run(query)
    
    print("\nFinal Answer Received:")
    print(answer)
    
    print("Flushing traces...")
    try:
        from langfuse import flush
        flush()
    except ImportError:
        pass
    
    if "$" in answer or "USD" in answer:
        print("\nSUCCESS: Answer contains currency symbol.")
        sys.exit(0)
    else:
        print("\nFAILURE: Answer does not look like a price.")
        sys.exit(1)

if __name__ == "__main__":
    main()
