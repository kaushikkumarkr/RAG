import time
import json
import logging
from typing import Dict, Any, List
from rag.retrieval.service import RetrievalService
from rag.rerank.service import RerankerService
from rag.generation.service import GenerationService
from rag.agent.runner import AgentRunner
from apps.api.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CompareRAG")

DATASET_PATH = "eval/comparison_dataset.json"
RESULTS_PATH = "eval/comparison_results.json"

class ComparisonBenchmark:
    def __init__(self):
        # Initialize Standard Pipeline Components
        self.retriever = RetrievalService()
        self.reranker = RerankerService()
        self.generator = GenerationService()
        
        # Initialize Agentic Pipeline
        self.agent = AgentRunner()

    def run_standard_rag(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            # 1. Retrieve
            candidates = self.retriever.hybrid_search(query, top_k=20)
            # 2. Rerank
            ranked_chunks = self.reranker.rerank(query, candidates, top_k=5)
            # 3. Generate
            answer = self.generator.generate_answer(query, ranked_chunks)
            duration = time.time() - start_time
            return {
                "answer": answer,
                "latency_seconds": duration,
                "error": None
            }
        except Exception as e:
            return {
                "answer": None,
                "latency_seconds": time.time() - start_time,
                "error": str(e)
            }

    def run_agentic_rag(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            answer = self.agent.run(query)
            duration = time.time() - start_time
            return {
                "answer": answer,
                "latency_seconds": duration,
                "error": None
            }
        except Exception as e:
            return {
                "answer": None,
                "latency_seconds": time.time() - start_time,
                "error": str(e)
            }

    def evaluate(self):
        logger.info(f"Loading dataset from {DATASET_PATH}")
        with open(DATASET_PATH, 'r') as f:
            questions = json.load(f)

        results = []
        
        for idx, item in enumerate(questions):
            q = item["question"]
            q_type = item["type"]
            logger.info(f"Processing Q{idx+1} ({q_type}): {q}")

            # Run Standard
            std_res = self.run_standard_rag(q)
            logger.info(f"  Standard: {std_res['latency_seconds']:.2f}s")

            # Run Agentic
            agent_res = self.run_agentic_rag(q)
            logger.info(f"  Agentic:  {agent_res['latency_seconds']:.2f}s")

            results.append({
                "question": q,
                "type": q_type,
                "ground_truth": item["ground_truth"],
                "standard": std_res,
                "agentic": agent_res
            })

        # Save Results
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.print_summary(results)

    def print_summary(self, results):
        print("\n" + "="*50)
        print("COMPARATIVE EVALUATION RESULTS")
        print("="*50)
        print(f"{'Question Type':<20} | {'Method':<10} | {'Latency (s)':<10}")
        print("-" * 46)

        # Aggregate stats
        types = set(r["type"] for r in results)
        for t in types:
            subset = [r for r in results if r["type"] == t]
            avg_std = sum(r["standard"]["latency_seconds"] for r in subset) / len(subset)
            avg_agt = sum(r["agentic"]["latency_seconds"] for r in subset) / len(subset)
            
            print(f"{t:<20} | Standard   | {avg_std:.2f}")
            print(f"{'':<20} | Agentic    | {avg_agt:.2f}")
            print("-" * 46)

if __name__ == "__main__":
    benchmark = ComparisonBenchmark()
    benchmark.evaluate()
