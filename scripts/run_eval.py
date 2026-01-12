import asyncio
import json
import yaml
import sys
import os
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.retrieval.service import RetrievalService
from rag.rerank.service import RerankerService
from rag.generation.service import GenerationService

# Configure Ragas to use Local LLM
# Note: Ragas uses LangChain abstractions
start_url = "http://localhost:8080/v1"
llm = ChatOpenAI(
    base_url=start_url,
    api_key="mlx",
    model="mlx-community/Qwen2.5-7B-Instruct-4bit", 
    temperature=0,
    request_timeout=360
)

# Use local embeddings for metrics to be robust
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_data(path: str) -> List[Dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_thresholds(path: str) -> Dict[str, float]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

async def run_pipeline(questions: List[Dict]):
    retrieval_service = RetrievalService()
    reranker_service = RerankerService()
    generation_service = GenerationService()

    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print(f"Running evaluation on {len(questions)} examples...")

    for item in questions:
        q = item["question"]
        gt = item["ground_truth"]
        
        # 1. Retrieval
        candidates = retrieval_service.hybrid_search(q, top_k=10)
        
        # 2. Rerank
        top_chunks = reranker_service.rerank(q, candidates, top_k=5)
        
        # 3. Generate
        answer = generation_service.generate_answer(q, top_chunks)
        
        # Collect contexts content
        contexts = [c.content for c in top_chunks]
        
        results["question"].append(q)
        results["answer"].append(answer)
        results["contexts"].append(contexts)
        results["ground_truth"].append(gt)
        
        print(f"Processed: {q}")

    return Dataset.from_dict(results)

def main():
    questions = load_data("eval/eval_questions_sanity.jsonl")
    dataset = asyncio.run(run_pipeline(questions))
    
    print("Calculating RAGAS metrics (this may take a while)...")
    
    # We assign the LLM/Embeddings to the metrics
    # Note: In newer Ragas versions, you pass llm/embeddings to evaluate()
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )
    
    print("\n=== Evaluation Results ===")
    print(scores)
    
    # Save results
    df = scores.to_pandas()
    df.to_json("eval/results.json", orient="records", indent=2)
    print("Detailed results saved to eval/results.json")
    
    # Check thresholds
    thresholds = load_thresholds("eval/thresholds.yaml")
    failed = False
    
    # Convert Result object to dict for safe access
    scores_dict = dict(scores)
    
    for metric, threshold in thresholds.items():
        if metric not in scores_dict:
            continue
            
        score = scores_dict[metric]
        if score < threshold:
            print(f"FAIL: {metric} ({score:.4f}) < threshold ({threshold})")
            failed = True
        else:
            print(f"PASS: {metric} ({score:.4f}) >= threshold ({threshold})")
            
    if failed:
        sys.exit(1)
    else:
        print("\nAll quality gates passed!")

if __name__ == "__main__":
    main()
