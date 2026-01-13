import asyncio
import os
import json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langfuse import Langfuse
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import our content services
from rag.retrieval.service import RetrievalService
from rag.rerank.service import RerankerService
from rag.generation.service import GenerationService

# Init Langfuse
langfuse = Langfuse()

# Init Ragas LLM/Embeddings
start_url = "http://localhost:8080/v1"
llm = ChatOpenAI(
    base_url=start_url,
    api_key="mlx",
    model="mlx-community/Qwen2.5-7B-Instruct-4bit", 
    temperature=0
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

async def evaluate_row(question: str, ground_truth: str):
    # 1. Start a Trace
    trace = langfuse.trace(
        name="rag-evaluation",
        input={"question": question},
        metadata={"ground_truth": ground_truth}
    )
    
    # 2. Run Pipeline (Manually instrumented for clarity or just call services)
    # Using handler to link is complex with decorators, so we just run it.
    # The decorators in services will create DETACHED traces unless we pass context.
    # For now, let's just focus on getting the Answer + Context for Ragas.
    
    retriever = RetrievalService()
    reranker = RerankerService()
    generator = GenerationService()
    
    # Retrieval
    retrieval_span = trace.span(name="retrieval")
    candidates = retriever.hybrid_search(question, top_k=5)
    retrieval_span.end(output=len(candidates))
    
    # Rerank
    rerank_span = trace.span(name="rerank")
    top_chunks = reranker.rerank(question, candidates, top_k=3)
    rerank_span.end(output=len(top_chunks))
    
    # Generation
    gen_span = trace.span(name="generation")
    answer = generator.generate_answer(question, top_chunks)
    gen_span.end(output=answer)
    
    contexts = [c.content for c in top_chunks]
    
    # 3. Prepare dataset for Ragas (single row)
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth]
    }
    dataset = Dataset.from_dict(data)
    
    # 4. Run Ragas
    print(f"Evaluating: {question}")
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings
    )
    
    # 5. Push Scores to trace
    # Ragas returns a Result object.
    # Safe way: convert to pandas
    df_scores = scores.to_pandas()
    res = df_scores.iloc[0].to_dict()
    print(f"Scores: {res}")
    
    for metric, value in res.items():
        # Filter out non-metric columns like question/answer
        if isinstance(value, (int, float)) and metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
             trace.score(
                 name=metric,
                 value=value
             )
    
    trace.update(output=answer)
    
    return res

def main():
    print("Loading FiQA dataset (explodinggradients/fiqa)...")
    from datasets import load_dataset
    # Load dataset as per Ragas docs
    fiqa = load_dataset("explodinggradients/fiqa", "ragas_eval")["baseline"]
    
    # Select small sample for local testing (LLM is slow)
    sample_size = 3
    print(f"Selecting {sample_size} examples for End-to-End Test...")
    
    # FiQA structure: 'question', 'answer' (Ground Truth), 'contexts' (Gold chunks), 'ground_truths'
    # tailored for Ragas. We just need Question + GT.
    
    for i in range(sample_size):
        row = fiqa[i]
        question = row['question']
        # 'answer' or 'ground_truths'? Dataset usually has 'ground_truths' as list
        # Let's inspect or handle safely.
        # Ragas baseline often has 'ground_truths' list.
        gt = row.get('ground_truths', [row.get('answer', '')])[0]
        
        asyncio.run(evaluate_row(question, gt))

    print("Flushing data to Langfuse...")
    langfuse.flush()

if __name__ == "__main__":
    main()
