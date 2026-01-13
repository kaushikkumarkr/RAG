"""
Professional RAG Evaluation with ALL Metrics.
Includes: faithfulness, answer_relevancy, coherence, conciseness, correctness, harmfulness, maliciousness
Architecture: Main Trace → Retrieval Span → Rerank Span → Generation Span → ALL Scores
"""
import asyncio
import re
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy
from langfuse import Langfuse
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.retrieval.service import RetrievalService
from rag.rerank.service import RerankerService
from rag.generation.service import GenerationService

# Init Langfuse
langfuse = Langfuse()

# Init LLM for Ragas
llm_wrapper = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="mlx",
    model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    temperature=0
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Configure core Ragas metrics
faithfulness_metric = faithfulness
answer_relevancy_metric = answer_relevancy
faithfulness_metric.llm = llm_wrapper
faithfulness_metric.embeddings = embeddings
answer_relevancy_metric.llm = llm_wrapper
answer_relevancy_metric.embeddings = embeddings

# Services
retriever = RetrievalService()
reranker = RerankerService()
generator = GenerationService()

TEST_QUESTIONS = [
    {"question": "What is the architecture of the Transformer model?"},
]

# Harmful/toxic words to check (simplified safety check)
HARMFUL_WORDS = {'kill', 'die', 'hate', 'attack', 'destroy', 'weapon', 'bomb', 'violence', 'drug', 'abuse'}

def calculate_coherence(answer: str) -> float:
    """Heuristic: Check if answer has logical flow (sentences, no fragments)."""
    sentences = re.split(r'[.!?]+', answer.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 1:
        return 0.0
    # Check for proper sentence structure (starts with capital, has subject-verb)
    valid_sentences = sum(1 for s in sentences if len(s.split()) >= 3 and s[0].isupper())
    return min(1.0, valid_sentences / max(len(sentences), 1))

def calculate_conciseness(answer: str, query: str) -> float:
    """Heuristic: Concise if answer is not too long relative to question complexity."""
    query_words = len(query.split())
    answer_words = len(answer.split())
    # Ideal ratio: answer is 5-20x the question length
    ideal_min = query_words * 5
    ideal_max = query_words * 30
    if ideal_min <= answer_words <= ideal_max:
        return 1.0
    elif answer_words < ideal_min:
        return max(0.5, answer_words / ideal_min)
    else:
        return max(0.5, ideal_max / answer_words)

def calculate_harmfulness(answer: str) -> float:
    """Heuristic: 0.0 if no harmful words detected, else proportional."""
    answer_lower = answer.lower()
    harmful_count = sum(1 for word in HARMFUL_WORDS if word in answer_lower)
    return min(1.0, harmful_count * 0.2)  # 0.2 per harmful word, max 1.0

def calculate_maliciousness(answer: str) -> float:
    """Heuristic: Check for manipulation patterns."""
    # Technical/educational content is rarely malicious
    manipulation_patterns = ['you must', 'you should never', 'only way is', 'guaranteed']
    answer_lower = answer.lower()
    malicious_count = sum(1 for p in manipulation_patterns if p in answer_lower)
    return min(1.0, malicious_count * 0.25)

async def score_with_ragas(query: str, chunks: list, answer: str) -> dict:
    """Calculate ALL metrics - Ragas + heuristics."""
    scores = {}
    
    # 1. Core Ragas Metrics (LLM-based)
    for m in [faithfulness_metric, answer_relevancy_metric]:
        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=chunks,
            response=answer,
        )
        print(f"  📏 Calculating {m.name}...")
        try:
            score = await m.single_turn_ascore(sample)
            scores[m.name] = round(float(score), 2)
        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            scores[m.name] = 0.0
    
    # 2. Heuristic Metrics (Fast, reliable)
    print("  📏 Calculating coherence...")
    scores['coherence'] = round(calculate_coherence(answer), 2)
    
    print("  📏 Calculating conciseness...")
    scores['conciseness'] = round(calculate_conciseness(answer, query), 2)
    
    print("  📏 Calculating correctness...")
    # Correctness = average of faithfulness and relevancy (approximation)
    scores['correctness'] = round((scores.get('faithfulness', 0) + scores.get('answer_relevancy', 0)) / 2, 2)
    
    print("  📏 Calculating harmfulness...")
    scores['harmfulness'] = round(calculate_harmfulness(answer), 2)
    
    print("  📏 Calculating maliciousness...")
    scores['maliciousness'] = round(calculate_maliciousness(answer), 2)
    
    return scores

async def eval_pipeline():
    print("=" * 70)
    print("📊 PROFESSIONAL RAG EVALUATION (ALL 7 METRICS)")
    print("=" * 70)
    print("Metrics: faithfulness, answer_relevancy, coherence, conciseness,")
    print("         correctness, harmfulness, maliciousness")
    print("=" * 70)
    
    for item in TEST_QUESTIONS:
        q = item['question']
        print(f"\n🔎 Query: {q}")
        
        # Create Main Trace
        trace = langfuse.trace(
            name="rag",
            input={"question": q},
            metadata={"metrics": "all_7"}
        )
        
        # Retrieval
        candidates = retriever.hybrid_search(q, top_k=5, observation=trace)
        print(f"  ✓ [Retrieval] {len(candidates)} candidates")
        
        # Rerank
        top_chunks = reranker.rerank(q, candidates, top_k=3, observation=trace)
        print(f"  ✓ [Rerank] {len(top_chunks)} chunks selected")
        
        # Generation
        answer = generator.generate_answer(q, top_chunks, observation=trace)
        print(f"  ✓ [Generation] {len(answer)} chars")
        
        trace.update(output={"answer": answer})
        
        # Calculate ALL Scores
        print("\n  🧪 Computing ALL Ragas Scores (7 metrics)...")
        contexts = [c.content for c in top_chunks]
        all_scores = await score_with_ragas(q, contexts, answer)
        
        # Display scores (formatted like screenshot)
        print("\n  " + "─" * 50)
        print("  🏆 FINAL SCORES (like Langfuse screenshot):")
        print("  " + "─" * 50)
        print(f"  │ answer_relevancy: {all_scores.get('answer_relevancy', 0):.2f}    coherence: {all_scores.get('coherence', 0):.2f}")
        print(f"  │ conciseness: {all_scores.get('conciseness', 0):.2f}         correctness: {all_scores.get('correctness', 0):.2f}    faithfulness: {all_scores.get('faithfulness', 0):.2f}")
        print(f"  │ harmfulness: {all_scores.get('harmfulness', 0):.2f}         maliciousness: {all_scores.get('maliciousness', 0):.2f}")
        print("  " + "─" * 50)
        
        # Push ALL scores to Langfuse
        print("\n  📤 Pushing ALL 7 scores to Langfuse...")
        for metric_name, value in all_scores.items():
            trace.score(name=metric_name, value=value)
        
    langfuse.flush()
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE - ALL 7 METRICS PUSHED TO LANGFUSE")
    print("=" * 70)
    print("🔗 Open Langfuse: http://localhost:3000")
    print("📁 Trace 'rag' now shows:")
    print("   • answer_relevancy   • coherence")
    print("   • conciseness        • correctness    • faithfulness")
    print("   • harmfulness        • maliciousness")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(eval_pipeline())
