"""
🏆 ULTIMATE PROFESSIONAL RAG EVALUATION
=========================================
Enterprise-grade RAG evaluation with complete Langfuse observability.

Features:
- Input Guardrails (jailbreak, PII, off-topic detection)
- Hybrid Retrieval + Cross-Encoder Reranking
- LLM Generation with Citations
- Citation Extraction & Validation (phantom detection)
- Multi-Signal Confidence Scoring
- Output Guardrails (toxicity, hallucination, refusal)
- Ragas Quality Metrics (faithfulness, relevancy)

Trace Structure:
📦 rag-professional
├── 🛡️ input_guardrails
├── 🔍 hybrid_search
├── 🔄 rerank
├── 💬 generation
├── 📚 citation_processing
├── 📊 confidence_scoring
├── 🛡️ output_guardrails
└── Scores: guard_*, faithfulness, answer_relevancy, confidence
"""
import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy
from langfuse import Langfuse
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import all services
from rag.retrieval.service import RetrievalService
from rag.rerank.service import RerankerService
from rag.generation.service import GenerationService
from rag.guardrails.service import GuardrailService
from rag.citations.service import CitationService
from rag.confidence.service import ConfidenceService

# Initialize Langfuse
langfuse = Langfuse()

# Initialize LLM for Ragas
llm_wrapper = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="mlx",
    model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    temperature=0
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Configure Ragas metrics
faithfulness_metric = faithfulness
answer_relevancy_metric = answer_relevancy
faithfulness_metric.llm = llm_wrapper
faithfulness_metric.embeddings = embeddings
answer_relevancy_metric.llm = llm_wrapper
answer_relevancy_metric.embeddings = embeddings

# Initialize all services
retriever = RetrievalService()
reranker = RerankerService()
generator = GenerationService()
guardrails = GuardrailService()
citations = CitationService()
confidence = ConfidenceService()

# Professional test suite
TEST_QUERIES = [
    {
        "query": "What is the architecture of the Transformer model?",
        "expected_topic": "transformer"
    },
    {
        "query": "How does attention work in neural networks?",
        "expected_topic": "attention"
    },
]


async def calculate_ragas_scores(query: str, contexts: list, answer: str) -> dict:
    """Calculate Ragas quality metrics."""
    scores = {}
    for metric in [faithfulness_metric, answer_relevancy_metric]:
        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=contexts,
            response=answer,
        )
        try:
            score = await metric.single_turn_ascore(sample)
            scores[metric.name] = round(float(score), 2)
        except Exception as e:
            scores[metric.name] = 0.0
    return scores


async def run_professional_evaluation():
    """Run enterprise-grade RAG evaluation."""
    
    print("=" * 80)
    print("🏆 ULTIMATE PROFESSIONAL RAG EVALUATION")
    print("=" * 80)
    print("Enterprise-grade evaluation with complete Langfuse observability")
    print("=" * 80)
    
    results = []
    
    for test_case in TEST_QUERIES:
        query = test_case["query"]
        print(f"\n{'─' * 80}")
        print(f"📝 Query: {query}")
        print(f"{'─' * 80}")
        
        # =====================================================
        # CREATE MAIN TRACE (Professional naming)
        # =====================================================
        trace = langfuse.trace(
            name="rag-professional",
            input={"question": query},
            metadata={
                "version": "2.0",
                "features": ["guardrails", "citations", "confidence"],
                "expected_topic": test_case.get("expected_topic")
            }
        )
        
        # =====================================================
        # STEP 1: INPUT GUARDRAILS
        # =====================================================
        print("  [1/7] 🛡️ Input Guardrails...", end=" ")
        input_check = guardrails.check_input(query, observation=trace)
        
        # Log guardrail scores
        for result in input_check.results:
            trace.score(
                name=f"guard_input_{result.guard_type.value}",
                value=1.0 if not result.triggered else 0.0
            )
        
        if not input_check.passed:
            print(f"BLOCKED ({input_check.blocked_by.value})")
            trace.update(output={"status": "blocked", "reason": input_check.blocked_by.value})
            continue
        print("✅ Passed")
        
        # =====================================================
        # STEP 2: RETRIEVAL
        # =====================================================
        print("  [2/7] 🔍 Hybrid Search...", end=" ")
        candidates = retriever.hybrid_search(query, top_k=10, observation=trace)
        print(f"✅ {len(candidates)} candidates")
        
        # =====================================================
        # STEP 3: RERANKING
        # =====================================================
        print("  [3/7] 🔄 Cross-Encoder Reranking...", end=" ")
        top_chunks = reranker.rerank(query, candidates, top_k=3, observation=trace)
        print(f"✅ Top {len(top_chunks)} selected")
        
        # =====================================================
        # STEP 4: GENERATION
        # =====================================================
        print("  [4/7] 💬 LLM Generation...", end=" ")
        raw_answer = generator.generate_answer(query, top_chunks, observation=trace)
        print(f"✅ {len(raw_answer)} chars")
        
        # =====================================================
        # STEP 5: CITATION PROCESSING
        # =====================================================
        print("  [5/7] 📚 Citation Processing...", end=" ")
        citation_result = citations.process(raw_answer, top_chunks, observation=trace)
        formatted_answer = citation_result.formatted_answer
        
        # Log citation metrics
        trace.score(name="citation_count", value=float(citation_result.citation_count))
        trace.score(name="citation_valid", value=float(citation_result.valid_count))
        trace.score(name="citation_phantom", value=float(len(citation_result.phantom_citations)))
        
        if citation_result.has_phantoms:
            print(f"⚠️ {len(citation_result.phantom_citations)} phantom citations")
        else:
            print(f"✅ {citation_result.valid_count} valid citations")
        
        # =====================================================
        # STEP 6: CONFIDENCE SCORING
        # =====================================================
        print("  [6/7] 📊 Confidence Scoring...", end=" ")
        confidence_result = confidence.calculate(
            formatted_answer, top_chunks, citation_result, observation=trace
        )
        
        # Log confidence score and breakdown
        trace.score(name="confidence", value=confidence_result.score)
        for signal_name, signal_value in confidence_result.breakdown.items():
            trace.score(name=f"conf_{signal_name}", value=round(signal_value, 2))
        
        print(f"✅ {confidence_result.score:.2f} ({confidence_result.level})")
        
        # =====================================================
        # STEP 7: OUTPUT GUARDRAILS
        # =====================================================
        print("  [7/7] 🛡️ Output Guardrails...", end=" ")
        output_check = guardrails.check_output(formatted_answer, top_chunks, observation=trace)
        
        for result in output_check.results:
            trace.score(
                name=f"guard_output_{result.guard_type.value}",
                value=1.0 if not result.triggered else 0.0
            )
        
        if not output_check.passed:
            print(f"BLOCKED ({output_check.blocked_by.value})")
            formatted_answer = "Response blocked due to safety constraints."
        else:
            print("✅ Passed")
        
        # Update trace with final answer
        trace.update(output={
            "answer": formatted_answer,
            "confidence": confidence_result.score,
            "confidence_level": confidence_result.level,
            "citations": citation_result.valid_count
        })
        
        # =====================================================
        # RAGAS QUALITY EVALUATION
        # =====================================================
        print("\n  📈 Computing Ragas Quality Metrics...")
        contexts = [c.content for c in top_chunks]
        ragas_scores = await calculate_ragas_scores(query, contexts, formatted_answer)
        
        for metric_name, score in ragas_scores.items():
            print(f"      • {metric_name}: {score:.2f}")
            trace.score(name=metric_name, value=score)
        
        # =====================================================
        # DISPLAY FINAL RESULTS
        # =====================================================
        print("\n  ╔════════════════════════════════════════════════════════╗")
        print("  ║                   📊 FINAL RESULTS                     ║")
        print("  ╠════════════════════════════════════════════════════════╣")
        print(f"  ║ Confidence: {confidence_result.score:.2f} ({confidence_result.level:^8})                     ║")
        print(f"  ║ Faithfulness: {ragas_scores.get('faithfulness', 0):.2f}                               ║")
        print(f"  ║ Relevancy: {ragas_scores.get('answer_relevancy', 0):.2f}                                  ║")
        print(f"  ║ Citations: {citation_result.valid_count} valid, {len(citation_result.phantom_citations)} phantom                       ║")
        print("  ╚════════════════════════════════════════════════════════╝")
        
        results.append({
            "query": query,
            "confidence": confidence_result.score,
            "faithfulness": ragas_scores.get("faithfulness", 0),
            "relevancy": ragas_scores.get("answer_relevancy", 0),
            "citations": citation_result.valid_count
        })
    
    # Flush to Langfuse
    langfuse.flush()
    
    # =====================================================
    # SUMMARY
    # =====================================================
    print("\n" + "=" * 80)
    print("🏆 EVALUATION COMPLETE")
    print("=" * 80)
    
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["relevancy"] for r in results) / len(results)
    
    print(f"""
📊 AGGREGATE METRICS
────────────────────
  • Average Confidence:   {avg_confidence:.2f}
  • Average Faithfulness: {avg_faithfulness:.2f}
  • Average Relevancy:    {avg_relevancy:.2f}
  • Total Queries:        {len(results)}

🔗 LANGFUSE DASHBOARD
────────────────────
  Open: http://localhost:3000
  
  Each trace 'rag-professional' shows:
  ├── 🛡️ input_guardrails (Span)
  ├── 🔍 hybrid_search (Span)
  ├── 🔄 rerank (Span)
  ├── 💬 generation (Span)
  ├── 📚 citation_processing (Span)
  ├── 📊 confidence_scoring (Span)
  └── 🛡️ output_guardrails (Span)
  
  Scores attached to each trace:
  • guard_input_*      (Input safety)
  • guard_output_*     (Output safety)
  • citation_*         (Citation metrics)
  • confidence         (Overall confidence)
  • conf_*             (Confidence breakdown)
  • faithfulness       (Ragas)
  • answer_relevancy   (Ragas)
""")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_professional_evaluation())
