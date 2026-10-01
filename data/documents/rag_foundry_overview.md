# RAG Foundry overview

RAG Foundry ingests PDF, Markdown, and text documents. It splits documents into
chunks, embeds them with `all-MiniLM-L6-v2`, and stores dense vectors in Qdrant.
It also builds a BM25 sparse index. Hybrid retrieval combines normalized dense
and sparse scores, then a cross-encoder reranks the candidates.

The configured local generation model is Qwen2.5-7B-Instruct-4bit served with
MLX on Apple Silicon. The project includes a ReAct agent and the following MCP
connectors: Filesystem for local text documents, Fetch for public web pages, Git
for repository history, and Memory for persistent facts.

The evaluation workflow checks input and output guardrails, validates citation
references, calculates a five-signal confidence score, and can record Ragas
faithfulness and answer-relevancy scores in Langfuse.
