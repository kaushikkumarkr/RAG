import streamlit as st
import requests
import json

# Configuration
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Foundry",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Foundry")

# Sidebar Configuration
with st.sidebar:
    st.header("Search Configuration")
    use_hybrid = st.toggle("Use Hybrid Search", value=True)
    
    if use_hybrid:
        alpha = st.slider("Alpha (Dense Weight)", 0.0, 1.0, 0.5, 0.1)
    else:
        alpha = None
        
    top_k = st.slider("Top K Retrieval", 1, 20, 5)
    
    st.divider()
    st.header("Ingestion")
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "md"])
    if uploaded_file is not None:
        if st.button("Ingest File"):
            with st.spinner("Ingesting..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                try:
                    response = requests.post(f"{API_URL}/ingest/file", files=files)
                    if response.status_code == 200:
                        st.success(f"Ingested {uploaded_file.name} successfully!")
                        st.json(response.json())
                    else:
                        st.error(f"Failed to ingest: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {e}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            payload = {
                "question": prompt,
                "use_hybrid": use_hybrid,
                "top_k": top_k
            }
            if alpha is not None:
                payload["alpha"] = alpha
                
            response = requests.post(f"{API_URL}/ask", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer returned.")
                context = data.get("citations", [])
                
                # Display Answer
                message_placeholder.markdown(answer)
                
                # Display Citations in Expander
                if context:
                    with st.expander("📚 Sources"):
                        for idx, item in enumerate(context):
                            st.markdown(
                                f"**Source {idx+1}** · chunk {item.get('chunk_index', 'N/A')} "
                                f"· score {item.get('score', 0):.3f}"
                            )
                            st.caption(f"Document: {item.get('doc_id', 'N/A')}")
                            st.text(item.get("content", ""))
                            st.divider()
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
        except Exception as e:
            error_msg = f"Connection Error: {e}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.divider()
with st.container(border=True):
    st.subheader("🧭 MCP Research Agent")
    st.caption("Ask the agent to use the project’s Filesystem, Fetch, Git, or Memory tools.")
    agent_question = st.text_input(
        "Question for the agent",
        value="Read rag_foundry_overview.md and tell me which vector database it uses.",
        key="agent_question",
    )
    if st.button("Run MCP agent", type="primary", use_container_width=True):
        with st.spinner("Agent is selecting and calling tools..."):
            try:
                response = requests.post(
                    f"{API_URL}/agent/ask",
                    json={"question": agent_question},
                    timeout=180,
                )
                response.raise_for_status()
                st.session_state["agent_answer"] = response.json().get(
                    "answer", "No answer returned."
                )
            except requests.RequestException as e:
                st.session_state["agent_answer"] = f"Agent request failed: {e}"

    if st.session_state.get("agent_answer"):
        st.markdown(st.session_state["agent_answer"])
