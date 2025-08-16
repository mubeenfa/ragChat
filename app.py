"""
Streamlit UI for Local RAG — ChatGPT-like layout
Run: streamlit run app_streamlit.py
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Local RAG", page_icon="📚", layout="wide")

# Session state for conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=4, step=1)
    st.markdown("---")
    if st.button("Reload backend"):
        try:
            res = requests.post(f"{API_URL}/reload", timeout=60)
            if res.ok:
                st.success("Backend reloaded.")
            else:
                st.error(f"Reload failed: {res.status_code}")
        except Exception as e:
            st.error(f"Reload error: {e}")

# Display past conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input at the bottom (unchanged)
if question := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # Backend health check
    try:
        hc = requests.get(f"{API_URL}/health", timeout=10)
        if not hc.ok:
            st.error("Backend not healthy. Start FastAPI or check logs.")
    except Exception as e:
        st.error(f"Could not reach API: {e}")

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving & generating answer..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "k": top_k},
                    timeout=120
                )
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        if resp.ok:
            data = resp.json()
            answer = data["answer"]

            # Show answer
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Show references
            if data.get("references"):
                st.markdown("**📖 References:**")
                for i, ref in enumerate(data["references"], start=1):
                    src = ref.get("source", "unknown")
                    page = ref.get("page")
                    # dist = ref.get("distance")
                    # snippet = ref.get("snippet", "")
                    meta = f"**Source:** {src}"
                    if page is not None:
                        meta += f" | **Page:** {page}"
                    # if dist is not None:
                    #     meta += f" | **Distance:** {dist:.4f}"
                    st.markdown(f"{meta}")
                    # st.code(snippet)
            else:
                st.info("No references returned.")
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
