import streamlit as st
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from rag_chain import build_rag_chain

st.set_page_config(
    page_title = "Credit Policy Interpreter",
    page_icon = "🏦",
    layout = 'centered'
)

st.title("🏦 Credit Policy Interpreter")
st.caption("Ask questions about RBI and NBFC regulatory guidelines. Powered by RAG + Llama 3.3 70B.")


@st.cache_resource(show_spinner="Loading regulatory documents and building knowledge base...")
def get_rag_chain():
    chain, count = build_rag_chain()
    return chain, count


rag_chain, chunk_count = get_rag_chain()

st.sidebar.header("About")
st.sidebar.write(f"**Knowledge base:** {chunk_count} document chunks indexed")
st.sidebar.write("**Documents:** RBI Digital Lending Guidelines, RBI Fair Practices Code")
st.sidebar.write("**Model:** Llama 3.3 70B via Groq")
st.sidebar.write("**Embedding:** all-MiniLM-L6-v2")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by [Arindam Bhadra](https://github.com/joy4t)")

# Chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.chat_input("Ask a question about RBI/NBFC regulations...")

if question:
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching regulatory documents..."):
            answer = rag_chain.invoke(question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})