import streamlit as st
import pandas as pd
import json
import numpy as np
from openai import OpenAI

# ----------------------------
# UI Configuration
# ----------------------------
st.set_page_config(page_title="Fabrics Expert QA - Agentic RAG", layout="wide")
st.title("🧵 Fabrics Expert QA - Agentic RAG")
st.markdown("Upload your `fabric_corpus_500.jsonl` and ask the expert system.")

# Sidebar for Configuration
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter OpenAI API Key:", type="password")
    uploaded_file = st.file_uploader("Upload Fabric Corpus (JSONL)", type=["jsonl"])
    
    st.divider()
    st.markdown("### Quick Demo Questions")
    sample_questions = [
        "What is the bleach ratio for Sunbrella mildew?",
        "How to clean coffee stains from Enduratex?",
        "Is acetone safe for MarineShield?",
        "How much soap for general maintenance of PatioPro?",
        "What is the fix for recurring mildew on ContractPlus?"
    ]
    
    clicked_q = None
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            clicked_q = q


# ----------------------------
# Helper Functions
# ----------------------------

def search_fabric_docs(query, corpus):
    """Simple retrieval tool the agent can call."""
    results = []
    
    for doc in corpus:
        if any(word.lower() in doc['search_text'].lower() for word in query.split()):
            results.append(doc)
    
    if not results:
        results = corpus[:3]
    
    return results[:3]


# ----------------------------
# Ingestion
# ----------------------------

if uploaded_file and api_key:
    client = OpenAI(api_key=api_key)
    
    if 'corpus_data' not in st.session_state:
        with st.spinner("Processing corpus..."):
            lines = uploaded_file.getvalue().decode("utf-8").splitlines()
            data = [json.loads(line) for line in lines]
            
            for item in data:
                item['search_text'] = f"{item['product_line']} {item['fabric_type']} {item['use_case']} {item['content']}"
            
            st.session_state.corpus_data = data
            st.success(f"Loaded {len(data)} documents.")


# ----------------------------
# Chat Interface
# ----------------------------

query = st.text_input("Ask a question about fabrics:", value=clicked_q if clicked_q else "")

if query and api_key:

    if 'corpus_data' not in st.session_state:
        st.warning("Please upload the corpus file first.")
        st.stop()

    with st.spinner("Agent thinking..."):

        corpus = st.session_state.corpus_data

        # ----------------------------
        # Tool Definition
        # ----------------------------
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_fabric_docs",
                    "description": "Search the fabric knowledge base for cleaning instructions or maintenance guidance.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        messages = [
            {
                "role": "system",
                "content": "You are a fabrics expert assistant. If you need information, call the search_fabric_docs tool. Keep answers under 25 words."
            },
            {
                "role": "user",
                "content": query
            }
        ]

        final_answer = None
        retrieved_docs = []

        # ----------------------------
        # Agent Loop
        # ----------------------------
        for step in range(3):

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0
            )

            msg = response.choices[0].message

            # If tool is called
            if msg.tool_calls:

                tool_call = msg.tool_calls[0]
                args = json.loads(tool_call.function.arguments)

                docs = search_fabric_docs(args["query"], corpus)
                retrieved_docs = docs

                context = "\n\n".join(
                    [f"Doc {d['doc_id']}: {d['content']}" for d in docs]
                )

                messages.append(msg)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": context
                })

            else:
                final_answer = msg.content
                break

        # ----------------------------
        # Display Results
        # ----------------------------

        if final_answer:
            st.markdown("### Answer")
            st.info(final_answer)

        if retrieved_docs:
            with st.expander("View Source Documents"):
                for d in retrieved_docs:
                    st.write(f"**{d['doc_id']}** - {d['product_line']}")
                    st.write(d['content'])
