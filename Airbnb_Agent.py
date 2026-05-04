import streamlit as st
import pandas as pd
import json
from openai import OpenAI

# ----------------------------
# UI Configuration & Paper Context
# ----------------------------
st.set_page_config(page_title="Airbnb Remote Investment Agent", layout="wide")
st.title("🏠 Airbnb Remote Investment Consultant")
st.markdown("""
**Agentic Decision Support System** based on: 
*Xia, L. (2026). Decoding Airbnb location strategy for remote hosts: A location theory approach.*
""")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key:", type="password")
    # In a real scenario, you'd upload your paper's dataset (InsideAirbnb + Census data)
    st.info("Agent uses a Two-Tier Framework: Macro (City) & Micro (Neighborhood)")
    
    st.divider()
    st.markdown("### Strategic Queries")
    samples = [
        "Why is high listing density risky for remote hosts?",
        "Should I invest in a high job-growth city as a remote host?",
        "How much does proximity to attractions offset the remote host disadvantage?"
    ]
    clicked_q = None
    for q in samples:
        if st.button(q, use_container_width=True):
            clicked_q = q

# ----------------------------
# RAG Toolset: Retrieval Logic
# ----------------------------
def get_spatial_theory_context(query):
    """
    Simulates retrieval from the paper's findings.
    In production, this would query a vector DB of your paper's results.
    """
    knowledge_base = {
        "remote_gap": "Remote listings receive 12-16% fewer reviews than local ones.",
        "macro_job_market": "Job growth negatively affects remote performance (beta=-0.015) due to regulatory scrutiny.",
        "micro_attractions": "Nearby attractions (0.5mi) are the strongest predictors for remote success (beta=0.281).",
        "competition": "Neighborhood competition negatively impacts remote hosts (beta=-0.009)[cite: 2].",
        "diversity": "Racial diversity improves local performance but may challenge remote management (beta=-0.131)[cite: 2]."
    }
    # Simplified keyword retrieval for demo
    context = ""
    if "job" in query.lower(): context += knowledge_base["macro_job_market"]
    if "attraction" in query.lower(): context += knowledge_base["micro_attractions"]
    if "density" in query.lower() or "competition" in query.lower(): context += knowledge_base["competition"]
    if "gap" in query.lower() or "performance" in query.lower(): context += knowledge_base["remote_gap"]
    
    return context if context else "General location theory: Select moderate-tier cities and amenity-rich blocks[cite: 2]."

# ----------------------------
# Agent Execution Loop
# ----------------------------
query = st.text_input("Consult the Investment Agent:", value=clicked_q if clicked_q else "")

if query and api_key:
    client = OpenAI(api_key=api_key)
    
    with st.spinner("Analyzing spatial-economic factors..."):
        # Define the tool/function for the Agent
        tools = [{
            "type": "function",
            "function": {
                "name": "get_spatial_theory_context",
                "description": "Get specific regression findings and location theory insights from Dr. Xia's paper.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The user's investment concern"}
                    },
                    "required": ["query"]
                }
            }
        }]

        messages = [
            {"role": "system", "content": "You are a Real Estate Investment Agent specialized in Airbnb. Use the provided tools to cite specific coefficients (beta values) from the user's research paper[cite: 2]. Advice must distinguish between Macro and Micro tiers."},
            {"role": "user", "content": query}
        ]

        # Standard Agent Loop (similar to your Fabric Expert)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            context = get_spatial_theory_context(json.loads(tool_call.function.arguments)["query"])
            
            messages.append(msg)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": context})
            
            # Final generation with context
            final_res = client.chat.completions.create(model="gpt-4o", messages=messages)
            st.markdown("### Investment Recommendation")
            st.success(final_res.choices[0].message.content)
            
            with st.expander("Evidence from Paper (Table 3)"):
                st.write(context)
        else:
            st.info(msg.content)