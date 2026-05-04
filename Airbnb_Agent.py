import streamlit as st
import pandas as pd
import numpy as np
import json
from openai import OpenAI

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Airbnb Location AI Agent", layout="wide")

st.title("🏡 Airbnb Location Intelligence Agent (ALIA)")
st.markdown("AI-powered location strategy for remote Airbnb investment")

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")

    st.markdown("### Example Queries")
    demo_qs = [
        "Best city for remote Airbnb investment?",
        "Compare Austin vs Nashville",
        "What kind of neighborhood should I choose?",
        "Where should I invest with low competition?"
    ]

    selected_q = None
    for q in demo_qs:
        if st.button(q):
            selected_q = q

# ----------------------------
# Sample DATA (replace later)
# ----------------------------
city_data = pd.DataFrame([
    {"city": "Austin", "tourism": 8, "jobs": 7, "density": 9},
    {"city": "Nashville", "tourism": 7, "jobs": 6, "density": 6},
    {"city": "Asheville", "tourism": 6, "jobs": 5, "density": 4},
])

neighborhood_data = pd.DataFrame([
    {"city": "Austin", "area": "Downtown", "attractions": 9, "distance": 1, "competition": 9},
    {"city": "Austin", "area": "Zilker", "attractions": 8, "distance": 2, "competition": 6},
    {"city": "Nashville", "area": "The Gulch", "attractions": 8, "distance": 1, "competition": 7},
])

# ----------------------------
# SCORING FUNCTIONS (YOUR PAPER)
# ----------------------------

def score_city(row):
    # Based on your findings:
    # - high tourism can be BAD (saturation)
    # - high density BAD
    # - moderate jobs GOOD

    return (
        row["jobs"] * 0.4 +
        (10 - row["density"]) * 0.4 +
        (10 - row["tourism"]) * 0.2
    )


def score_neighborhood(row):
    return (
        row["attractions"] * 0.5 +
        (10 - row["competition"]) * 0.4 +
        (10 - row["distance"]) * 0.1
    )

# ----------------------------
# TOOL: SCORING ENGINE
# ----------------------------

def run_location_analysis():

    df = city_data.copy()
    df["score"] = df.apply(score_city, axis=1)

    if df.empty:
        return {"error": "No city data available"}

    best_city = df.sort_values("score", ascending=False).iloc[0]

    # Normalize strings (VERY IMPORTANT)
    city_name = str(best_city["city"]).strip().lower()

    ndf = neighborhood_data.copy()
    ndf["city_clean"] = ndf["city"].str.strip().str.lower()

    ndf = ndf[ndf["city_clean"] == city_name]

    # 🚨 KEY FIX: handle empty neighborhoods
    if ndf.empty:
        return {
            "city": best_city["city"],
            "city_score": round(best_city["score"], 2),
            "area": "No neighborhood data available",
            "area_score": None,
            "note": "No matching neighborhood found. Check data consistency."
        }

    ndf["score"] = ndf.apply(score_neighborhood, axis=1)

    best_area = ndf.sort_values("score", ascending=False).iloc[0]

    return {
        "city": best_city["city"],
        "city_score": round(best_city["score"], 2),
        "area": best_area["area"],
        "area_score": round(best_area["score"], 2)
    }


# ----------------------------
# RAG: PAPER KNOWLEDGE BASE
# ----------------------------

paper_chunks = [
    "Remote hosts perform worse in highly saturated tourism markets.",
    "Moderate tourism cities outperform high-tourism cities for remote hosts.",
    "Neighborhoods with strong attractions improve performance.",
    "High competition reduces performance for remote hosts.",
    "Location selection is the MOST important decision before operation."
]

def search_knowledge(query):
    return paper_chunks[:3]

# ----------------------------
# AGENT
# ----------------------------

query = st.text_input("Ask your Airbnb investment question:", value=selected_q if selected_q else "")

if query and api_key:

    client = OpenAI(api_key=api_key)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_location_analysis",
                "description": "Run macro and micro Airbnb location scoring",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "Retrieve Airbnb strategy knowledge",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    messages = [
        {
            "role": "system",
            "content": """
            You are an Airbnb investment AI consultant.
            Use tools to:
            1. Analyze best locations
            2. Explain reasoning using research insights
            """
        },
        {"role": "user", "content": query}
    ]

    with st.spinner("Agent thinking..."):

        for _ in range(3):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                tool_call = msg.tool_calls[0]

                if tool_call.function.name == "run_location_analysis":
                    result = run_location_analysis()
                    messages.append(msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })

                elif tool_call.function.name == "search_knowledge":
                    args = json.loads(tool_call.function.arguments)
                    docs = search_knowledge(args["query"])

                    messages.append(msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "\n".join(docs)
                    })

            else:
                final_answer = msg.content
                break

    st.markdown("### 📊 Recommendation")
    st.success(final_answer)
