import streamlit as st
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import contextlib
import traceback

# -----------------------------
# OpenAI Setup
# -----------------------------
client = OpenAI(api_key="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")

# -----------------------------
# UI
# -----------------------------
st.title("📊 Operations Management AI Code Agent")

st.markdown("Upload a dataset and describe your operations analytics task.")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

df = None

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# -----------------------------
# Dataset Profiling
# -----------------------------
def profile_dataset(df):

    info = f"""
Columns: {list(df.columns)}
Shape: {df.shape}

Summary:
{df.describe().to_string()}

Missing values:
{df.isnull().sum().to_string()}
"""

    return info

# -----------------------------
# Code Generation Agent
# -----------------------------
def generate_code(task, dataset_info):

    system_prompt = """
You are an expert Operations Management data scientist.

Return ONLY executable Python code.

Rules:
- The dataset is already loaded as a pandas DataFrame called df.
- Do NOT redefine df.
- Use pandas, numpy, matplotlib, scipy, statsmodels, sklearn.
- Print important results.
- Generate plots when appropriate.
- Do NOT include explanations or markdown.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":f"Dataset info:\n{dataset_info}\n\nTask:\n{task}"}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# -----------------------------
# Clean LLM Code
# -----------------------------
def clean_code(code):

    if "```" in code:
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1]

    code = code.replace("python","")
    return code.strip()


# -----------------------------
# Safe Execution
# -----------------------------
def execute_code(code, df):

    local_vars = {
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt
    }

    output_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, {}, local_vars)

        return output_buffer.getvalue(), None

    except Exception:
        return None, traceback.format_exc()


# -----------------------------
# Auto Repair Agent
# -----------------------------
def repair_code(code, error):

    repair_prompt = f"""
The following Python code produced an error.

CODE:
{code}

ERROR:
{error}

Return corrected Python code only.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":repair_prompt}],
        temperature=0
    )

    return clean_code(response.choices[0].message.content)


# -----------------------------
# User Task
# -----------------------------
task = st.text_area("Describe your operations management task:")

# -----------------------------
# Run Agent
# -----------------------------
if st.button("Generate & Run Code"):

    if df is None:
        st.warning("Upload a dataset first.")

    elif not task:
        st.warning("Please describe your task.")

    else:

        st.subheader("Dataset Profiling")

        dataset_info = profile_dataset(df)
        st.text(dataset_info)

        st.subheader("Generating Code")

        code = generate_code(task, dataset_info)
        code = clean_code(code)

        st.subheader("Generated Code")
        st.code(code)

        st.subheader("Execution Output")

        output, error = execute_code(code, df)

        # -----------------------------
        # Auto Fix Loop
        # -----------------------------
        if error:

            st.warning("Initial execution failed. Attempting auto-repair...")

            fixed_code = repair_code(code, error)

            st.subheader("Repaired Code")
            st.code(fixed_code)

            output, error = execute_code(fixed_code, df)

        if error:
            st.error("Execution failed")
            st.code(error)

        else:
            st.text(output)
            st.pyplot(plt.gcf())
