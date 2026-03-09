import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import json

# -----------------------------
st.title("🤖 LLM Scheduling / Resource Allocation Agent")
st.markdown("""
Upload a task CSV and describe your scheduling goal/constraints.
The LLM will generate a schedule directly and visualize it.
""")

# -----------------------------
api_key = st.text_input("Enter OpenAI API Key", type="password")
client = OpenAI(api_key=api_key) if api_key else None

# -----------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
df = pd.read_csv(uploaded_file) if uploaded_file else None

if df is not None and not df.empty:
    st.subheader("Task Dataset")
    st.dataframe(df)

user_task = st.text_area(
    "Describe scheduling goal/constraints (e.g., 'maximize priority, avoid overlapping Resource_2 tasks'):"
)

# -----------------------------
def generate_schedule(df_columns, task_description, client):
    """
    LLM returns schedule directly as JSON
    """
    system_prompt = f"""
You are an expert Operations Management data scientist.

You are given a pandas DataFrame called df with columns: {df_columns}.
The user wants a schedule based on: {task_description}

Return the schedule directly as JSON array with objects:
[
  {{
    "Task": "...",
    "Resource": "...",
    "Start": number,
    "End": number
  }},
  ...
]

Do NOT return Python code.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Columns: {df_columns}\nTask: {task_description}"}
        ],
        temperature=0
    )
    text = response.choices[0].message.content.strip()
    # Attempt to parse JSON
    try:
        schedule_json = json.loads(text)
        return pd.DataFrame(schedule_json)
    except Exception:
        st.error("Failed to parse schedule from LLM output. Here is raw output:")
        st.code(text)
        return None

# -----------------------------
if st.button("Generate Schedule"):
    if df is None or df.empty:
        st.warning("Upload a dataset first.")
    elif not user_task:
        st.warning("Describe your scheduling goal/constraints.")
    elif not client:
        st.warning("Enter your OpenAI API key first.")
    else:
        schedule_df = generate_schedule(list(df.columns), user_task, client)
        if schedule_df is not None and not schedule_df.empty:
            st.subheader("Scheduled Tasks")
            st.dataframe(schedule_df)

            # Gantt chart
            fig, ax = plt.subplots(figsize=(10,5))
            colors = plt.cm.tab20.colors
            resources = schedule_df["Resource"].unique()
            for i, resource in enumerate(resources):
                res_tasks = schedule_df[schedule_df["Resource"] == resource]
                ax.barh(
                    y=[resource]*len(res_tasks),
                    width=res_tasks["End"]-res_tasks["Start"],
                    left=res_tasks["Start"],
                    color=colors[i % len(colors)],
                    edgecolor="black"
                )
                for _, row in res_tasks.iterrows():
                    ax.text(
                        row["Start"] + (row["End"]-row["Start"])/2,
                        i,
                        row["Task"],
                        va="center", ha="center",
                        color="white", fontsize=8
                    )
            ax.set_xlabel("Time")
            ax.set_ylabel("Resource")
            ax.set_title("Gantt Chart of Scheduled Tasks")
            st.pyplot(fig)

            # Resource utilization
            st.subheader("Resource Utilization")
            utilization = schedule_df.groupby("Resource").apply(lambda x: x["End"].max()).to_frame(name="Total Hours")
            st.bar_chart(utilization)
