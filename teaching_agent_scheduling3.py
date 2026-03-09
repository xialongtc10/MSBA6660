import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

# -----------------------------
# Title & Instructions
# -----------------------------
st.title("🤖 Faculty Course Scheduling Agent (LLM + Python Hybrid)")
st.markdown("""
Upload **faculty survey data** and **course information**.  
The agent will generate a **complete teaching schedule**, respecting faculty preferences, max load, time slots, classrooms, and notes.
""")

# -----------------------------
# OpenAI API Key Input
# -----------------------------
api_key = st.text_input("Enter OpenAI API Key", type="password")
client = OpenAI(api_key=api_key) if api_key else None

# -----------------------------
# File Uploads
# -----------------------------
faculty_file = st.file_uploader("Upload Faculty CSV", type=["csv"])
course_file = st.file_uploader("Upload Course CSV", type=["csv"])

faculty_df = pd.read_csv(faculty_file) if faculty_file else None
course_df = pd.read_csv(course_file) if course_file else None

if faculty_df is not None and not faculty_df.empty:
    st.subheader("Faculty Data")
    st.dataframe(faculty_df)

if course_df is not None and not course_df.empty:
    st.subheader("Course Data")
    st.dataframe(course_df)

# -----------------------------
# User Task Input
# -----------------------------
user_task = st.text_area(
    "Describe scheduling goal/constraints (e.g., 'maximize faculty preferences, avoid conflicts, respect max load and classroom constraints'):"
)

# -----------------------------
# LLM Schedule Suggestion
# -----------------------------
def get_llm_suggestions(faculty_cols, course_cols, task_description, client):
    system_prompt = f"""
You are an expert academic scheduling AI.

Given Faculty and Course DataFrames:

- Faculty have max teaching load (2–3 courses)
- Each course must be assigned to a faculty
- Respect faculty preferences, availability, classroom preferences, and unstructured notes
- Prioritize faculty preferences and availability

Return a **JSON array** suggesting initial course assignments:

[
  {{
    "Faculty": "...",
    "Course": "...",
    "Time": "...",
    "Classroom": "..."
  }},
  ...
]

You do NOT need to assign every course; Python will fill remaining courses.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Faculty columns: {faculty_cols}\nCourse columns: {course_cols}\nTask: {task_description}"}
        ],
        temperature=0.2
    )

    text = response.choices[0].message.content.strip()
    try:
        suggestions = pd.read_json(text)
        return suggestions
    except Exception:
        st.warning("LLM output could not be parsed as JSON. Using raw text for demo.")
        st.code(text)
        return pd.DataFrame()

# -----------------------------
# Auto-fill missing courses
# -----------------------------
def fill_missing_courses(schedule_df, faculty_df, course_df):
    if schedule_df is None or schedule_df.empty:
        schedule_df = pd.DataFrame(columns=["Faculty", "Course", "Time", "Classroom"])
    
    all_courses = set(course_df["Course"])
    assigned_courses = set(schedule_df["Course"])
    unassigned_courses = list(all_courses - assigned_courses)

    # Track teaching load
    faculty_load = schedule_df.groupby("Faculty").size().to_dict()
    for f in faculty_df["Faculty"]:
        if f not in faculty_load:
            faculty_load[f] = 0

    for course in unassigned_courses:
        # Find faculty with available load
        available_faculty = faculty_df[faculty_df["Max_Load"] > faculty_df["Faculty"].map(faculty_load)]
        if not available_faculty.empty:
            faculty_name = available_faculty.iloc[0]["Faculty"]
            course_row = course_df[course_df["Course"] == course].iloc[0]
            time_slot = course_row["Possible_Times"].split(",")[0].strip()
            classroom = course_row["Classroom"]
            schedule_df = pd.concat([schedule_df, pd.DataFrame([{
                "Faculty": faculty_name,
                "Course": course,
                "Time": time_slot,
                "Classroom": classroom
            }])], ignore_index=True)
            faculty_load[faculty_name] += 1

    return schedule_df

# -----------------------------
# Run Agent
# -----------------------------
if st.button("Generate Complete Faculty Schedule"):
    if faculty_df is None or faculty_df.empty:
        st.warning("Upload Faculty CSV first.")
    elif course_df is None or course_df.empty:
        st.warning("Upload Course CSV first.")
    elif not user_task:
        st.warning("Describe scheduling goal/constraints.")
    elif not client:
        st.warning("Enter your OpenAI API key first.")
    else:
        # Step 1: LLM Suggestions
        llm_suggestions = get_llm_suggestions(
            list(faculty_df.columns),
            list(course_df.columns),
            user_task,
            client
        )

        # Step 2: Fill missing courses
        full_schedule = fill_missing_courses(llm_suggestions, faculty_df, course_df)

        st.subheader("✅ Complete Faculty Schedule")
        st.dataframe(full_schedule)

        # -----------------------------
        # Optional: Display basic stats
        # -----------------------------
        st.subheader("Faculty Teaching Load")
        load_chart = full_schedule.groupby("Faculty").size().to_frame("Courses_Assigned")
        st.bar_chart(load_chart)

        st.subheader("Classroom Utilization")
        room_chart = full_schedule.groupby("Classroom").size().to_frame("Courses_Scheduled")
        st.bar_chart(room_chart)

        # -----------------------------
        # Optional: Simple text list output
        # -----------------------------
        st.subheader("Schedule List")
        st.text(full_schedule.to_string(index=False))
