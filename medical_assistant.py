import streamlit as st
from openai import OpenAI

# --- UI Configuration ---
st.set_page_config(page_title="Medical Note Explainer", page_icon="🏥")
st.title("🏥 Patient-Friendly Clinical Note Explainer")
st.markdown("""
This tool converts complex medical jargon into clear, simple language for patients and caregivers.
""")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter OpenAI API Key:", type="password")
    model_choice = st.selectbox("Select Model", ["gpt-4o-mini", "gpt-4o"])
    temp = st.slider("Tone Sensitivity (Temperature)", 0.0, 1.0, 0.3)

# --- Logic ---
if api_key:
    client = OpenAI(api_key=api_key)

    note_input = st.text_area("Paste the clinical note here:", height=300)

    if st.button("Explain Note"):
        if note_input:
            with st.spinner("Translating medical jargon..."):
                try:
                    response = client.chat.completions.create(
                        model=model_choice,
                        temperature=temp,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Explain the clinical note in a structured way for a patient:\n"
                                    "1. What happened\n"
                                    "2. Why it matters\n"
                                    "3. What treatment is being done\n"
                                    "4. What to expect next\n"
                                    "Keep language simple and reassuring."
                                )
                            },
                            {"role": "user", "content": note_input}
                        ]
                    )
                    
                    explanation = response.choices[0].message.content
                    st.subheader("Your Explanation")
                    st.write(explanation)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please paste a note first.")
else:
    st.info("Please enter your OpenAI API key in the sidebar to begin.")