import streamlit as st
from openai import OpenAI

# --- UI Configuration ---
st.set_page_config(page_title="Medical Note Assistant", page_icon="🏥", layout="wide")
st.title("🏥 Clinical Note Assistant: From Jargon to Clarity")
st.markdown("""
This tool uses Generative AI (GPT) to translate complex medical notes into patient-friendly language. 
After the initial explanation, you can ask follow-up questions to understand the diagnosis better.
""")

# --- Data: Pre-loaded Demos ---
demos = {
    "Select a demo...": "",
    "Pediatric Ear Infection": """Patient is a 2-year-old male with a history of recurrent acute otitis media (6 times in 12 months), presenting with persistent middle ear effusion despite multiple courses of oral antibiotics including amoxicillin-clavulanate and cefdinir. Caregiver reports increased irritability, nighttime awakenings, and delayed speech development.

Otoscopic examination reveals bilateral tympanic membrane retraction with decreased mobility on pneumatic otoscopy and presence of serous effusion. No signs of acute perforation.

Audiometric screening suggests mild conductive hearing loss. Given the chronicity of effusion (>3 months) and impact on auditory development, patient meets criteria for myringotomy with tympanostomy tube placement.

Procedure, risks, and benefits were discussed with caregivers. Plan to proceed with bilateral ear tube insertion under general anesthesia.""",
    
    "Postpartum Hemorrhage": """Patient is a 32-year-old G2P2 female status post spontaneous vaginal delivery complicated by severe postpartum hemorrhage secondary to uterine atony. Estimated blood loss was approximately 1800 mL. Initial management included uterine massage, administration of oxytocin, methylergonovine, and carboprost, with partial response.

Due to ongoing bleeding and hemodynamic instability (tachycardia, hypotension), patient required activation of massive transfusion protocol, receiving 4 units packed red blood cells, 2 units fresh frozen plasma, and 1 unit platelets.

Patient was transferred to ICU for close monitoring. Hemoglobin stabilized post-transfusion. Uterus remains firm with minimal ongoing bleeding. Vital signs improving but remain labile.

Plan includes continued hemodynamic monitoring, serial hemoglobin checks, and evaluation for potential delayed complications including infection or coagulopathy."""
}

# --- Sidebar: Settings & Key ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter OpenAI API Key:", type="password")
    
    st.divider()
    
    # Tone Sensitivity (Temperature) Explanation
    st.markdown("**Tone Sensitivity (Temperature)**")
    temp = st.slider("Level:", 0.0, 1.0, 0.3, help="0.0 is very factual and consistent. 1.0 is more creative and varied.")
    st.caption("Low = Medical Accuracy | High = Conversational")

    st.divider()
    
    # Demo Selection
    selected_demo_name = st.selectbox("Quick Load Demo:", list(demos.keys()))
    default_text = demos[selected_demo_name]

# --- Main Interface ---
if api_key:
    client = OpenAI(api_key=api_key)

    # Input area: Value is tied to the dropdown but can be edited manually
    note_input = st.text_area("Clinical Note (Paste your own or use a demo):", value=default_text, height=250)

    # State management for Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Generate Explanation", use_container_width=True):
            if note_input:
                # Reset chat and start with the medical explanation prompt
                st.session_state.messages = [
                    {"role": "system", "content": "Explain the clinical note in a structured way: 1. What happened, 2. Why it matters, 3. Treatment, 4. Next steps. Use simple, reassuring, jargon-free language."},
                    {"role": "user", "content": f"Please explain this note:\n\n{note_input}"}
                ]
                
                with st.spinner("Analyzing..."):
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=st.session_state.messages,
                        temperature=temp
                    )
                    res_content = response.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": res_content})
            else:
                st.warning("Please provide a note.")

    # --- Chat Display ---
    st.divider()
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # --- Follow-up Q&A ---
    if prompt := st.chat_input("Ask a follow-up question about this case..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages,
                    temperature=temp
                )
                res_text = response.choices[0].message.content
                st.write(res_text)
                st.session_state.messages.append({"role": "assistant", "content": res_text})

else:
    st.info("👋 Please enter your OpenAI API key in the sidebar to unlock the medical assistant.")