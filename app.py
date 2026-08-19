import streamlit as st

# Basic page configuration
st.set_page_config(
    page_title="Bharat Seva AI",
    page_icon="🇮🇳",
    layout="centered"
)

# ---------- Sidebar: User Profile ----------
with st.sidebar:
    st.header("Your Profile")
    st.caption("This information will later help personalize scheme recommendations.")

    age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)
    state = st.selectbox(
        "State",
        [
            "Andhra Pradesh", "Bihar", "Delhi", "Gujarat", "Karnataka",
            "Kerala", "Madhya Pradesh", "Maharashtra", "Punjab",
            "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh",
            "West Bengal", "Other"
        ]
    )
    occupation = st.text_input("Occupation", placeholder="e.g. Student, Farmer, Business Owner")
    income = st.text_input("Annual Household Income (₹)", placeholder="e.g. 250000")

# ---------- Main Page: Title & Description ----------
st.title("Bharat Seva AI 🇮🇳")
st.write(
    "Bharat Seva AI helps citizens of India discover and understand "
    "government schemes relevant to them — including benefits, eligibility "
    "criteria, required documents, and how to apply."
)

st.divider()

# ---------- Chat Section ----------
st.subheader("Ask Bharat Seva AI")
st.write("Type a question below about government schemes you're interested in.")

user_question = st.chat_input("Ask about a government scheme...")

# Display the user's question so the page reacts, even though there is no AI response yet
if user_question:
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        st.write(
            "This is a placeholder response. Answering with real scheme "
            "information will be added in a later step."
        )
