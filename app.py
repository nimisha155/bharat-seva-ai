import streamlit as st
from google import genai
from google.genai import types

# Basic page configuration
st.set_page_config(
    page_title="Bharat Seva AI",
    page_icon="🇮🇳",
    layout="centered"
)

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "You are Bharat Seva AI, an assistant that helps people understand Indian "
    "government schemes and public services. Give clear, concise and useful "
    "information. Do not invent scheme names, eligibility criteria, benefits, "
    "deadlines or application links. If you are uncertain or do not have "
    "verified information, clearly say so. Explain that users should verify "
    "important details on the official government website. Use the user's "
    "profile only to personalize the response when relevant."
)


# ---------- Gemini client setup ----------
def get_gemini_client():
    """
    Creates and returns a Gemini client using the API key stored in
    Streamlit Secrets. Returns None if the key is missing so the app
    can show a friendly error instead of crashing.
    """
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


# ---------- Sidebar: User Profile ----------
with st.sidebar:
    st.header("Your Profile")
    st.caption("This information helps personalize scheme recommendations.")

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

# ---------- Conversation history ----------
# Stored in session_state so it persists across reruns during this session
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

# ---------- Chat Section ----------
st.subheader("Ask Bharat Seva AI")
st.write("Type a question below about government schemes you're interested in.")

# Re-display past messages so history stays visible after each rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def build_profile_context():
    """Formats the sidebar profile fields into a short text block for the model."""
    return (
        f"User profile:\n"
        f"- Age: {age}\n"
        f"- State: {state}\n"
        f"- Occupation: {occupation if occupation else 'Not provided'}\n"
        f"- Annual household income: {income if income else 'Not provided'}\n"
    )


def get_gemini_response(client, user_question):
    """
    Sends the user's question, along with profile context and past
    conversation, to Gemini and returns the text response.
    Raises exceptions on API failure so the caller can handle them.
    """
    profile_context = build_profile_context()

    # Build conversation history in the format expected by the SDK
    contents = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )

    # Add the new user question, prefixed with profile context for personalization
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=f"{profile_context}\nQuestion: {user_question}")]
        )
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION
        )
    )

    return response.text


# ---------- Handle new user input ----------
user_question = st.chat_input("Ask about a government scheme...")

if user_question:
    # Guard against empty/whitespace-only input
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        # Show and store the user's message
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        # Get and show the assistant's response
        with st.chat_message("assistant"):
            client = get_gemini_client()

            if client is None:
                error_text = (
                    "Bharat Seva AI is not fully configured yet. "
                    "The Gemini API key is missing. Please contact the app "
                    "administrator."
                )
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})
            else:
                try:
                    with st.spinner("Thinking..."):
                        answer = get_gemini_response(client, user_question)
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception:
                    error_text = (
                        "Sorry, something went wrong while getting a response. "
                        "Please try again in a moment."
                    )
                    st.error(error_text)
                    st.session_state.messages.append({"role": "assistant", "content": error_text})
