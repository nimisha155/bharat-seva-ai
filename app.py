import streamlit as st
from google import genai
from google.genai import types

from utils.retriever import retrieve_schemes, RetrieverError

st.set_page_config(
    page_title="Bharat Seva AI",
    page_icon="🇮🇳",
    layout="centered"
)

MODEL_NAME = "gemini-2.5-flash"
TOP_K_SCHEMES = 3

SYSTEM_INSTRUCTION = (
    "You are Bharat Seva AI, an assistant that helps people understand Indian "
    "government schemes and public services. "
    "You will be given a 'Retrieved scheme context' section for each question. "
    "This retrieved context is the primary source of truth for any scheme-specific "
    "factual claims you make, such as scheme names, eligibility criteria, benefits, "
    "required documents, deadlines, or application links. "
    "Do NOT invent or guess scheme-specific details. Do not rely on your general "
    "training knowledge to fill in facts that are missing from the retrieved context. "
    "If the retrieved context does not contain enough information to answer the "
    "question, clearly say that this could not be verified from the current "
    "knowledge base, rather than guessing. "
    "Clearly distinguish between verified information from the retrieved knowledge "
    "base and general guidance or explanation you are adding. "
    "Use the user's profile only to personalize the explanation when relevant. "
    "Always encourage the user to verify important details on the official "
    "government source, since scheme rules can change over time."
)

NO_CONTEXT_MESSAGE = (
    "I couldn't find a scheme in Bharat Seva AI's current knowledge base that is "
    "clearly relevant to your question. To help you better, please try asking a "
    "more specific question — for example, naming the type of support you're "
    "looking for (such as housing, healthcare, education, or a business loan) or "
    "the scheme name if you know it."
)


def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


with st.sidebar:
    st.header("Your Profile")
    st.caption("This information helps personalize scheme recommendations.")

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=25,
        step=1
    )

    state = st.selectbox(
        "State",
        [
            "Andhra Pradesh",
            "Bihar",
            "Delhi",
            "Gujarat",
            "Karnataka",
            "Kerala",
            "Madhya Pradesh",
            "Maharashtra",
            "Punjab",
            "Rajasthan",
            "Tamil Nadu",
            "Telangana",
            "Uttar Pradesh",
            "West Bengal",
            "Other"
        ]
    )

    occupation = st.text_input(
        "Occupation",
        placeholder="e.g. Student, Farmer, Business Owner"
    )

    income = st.text_input(
        "Annual Household Income (₹)",
        placeholder="e.g. 250000"
    )


st.title("Bharat Seva AI 🇮🇳")

st.write(
    "Bharat Seva AI helps citizens of India discover and understand "
    "government schemes relevant to them — including benefits, eligibility "
    "criteria, required documents, and how to apply."
)

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []


st.subheader("Ask Bharat Seva AI")

st.write(
    "Type a question below about government schemes you're interested in."
)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        for source in message.get("sources", []):
            st.markdown(
                f"**{source['name']}**  \n"
                f"Official source: {source['source_url']}  \n"
                f"Application / official portal: {source['official_url']}"
            )


def build_profile_context():
    return (
        f"User profile:\n"
        f"- Age: {age}\n"
        f"- State: {state}\n"
        f"- Occupation: {occupation if occupation else 'Not provided'}\n"
        f"- Annual household income: "
        f"{income if income else 'Not provided'}\n"
    )


def build_retrieval_query(user_question):
    profile_bits = []

    if occupation:
        profile_bits.append(f"occupation: {occupation}")

    if state and state != "Other":
        profile_bits.append(f"state: {state}")

    if income:
        profile_bits.append(f"annual household income: {income}")

    profile_bits.append(f"age: {age}")

    if profile_bits:
        return (
            f"{user_question} "
            f"(User details — {', '.join(profile_bits)})"
        )

    return user_question


def build_scheme_context(retrieved_schemes):
    blocks = []

    for item in retrieved_schemes:
        scheme = item["scheme"]

        block = (
            f"Scheme name: {scheme.get('name', 'Not specified')}\n"
            f"Ministry: {scheme.get('ministry', 'Not specified')}\n"
            f"Category: {scheme.get('category', 'Not specified')}\n"
            f"Description: {scheme.get('description', 'Not specified')}\n"
            f"Benefits: {scheme.get('benefits', 'Not specified')}\n"
            f"Eligibility: {scheme.get('eligibility', 'Not specified')}\n"
            f"Required documents: "
            f"{scheme.get('required_documents', 'Not specified')}\n"
            f"Application process: "
            f"{scheme.get('application_process', 'Not specified')}\n"
            f"Target beneficiaries: "
            f"{scheme.get('target_beneficiaries', 'Not specified')}\n"
            f"States: {scheme.get('states', 'Not specified')}\n"
            f"Official URL: "
            f"{scheme.get('official_url', 'Not specified')}\n"
            f"Source URL: "
            f"{scheme.get('source_url', 'Not specified')}\n"
            f"Last verified: "
            f"{scheme.get('last_verified', 'Not specified')}"
        )

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def get_gemini_response(client, user_question, scheme_context):
    profile_context = build_profile_context()

    contents = []

    # Exclude the current user message because it is added separately below.
    for msg in st.session_state.messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"

        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])]
            )
        )

    turn_text = (
        f"{profile_context}\n"
        f"Retrieved scheme context (source of truth for this answer):\n"
        f"{scheme_context}\n\n"
        f"Question: {user_question}"
    )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=turn_text)]
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


user_question = st.chat_input(
    "Ask about a government scheme..."
)

if user_question:

    if not user_question.strip():
        st.warning("Please enter a question.")

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):

            client = get_gemini_client()

            if client is None:

                error_text = (
                    "Bharat Seva AI is not fully configured yet. "
                    "The Gemini API key is missing. Please contact "
                    "the app administrator."
                )

                st.error(error_text)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_text
                    }
                )

            else:

                retrieved_schemes = None
                retrieval_failed = False

                try:

                    with st.spinner(
                        "Searching the scheme knowledge base..."
                    ):

                        retrieval_query = build_retrieval_query(
                            user_question
                        )

                        retrieved_schemes = retrieve_schemes(
                            retrieval_query,
                            top_k=TOP_K_SCHEMES
                        )

                except RetrieverError:

                    retrieval_failed = True

                except Exception:

                    retrieval_failed = True


                if retrieval_failed:

                    error_text = (
                        "Sorry, I couldn't search the scheme "
                        "knowledge base right now, so I can't "
                        "give a verified answer. Please try again "
                        "in a moment."
                    )

                    st.error(error_text)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_text
                        }
                    )


                elif not retrieved_schemes:

                    st.write(NO_CONTEXT_MESSAGE)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": NO_CONTEXT_MESSAGE
                        }
                    )


                else:

                    try:

                        scheme_context = build_scheme_context(
                            retrieved_schemes
                        )

                        with st.spinner("Thinking..."):

                            answer = get_gemini_response(
                                client,
                                user_question,
                                scheme_context
                            )

                        sources = [
                            {
                                "name": item["scheme"].get(
                                    "name",
                                    "Unknown scheme"
                                ),
                                "official_url": item["scheme"].get(
                                    "official_url",
                                    "Not available"
                                ),
                                "source_url": item["scheme"].get(
                                    "source_url",
                                    "Not available"
                                )
                            }
                            for item in retrieved_schemes
                        ]

                        st.write(answer)

                        for source in sources:

                            st.markdown(
                                f"**{source['name']}**  \n"
                                f"Official source: "
                                f"{source['source_url']}  \n"
                                f"Application / official portal: "
                                f"{source['official_url']}"
                            )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            }
                        )

                    except Exception:

                        error_text = (
                            "Sorry, something went wrong while "
                            "generating a response. Please try "
                            "again in a moment."
                        )

                        st.error(error_text)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": error_text
                            }
                        )
