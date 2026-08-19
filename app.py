import streamlit as st
from google import genai
from google.genai import types

from utils.retriever import retrieve_schemes, RetrieverError
from utils.eligibility import evaluate_eligibility, get_eligibility_summary


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Bharat Seva AI",
    page_icon="🇮🇳",
    layout="centered"
)


MODEL_NAME = "gemini-2.5-flash"
TOP_K_SCHEMES = 3


# =========================================================
# SPECIFIC SCHEME KEYWORDS
# =========================================================

SPECIFIC_SCHEME_KEYWORDS = {
    "pm kisan": "pm-kisan",
    "pm-kisan": "pm-kisan",
    "kisan samman": "pm-kisan",

    "pmay": "pmay",
    "pmay urban": "pmay",

    "ayushman bharat": "ab-pmjay",
    "pm jay": "ab-pmjay",
    "pmjay": "ab-pmjay",

    "ujjwala": "pmuy",
    "pmuy": "pmuy",

    "jan dhan": "pmjdy",
    "pmjdy": "pmjdy",

    "mudra": "pmmy",
    "pmmy": "pmmy",

    "pmkvy": "pmkvy",
    "kaushal vikas": "pmkvy",

    "central sector scholarship": "csss",
    "college scholarship": "csss",

    "sukanya": "ssy",
    "sukanya samriddhi": "ssy",

    "svanidhi": "pm-svanidhi",
    "street vendor": "pm-svanidhi",

    "stand up india": "stand-up-india",
    "stand-up india": "stand-up-india",

    "adip": "adip"
}


# =========================================================
# SYSTEM INSTRUCTION
# =========================================================

SYSTEM_INSTRUCTION = (
    "You are Bharat Seva AI, an assistant that helps people understand Indian "
    "government schemes and public services. "
    "You will be given retrieved scheme information and, when relevant, "
    "eligibility evaluation results. "
    "The retrieved scheme context is the primary source of truth for "
    "scheme-specific factual claims. "
    "Do not invent or guess scheme names, eligibility criteria, benefits, "
    "required documents, deadlines, or application links. "
    "Do not override deterministic eligibility results provided by the "
    "eligibility engine. "
    "If eligibility is marked potentially eligible, explain that additional "
    "information is required rather than claiming the user is eligible. "
    "If eligibility is marked not eligible, clearly explain the known failed "
    "condition without being unnecessarily discouraging. "
    "Use the user's profile to personalize the explanation when relevant. "
    "Encourage users to verify important details on the official government "
    "source because scheme rules can change."
)


NO_CONTEXT_MESSAGE = (
    "I couldn't find a sufficiently relevant scheme in Bharat Seva AI's "
    "current knowledge base. Try asking about a specific area such as "
    "education, healthcare, housing, farming, employment, or business."
)


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():

    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


# =========================================================
# SIDEBAR - USER PROFILE
# =========================================================

with st.sidebar:

    st.header("Your Profile")

    st.caption(
        "These details help Bharat Seva AI personalize "
        "scheme recommendations and eligibility checks."
    )

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

    income_text = st.text_input(
        "Annual Household Income (₹)",
        placeholder="e.g. 250000"
    )

    st.divider()

    st.subheader("Eligibility Details")

    gender = st.selectbox(
        "Gender",
        [
            "Prefer not to say",
            "Female",
            "Male",
            "Other"
        ]
    )

    has_landholding = st.selectbox(
        "Do you have agricultural land?",
        [
            "Not provided",
            "Yes",
            "No"
        ]
    )

    has_disability = st.selectbox(
        "Do you have a disability?",
        [
            "Not provided",
            "Yes",
            "No"
        ]
    )

    disability_percentage = st.number_input(
        "Disability percentage",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Enter 0 if disability information is not applicable."
    )

    has_existing_business = st.selectbox(
        "Do you have an existing business?",
        [
            "Not provided",
            "Yes",
            "No"
        ]
    )

    owns_house = st.selectbox(
        "Do you currently own a house?",
        [
            "Not provided",
            "Yes",
            "No"
        ]
    )


# =========================================================
# MAIN PAGE
# =========================================================

st.title("Bharat Seva AI 🇮🇳")

st.write(
    "Discover Indian government schemes relevant to you, "
    "understand their benefits and eligibility, and find official "
    "application sources."
)

st.divider()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# DISPLAY PREVIOUS MESSAGES
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        for source in message.get("sources", []):

            st.markdown(
                f"**{source['name']}**  \n"
                f"Official source: {source['source_url']}  \n"
                f"Application / official portal: {source['official_url']}"
            )


# =========================================================
# PROFILE HELPERS
# =========================================================

def parse_income():

    if not income_text.strip():
        return None

    try:

        value = float(
            income_text
            .replace(",", "")
            .replace("₹", "")
            .strip()
        )

        if value < 0:
            return None

        return value

    except ValueError:

        return None


def build_profile_context():

    return (
        f"User profile:\n"
        f"- Age: {age}\n"
        f"- State: {state}\n"
        f"- Occupation: "
        f"{occupation if occupation else 'Not provided'}\n"
        f"- Annual household income: "
        f"{income_text if income_text else 'Not provided'}\n"
        f"- Gender: "
        f"{gender if gender != 'Prefer not to say' else 'Not provided'}\n"
    )


def build_user_profile():

    income = parse_income()

    profile = {
        "age": age,
        "state": state,
        "occupation": occupation if occupation else None,
        "annual_income": income,
        "gender": (
            gender.lower()
            if gender != "Prefer not to say"
            else None
        ),
        "has_landholding": (
            True
            if has_landholding == "Yes"
            else False
            if has_landholding == "No"
            else None
        ),
        "has_disability": (
            True
            if has_disability == "Yes"
            else False
            if has_disability == "No"
            else None
        ),
        "disability_percentage": (
            disability_percentage
            if has_disability == "Yes"
            else None
        ),
        "has_existing_business": (
            True
            if has_existing_business == "Yes"
            else False
            if has_existing_business == "No"
            else None
        ),
        "owns_house": (
            True
            if owns_house == "Yes"
            else False
            if owns_house == "No"
            else None
        )
    }

    return profile


def build_retrieval_query(user_question):

    profile_bits = []

    if occupation:

        profile_bits.append(
            f"occupation: {occupation}"
        )

    if state and state != "Other":

        profile_bits.append(
            f"state: {state}"
        )

    if income_text:

        profile_bits.append(
            f"annual household income: {income_text}"
        )

    profile_bits.append(
        f"age: {age}"
    )

    if gender != "Prefer not to say":

        profile_bits.append(
            f"gender: {gender}"
        )

    if profile_bits:

        return (
            f"{user_question} "
            f"(User details — {', '.join(profile_bits)})"
        )

    return user_question


# =========================================================
# SCHEME CONTEXT
# =========================================================

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


# =========================================================
# ELIGIBILITY CONTEXT
# =========================================================

def build_eligibility_context(results):

    blocks = []

    for result in results:

        scheme = result["scheme"]
        evaluation = result["evaluation"]

        block = (
            f"Scheme: {scheme.get('name', 'Unknown')}\n"
            f"Eligibility status: {evaluation['status']}\n"
            f"Summary: {get_eligibility_summary(evaluation)}\n"
            f"Passed conditions: "
            f"{'; '.join(evaluation['passed']) if evaluation['passed'] else 'None'}\n"
            f"Failed conditions: "
            f"{'; '.join(evaluation['failed']) if evaluation['failed'] else 'None'}\n"
            f"Unknown conditions: "
            f"{'; '.join(evaluation['unknown']) if evaluation['unknown'] else 'None'}\n"
            f"Missing information: "
            f"{', '.join(evaluation['missing_information']) if evaluation['missing_information'] else 'None'}"
        )

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


# =========================================================
# GEMINI RESPONSE
# =========================================================

def get_gemini_response(
    client,
    user_question,
    scheme_context,
    eligibility_context
):

    profile_context = build_profile_context()

    contents = []

    for msg in st.session_state.messages[:-1]:

        role = (
            "user"
            if msg["role"] == "user"
            else "model"
        )

        contents.append(
            types.Content(
                role=role,
                parts=[
                    types.Part(
                        text=msg["content"]
                    )
                ]
            )
        )

    turn_text = (
        f"{profile_context}\n\n"
        f"Retrieved scheme context:\n"
        f"{scheme_context}\n\n"
        f"Deterministic eligibility evaluation:\n"
        f"{eligibility_context}\n\n"
        f"Question: {user_question}"
    )

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=turn_text
                )
            ]
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


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Ask about a government scheme..."
)


if user_question:

    if not user_question.strip():

        st.warning(
            "Please enter a question."
        )

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
                    "The Gemini API key is missing."
                )

                st.error(error_text)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_text
                    }
                )

            else:

                # =================================================
                # RETRIEVAL
                # =================================================

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

                    retrieved_schemes = None

                    st.error(
                        "I couldn't search the scheme knowledge "
                        "base right now. Please try again."
                    )

                except Exception:

                    retrieved_schemes = None

                    st.error(
                        "Something went wrong while searching "
                        "the scheme knowledge base."
                    )


                # =================================================
                # SMART SPECIFIC-SCHEME PRIORITIZATION
                # =================================================

                if retrieved_schemes:

                    question_lower = user_question.lower()

                    specific_scheme_id = None

                    for keyword, scheme_id in (
                        SPECIFIC_SCHEME_KEYWORDS.items()
                    ):

                        if keyword in question_lower:

                            specific_scheme_id = scheme_id
                            break

                    if specific_scheme_id:

                        matching_scheme = None
                        other_schemes = []

                        for item in retrieved_schemes:

                            if (
                                item["scheme"].get("scheme_id")
                                == specific_scheme_id
                            ):

                                matching_scheme = item

                            else:

                                other_schemes.append(item)

                        if matching_scheme:

                            retrieved_schemes = [
                                matching_scheme
                            ] + other_schemes


                # =================================================
                # NO RESULTS
                # =================================================

                if retrieved_schemes is None:

                    error_text = (
                        "I couldn't retrieve verified scheme "
                        "information right now."
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_text
                        }
                    )

                elif not retrieved_schemes:

                    st.write(
                        NO_CONTEXT_MESSAGE
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": NO_CONTEXT_MESSAGE
                        }
                    )

                else:

                    # =================================================
                    # ELIGIBILITY EVALUATION
                    # =================================================

                    user_profile = build_user_profile()

                    eligibility_results = []

                    for item in retrieved_schemes:

                        evaluation = evaluate_eligibility(
                            user_profile,
                            item["scheme"]
                        )

                        eligibility_results.append(
                            {
                                "scheme": item["scheme"],
                                "similarity": item["similarity"],
                                "evaluation": evaluation
                            }
                        )


                    # =================================================
                    # ELIGIBILITY OVERVIEW
                    # =================================================

                    st.subheader(
                        "Eligibility overview"
                    )

                    for result in eligibility_results:

                        scheme = result["scheme"]
                        evaluation = result["evaluation"]

                        status = evaluation["status"]

                        if status == "likely_eligible":

                            icon = "🟢"

                        elif status == "potentially_eligible":

                            icon = "🟡"

                        else:

                            icon = "🔴"

                        st.markdown(
                            f"### {icon} {scheme.get('name', 'Scheme')}"
                        )

                        st.write(
                            get_eligibility_summary(
                                evaluation
                            )
                        )

                        if evaluation["failed"]:

                            st.caption(
                                "Known issue: "
                                + " ".join(
                                    evaluation["failed"]
                                )
                            )

                        if evaluation["missing_information"]:

                            st.caption(
                                "Information needed: "
                                + ", ".join(
                                    evaluation[
                                        "missing_information"
                                    ]
                                )
                            )


                    # =================================================
                    # GEMINI GENERATION
                    # =================================================

                    try:

                        scheme_context = build_scheme_context(
                            retrieved_schemes
                        )

                        eligibility_context = (
                            build_eligibility_context(
                                eligibility_results
                            )
                        )

                        with st.spinner(
                            "Preparing your personalized answer..."
                        ):

                            answer = get_gemini_response(
                                client,
                                user_question,
                                scheme_context,
                                eligibility_context
                            )


                        # =================================================
                        # SOURCES
                        # =================================================

                        sources = []

                        for item in retrieved_schemes:

                            scheme = item["scheme"]

                            sources.append(
                                {
                                    "name": scheme.get(
                                        "name",
                                        "Unknown scheme"
                                    ),
                                    "official_url": scheme.get(
                                        "official_url",
                                        "Not available"
                                    ),
                                    "source_url": scheme.get(
                                        "source_url",
                                        "Not available"
                                    )
                                }
                            )


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
                            "generating the response. Please try again."
                        )

                        st.error(error_text)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": error_text
                            }
                        )
