from __future__ import annotations

import html
import os
import uuid
from pathlib import Path

import requests
import streamlit as st

# API_BASE_URL = os.getenv("API_BASE_URL", "https://ask-insurance-7kgu.onrender.com")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
LOGO_PATH = Path(__file__).resolve().parents[1] / "ask_inurane_logo.png"

st.set_page_config(page_title="Ask Insurance", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.25rem; }
        .chat-row { display: flex; width: 100%; margin: 0.75rem 0; }
        .chat-row.user { justify-content: flex-end; }
        .chat-row.assistant { justify-content: flex-start; }
        .chat-bubble {
            max-width: 72%; padding: 0.8rem 1rem; border-radius: 1rem;
            line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere;
        }
        .chat-row.user .chat-bubble {
            background: #2563eb; color: #ffffff; border-bottom-right-radius: 0.25rem;
        }
        .chat-row.assistant .chat-bubble {
            background: #1f2937; color: #f9fafb; border-bottom-left-radius: 0.25rem;
        }
        .turn-details { max-width: 72%; margin: 0 0 1.3rem 0; }
        div[data-testid="stChatInput"] > div,
        div[data-testid="stChatInput"] > div:focus-within {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
_, logo_column, _ = st.columns([4, 3, 4])
with logo_column:
    st.image(str(LOGO_PATH), use_container_width=True)
st.caption("A guided health-insurance buying assistant for India — not a policy seller.")


def api_call(method: str, path: str, **kwargs: object) -> dict:
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=90, **kwargs)
    if response.ok:
        return response.json()
    try:
        detail = response.json().get("detail", "Something went wrong.")
    except ValueError:
        detail = response.text or "Something went wrong."
    raise RuntimeError(detail)


def initialise_session() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = st.query_params.get("user", str(uuid.uuid4()))
        session = api_call("POST", "/sessions", json={"user_id": st.session_state.user_id})
        st.session_state.messages = [
            {
                "role": item["role"],
                "content": item["content"],
                "jargon": item.get("jargon", []),
                "jargon_only": item.get("jargon_only", False),
                "warnings": item.get("warnings", []),
                "recommendation": item.get("recommendation"),
            }
            for item in session["messages"]
        ]
        st.session_state.profile = session["profile"]
        st.query_params["user"] = st.session_state.user_id


def render_message(message: dict) -> None:
    role = message["role"]
    if not (role == "assistant" and message.get("jargon_only")):
        safe_content = html.escape(message["content"])
        st.markdown(
            f'<div class="chat-row {role}"><div class="chat-bubble">{safe_content}</div></div>',
            unsafe_allow_html=True,
        )
    if role == "assistant":
        render_turn_details(message)


def render_turn_details(message: dict) -> None:
    jargon = message.get("jargon", [])
    recommendation = message.get("recommendation")
    warnings = message.get("warnings", []) if recommendation else []
    if not any((jargon, warnings, recommendation)):
        return

    with st.container():
        st.markdown('<div class="turn-details">', unsafe_allow_html=True)
        if jargon:
            st.markdown("**Jargon, made simple**")
            for item in jargon:
                with st.expander(item["term"]):
                    st.write(item["simple_meaning"])
                    st.warning(f"Why it matters: {item['why_it_matters']}")
                    st.info(f"Example: {item['example']}")
                    st.success(f"What to prefer: {item['what_to_prefer']}")
        if warnings:
            st.markdown("**Things to check before you buy**")
            for warning in warnings:
                st.warning(
                    f"**{warning['title']} ({warning['severity']})** — "
                    f"{warning['why_it_matters']}\n\nCheck: {warning['what_to_check']}"
                )
        if recommendation:
            st.markdown("**Your health-insurance buying checklist**")
            st.write(recommendation["summary"])
            st.write(f"**Suggested structure:** {recommendation['suggested_policy_structure']}")
            st.write(f"**Target coverage:** {recommendation['target_coverage']}")
            st.markdown("**Must-have features**")
            st.write("\n".join(f"- {item}" for item in recommendation["must_have_features"]))
            st.markdown("**Verify before buying**")
            st.write("\n".join(f"- {item}" for item in recommendation["avoid_or_verify"]))
            st.markdown("**Questions for the insurer**")
            st.write("\n".join(f"- {item}" for item in recommendation["questions_for_insurer"]))
        st.markdown("</div>", unsafe_allow_html=True)


try:
    initialise_session()
except (requests.RequestException, RuntimeError) as error:
    st.error(f"Cannot connect to the backend: {error}")
    st.stop()

with st.sidebar:
    st.subheader("Your buying profile")
    st.caption("It updates as you chat and is kept only while the API is running.")
    if st.session_state.get("profile"):
        profile = st.session_state.profile
        for label, key in (
            ("Country", "country"),
            ("City", "city"),
            ("Cover needed for", "coverage_for"),
            ("Current cover", "current_cover"),
            ("Target sum insured", "sum_insured"),
        ):
            if profile.get(key):
                st.write(f"**{label}:** {profile[key]}")
    else:
        st.write("Start by telling me who needs cover.")

    if st.button("Start a new conversation"):
        for key in ("user_id", "messages", "profile"):
            st.session_state.pop(key, None)
        st.query_params.clear()
        st.rerun()

for message in st.session_state.messages:
    render_message(message)

if not st.session_state.messages:
    render_message(
        {
            "role": "assistant",
            "content": (
                "Hi! I’ll help you understand what health insurance cover may suit you.\n\n"
                "Here’s how I can help:\n"
                "• Understand health-insurance basics\n"
                "• Clarify coverage details and policy terms\n"
                "• Help you choose an appropriate sum insured\n"
                "• Highlight important things to check before you buy\n\n"
                "To begin, do you need health insurance for yourself or for your family?"
            ),
        }
    )

if prompt := st.chat_input("Tell me about the health cover you need"):
    user_turn = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_turn)
    render_message(user_turn)

    with st.spinner("Reviewing your needs..."):
        try:
            result = api_call(
                "POST",
                f"/sessions/{st.session_state.user_id}/messages",
                json={"message": prompt},
            )
        except (requests.RequestException, RuntimeError) as error:
            st.error(str(error))
            st.stop()

    assistant_turn = {
        "role": "assistant",
        "content": result["assistant_message"],
        "jargon": result.get("jargon", []),
        "jargon_only": result.get("jargon_only", False),
        "warnings": result.get("warnings", []),
        "recommendation": result.get("recommendation"),
    }
    st.session_state.messages.append(assistant_turn)
    st.session_state.profile = result["profile"]
    render_message(assistant_turn)

st.divider()
st.caption(
    "Educational guidance only. Final coverage, exclusions, premiums, and claim decisions depend on "
    "the insurer's policy wording and underwriting."
)
