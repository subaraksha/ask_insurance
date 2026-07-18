from __future__ import annotations

import html
import os
import uuid
from pathlib import Path

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "https://ask-insurance-7kgu.onrender.com")
# API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
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
        .product-card {
            border: 1px solid #dbe3ef; border-radius: 0.9rem; padding: 1rem;
            min-height: 11.5rem; background: #ffffff; margin-bottom: 0.45rem;
        }
        .product-card h4 { margin: 0 0 0.2rem 0; color: #111827; }
        .product-card p { color: #4b5563; margin: 0.4rem 0; }
        .product-card .insurer { color: #2563eb; font-weight: 600; }
        div[data-testid="stChatInput"] > div,
        div[data-testid="stChatInput"] > div:focus-within {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }
        /* Responsive Comparison Table Styling */
        .comparison-container {
            overflow-x: auto;
            width: 100%;
            margin: 1.5rem 0;
            border-radius: 0.5rem;
            border: 1px solid var(--border-color, rgba(128, 128, 128, 0.2));
            background-color: var(--background-color, transparent);
        }
        table.comparison-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
            text-align: left;
            color: var(--text-color, inherit);
        }
        table.comparison-table th {
            background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.15));
            font-weight: 700;
            padding: 12px 16px;
            border-bottom: 2px solid var(--border-color, rgba(128, 128, 128, 0.2));
            color: var(--primary-color, #2563eb);
            white-space: normal;
            vertical-align: middle;
        }
        table.comparison-table td {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color, rgba(128, 128, 128, 0.15));
            vertical-align: top;
            line-height: 1.5;
            white-space: normal;
            word-wrap: break-word;
        }
        table.comparison-table tr:hover {
            background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.05));
        }
        table.comparison-table td.criterion-cell {
            font-weight: 700;
            color: var(--text-color, inherit);
            opacity: 0.85;
            background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.1));
            width: 15%;
            min-width: 130px;
            white-space: normal;
        }
        table.comparison-table td ul {
            margin: 0;
            padding-left: 1.2rem;
        }
        table.comparison-table td li {
            margin-bottom: 4px;
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
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=300, **kwargs)
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
                "suggested_products": item.get("suggested_products", []),
            }
            for item in session["messages"]
        ]
        st.session_state.profile = session["profile"]
        st.query_params["user"] = st.session_state.user_id


def render_message(message: dict, index: int = 0) -> None:
    role = message["role"]
    if not (role == "assistant" and message.get("jargon_only")):
        safe_content = html.escape(message["content"])
        st.markdown(
            f'<div class="chat-row {role}"><div class="chat-bubble">{safe_content}</div></div>',
            unsafe_allow_html=True,
        )
    if role == "assistant":
        render_turn_details(message, index)


def render_turn_details(message: dict, index: int = 0) -> None:
    jargon = message.get("jargon", [])
    recommendation = message.get("recommendation")
    suggested_products = message.get("suggested_products", [])
    if not any((jargon, recommendation, suggested_products)):
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
        if suggested_products:
            st.markdown("**Suggested products**")
            st.caption(
                "These are retrieval matches to your request, not quotes or a recommendation to buy. "
                "Open the official wording and verify terms before deciding."
            )
            columns = st.columns(min(3, len(suggested_products)))
            for prod_idx, item in enumerate(suggested_products):
                st.session_state.product_candidates[item["product_id"]] = item
                with columns[prod_idx % len(columns)]:
                    st.markdown(
                        "<div class='product-card'>"
                        f"<h4>{html.escape(item['product'])}</h4>"
                        f"<p class='insurer'>{html.escape(item['insurance_company'])}</p>"
                        f"<p><strong>Designed for:</strong> {html.escape(item['primary_use_case'])}</p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    st.checkbox(
                        f"Select {item['product']}",
                        key=f"product-select-{item['product_id']}-{index}",
                    )
                    if st.button("View policy details", key=f"policy-details-{item['product_id']}-{index}", use_container_width=True):
                        with st.spinner(f"Reading {item['product']} policy wording..."):
                            try:
                                st.session_state.policy_insights[item["product_id"]] = api_call(
                                    "GET", f"/catalog/products/{item['product_id']}/insight"
                                )
                            except (requests.RequestException, RuntimeError) as error:
                                st.error(str(error))
                    st.link_button("Read official policy wording", item["pdf_source"], use_container_width=True)
                    if insight := st.session_state.policy_insights.get(item["product_id"]):
                        render_policy_insight(insight)
        st.markdown("</div>", unsafe_allow_html=True)


def render_policy_insight(insight: dict) -> None:
    """Display model output that is grounded in the selected product's stored wording."""
    with st.expander("Policy wording insights", expanded=True):
        if insight["source_status"] == "metadata_only":
            st.warning("Full wording was not available in the index. This view contains metadata only.")
        st.write(insight["overview"])
        for label, key in (
            ("Key covers", "key_covers"),
            ("Key exclusions", "key_exclusions"),
            ("Waiting periods", "waiting_periods"),
            ("Limits or cost sharing", "limits_or_cost_sharing"),
            ("Important checks", "important_checks"),
        ):
            if insight.get(key):
                st.markdown(f"**{label}**")
                st.markdown("\n".join(f"- {item}" for item in insight[key]))
        st.caption(insight["source_note"])


def render_comparison() -> None:
    candidates = st.session_state.get("product_candidates", {})
    selected = []
    for product_id, item in candidates.items():
        is_selected = False
        for k, v in st.session_state.items():
            if k.startswith(f"product-select-{product_id}-") and v:
                is_selected = True
                break
        if is_selected:
            selected.append(item)
    if not candidates:
        return
    st.caption(f"Select at least two products to compare ({len(selected)} selected).")
    if st.button(
        "Compare selected products",
        type="primary",
        disabled=len(selected) < 2,
        use_container_width=False,
    ):
        with st.spinner("Comparing policy wording excerpts..."):
            try:
                st.session_state.policy_comparison = api_call(
                    "POST",
                    "/catalog/compare",
                    json={"product_ids": [item["product_id"] for item in selected]},
                )
            except (requests.RequestException, RuntimeError) as error:
                st.error(str(error))

    comparison = st.session_state.get("policy_comparison")
    selected_ids = [item["product_id"] for item in selected]
    if comparison and comparison.get("product_ids") == selected_ids:
        st.subheader("Selected product comparison")
        
        # Build a beautiful, responsive, and fit-to-content HTML table
        html_table = "<div class='comparison-container'><table class='comparison-table'>"
        
        # Headers
        html_table += "<thead><tr><th>Criterion</th>"
        for label in comparison["product_labels"]:
            html_table += f"<th>{html.escape(label)}</th>"
        html_table += "</tr></thead><tbody>"
        
        # Rows
        for row in comparison["rows"]:
            html_table += "<tr>"
            html_table += f"<td class='criterion-cell'>{html.escape(row['criterion'])}</td>"
            for val in row["values"]:
                if isinstance(val, list):
                    bullet_list = "".join(f"<li>{html.escape(str(item))}</li>" for item in val)
                    html_table += f"<td><ul>{bullet_list}</ul></td>"
                else:
                    # Convert bullet points or lists if the text uses '*' or '-' prefixes, or has newlines
                    lines = str(val).split("\n")
                    if any(line.strip().startswith(("- ", "* ", "• ")) for line in lines if line.strip()):
                        list_items = []
                        for line in lines:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            if stripped.startswith("- "):
                                list_items.append(f"<li>{html.escape(stripped[2:])}</li>")
                            elif stripped.startswith("* "):
                                list_items.append(f"<li>{html.escape(stripped[2:])}</li>")
                            elif stripped.startswith("• "):
                                list_items.append(f"<li>{html.escape(stripped[2:])}</li>")
                            else:
                                list_items.append(f"<li>{html.escape(stripped)}</li>")
                        html_table += f"<td><ul>{''.join(list_items)}</ul></td>"
                    else:
                        escaped_val = html.escape(str(val)).replace("\n", "<br/>")
                        html_table += f"<td>{escaped_val}</td>"
            html_table += "</tr>"
            
        html_table += "</tbody></table></div>"
        
        st.markdown(html_table, unsafe_allow_html=True)
        for note in comparison.get("important_notes", []):
            st.info(note)
        st.caption(comparison["source_note"])


try:
    initialise_session()
    st.session_state.setdefault("product_candidates", {})
    st.session_state.setdefault("policy_insights", {})
    st.session_state.setdefault("policy_comparison", None)
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
            ("Age(s)", "ages"),
            ("Current cover", "current_cover"),
            ("Budget", "budget"),
            ("Target sum insured", "sum_insured"),
        ):
            if profile.get(key):
                st.write(f"**{label}:** {profile[key]}")
    else:
        st.write("Start by telling me who needs cover.")

    if st.button("Start a new conversation"):
        for key in ("user_id", "messages", "profile", "product_candidates", "policy_insights", "policy_comparison"):
            st.session_state.pop(key, None)
        for key in list(st.session_state):
            if key.startswith("product-select-"):
                st.session_state.pop(key)
        st.query_params.clear()
        st.rerun()

for idx, message in enumerate(st.session_state.messages):
    render_message(message, idx)

render_comparison()

if not st.session_state.messages:
    render_message(
        {
            "role": "assistant",
            "content": (
                "Hi! I’ll help you understand what health insurance cover may suit you.\n\n"
                "Here’s how I can help:\n"
                "• Understand health-insurance basics\n"
                "• Clarify coverage details and policy terms\n"
                "• Help you choose an appropriate sum insured\n\n"
                "To begin, do you need health insurance for yourself or for your family?"
            ),
        }
    )

if prompt := st.chat_input("Tell me about the health cover you need"):
    user_turn = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_turn)
    render_message(user_turn, len(st.session_state.messages) - 1)

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
        "suggested_products": result.get("suggested_products", []),
    }
    st.session_state.messages.append(assistant_turn)
    st.session_state.profile = result["profile"]
    render_message(assistant_turn, len(st.session_state.messages) - 1)

st.divider()
st.caption(
    "Educational guidance only. Final coverage, exclusions, premiums, and claim decisions depend on "
    "the insurer's policy wording and underwriting."
)
