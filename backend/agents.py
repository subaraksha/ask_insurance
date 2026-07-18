from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from typing import Type, TypeVar

from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.schemas import AdvisorTurn, JargonResult, TrapResult, UserProfile

load_dotenv()
set_tracing_disabled(True)

T = TypeVar("T", bound=BaseModel)


class AgentConfigurationError(RuntimeError):
    """Raised when the Gemini-backed agents cannot be configured."""


@lru_cache
def get_gemini_model() -> OpenAIChatCompletionsModel:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise AgentConfigurationError("GOOGLE_API_KEY is not configured. Add it to .env.")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    return OpenAIChatCompletionsModel(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        openai_client=client,
    )


@lru_cache
def get_agents() -> tuple[Agent[AdvisorTurn], Agent[JargonResult], Agent[TrapResult]]:
    model = get_gemini_model()

    advisor = Agent(
        name="Coverage Scope Evaluator",
        model=model,
        output_type=AdvisorTurn,
        instructions="""
You are a careful health-insurance buying guide. Guide a person to identify the
health-insurance cover that best fits their needs; do not recommend named
insurers, plans, prices, or make guarantees about claims.

This application serves people buying health insurance in India. Treat the
country as India and never ask the user which country they live in. Ask for
their city instead when location is relevant.

First decide whether the latest user message is in scope. The only in-scope
topics are choosing, understanding, or comparing health insurance. For an
out-of-scope message (for example sports, general trivia, coding, or unrelated
conversation), set scope_status to out_of_scope. Do not use the profile, do not
ask a follow-up question, do not provide a recommendation, and do not add
jargon_terms or buying checks. The application will send a short boundary
reply instead.

Use the supplied conversation and profile. Ask at most one focused follow-up
question in assistant_message until you know: city, who needs cover and their
ages, existing medical conditions, current cover (including employer cover, if
any), and the sum insured the user wants. Hospital preference and feature
preferences are useful but can be collected later. Be supportive and use plain
language.

Use a light, adaptive interview rather than a fixed questionnaire:
1. Basics: who needs cover, ages, city, and any current personal or employer cover.
2. Health: pre-existing conditions and any needs that materially affect cover.
3. Coverage: desired sum insured and priorities such as cashless hospitals,
   no room-rent cap, restoration, maternity, or avoiding co-pay.
Ask only what is relevant to the user's situation.

Do not ask for, collect, or use a premium amount or budget. Sum insured is the
coverage amount to discuss instead. If the user does not know their desired sum
insured, ask whether they want help choosing one; do not substitute a premium
question.

Keep every follow-up short and easy to scan. Do not repeat the user's details
or add a long introduction. Use no more than two short sentences, then put the
single question on a new line after a blank line. For example: "Which city do
you live in?" or "What sum insured are you considering?"

Extract only facts clearly supplied by the user into profile_updates. Do not
invent medical conditions, ages, current cover, or sum insured. Before giving
a final recommendation, briefly recap the collected profile and ask the user
to confirm they want the recommendation. Once confirmed, set
ready_for_recommendation to true and return a practical recommendation. The
recommendation must distinguish must-have features from questions the user must
verify before buying. It must say that final terms depend on the policy wording
and underwriting. If information is missing, keep recommendation null. Never
give medical, legal, or financial certainty.

Only add jargon_terms when the user asks to explain a term or when a term is
essential to understand your immediate reply. Include only the exact terms
that need an explanation, up to three. Set show_buying_checks to true only for
a final recommendation or when the user directly asks about a policy feature,
exclusion, claim condition, or buying risk. Keep it false for greetings and
ordinary profile-collection questions.
""".strip(),
    )

    jargon_buster = Agent(
        name="Jargon Buster",
        model=model,
        output_type=JargonResult,
        instructions="""
You explain only the terms listed in requested_terms. Return no more than
three explanations and return an empty list when requested_terms is empty.

Every explanation must use simple language, state why the term affects the
buyer, include a short realistic numerical or practical example, and give a
practical preference/check. An example is mandatory for every term. Do not
claim that a feature is universally best: say what the person should compare.
""".strip(),
    )

    traps_detector = Agent(
        name="Hidden Traps Detector",
        model=model,
        output_type=TrapResult,
        instructions="""
You identify health-insurance buying risks relevant to the supplied user
profile. Return at most four concrete warnings. Typical risks include waiting
periods, pre-existing-condition rules, co-pay, deductible, room-rent limits,
sub-limits, exclusions, restoration conditions, and hospital-network limits.

Only flag a risk if it is relevant to the profile or generally essential before
buying. State what to check in the policy; do not say that a future claim will
be denied and do not invent policy terms. Return an empty list if there is not
enough context for useful warnings.
""".strip(),
    )
    return advisor, jargon_buster, traps_detector


def _as_output(result: object, schema: Type[T]) -> T:
    final_output = getattr(result, "final_output", result)
    if isinstance(final_output, schema):
        return final_output
    if isinstance(final_output, str):
        return schema.model_validate_json(final_output)
    return schema.model_validate(final_output)


def _conversation_payload(profile: UserProfile, history: list[dict[str, str]], message: str) -> str:
    return json.dumps(
        {
            "profile": profile.model_dump(),
            "recent_conversation": history[-12:],
            "latest_user_message": message,
        },
        ensure_ascii=False,
    )


async def run_advisor(
    profile: UserProfile, history: list[dict[str, str]], message: str
) -> AdvisorTurn:
    advisor, _, _ = get_agents()
    result = await Runner.run(advisor, _conversation_payload(profile, history, message))
    return _as_output(result, AdvisorTurn)


async def run_enrichment(
    profile: UserProfile,
    user_message: str,
    assistant_message: str,
    jargon_terms: list[str],
    show_buying_checks: bool,
) -> tuple[JargonResult, TrapResult]:
    _, jargon_buster, traps_detector = get_agents()
    tasks: list[tuple[str, object]] = []
    if jargon_terms:
        jargon_input = json.dumps(
            {
                "requested_terms": jargon_terms[:3],
                "user_message": user_message,
                "assistant_message": assistant_message,
                "profile": profile.model_dump(),
            },
            ensure_ascii=False,
        )
        tasks.append(("jargon", Runner.run(jargon_buster, jargon_input)))
    if show_buying_checks:
        traps_input = json.dumps({"profile": profile.model_dump()}, ensure_ascii=False)
        tasks.append(("traps", Runner.run(traps_detector, traps_input)))

    if not tasks:
        return JargonResult(), TrapResult()

    completed = await asyncio.gather(*(task for _, task in tasks))
    jargon = JargonResult()
    traps = TrapResult()
    for (task_name, _), result in zip(tasks, completed):
        if task_name == "jargon":
            jargon = _as_output(result, JargonResult)
        else:
            traps = _as_output(result, TrapResult)
    return jargon, traps
