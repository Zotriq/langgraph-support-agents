"""The agent team as a LangGraph state machine.

    START
      │
   supervisor ── classifies the email and picks a specialist
      │
      ├── billing    (customer + invoice lookup, refund policy)
      ├── technical  (knowledge-base search)
      ├── sales      (pricing, plan fit)
      └── escalate   (anything else → human, no AI reply)
      │
    writer ◄──────────┐  drafts the reply from the specialist's notes
      │               │
    reviewer ─────────┘  QA: rejects drafts that promise things the notes don't support (max 2 rewrites)
      │
    human_approval   ── graph pauses here (interrupt) until a person approves, edits or rejects
      │
    send / discard
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .llm import LLM, chat_json
from .tools import CustomerDB, Email, KnowledgeBase

MAX_REWRITES = 2
CATEGORIES = ("billing", "technical", "sales", "other")


class SupportState(TypedDict, total=False):
    raw_email: str
    email: dict[str, str]
    category: str
    urgency: str
    summary: str
    notes: str
    sources: list[str]
    draft: str
    review_feedback: list[str]
    rewrites: int
    decision: str
    final_reply: str
    outcome: str


def build_graph(llm: LLM, *, customers: CustomerDB | None = None, kb: KnowledgeBase | None = None, outbox=None, checkpointer=None):
    customers = customers or CustomerDB()
    kb = kb or KnowledgeBase()
    outbox = outbox if outbox is not None else []

    def supervisor(state: SupportState) -> Command[Literal["billing", "technical", "sales", "escalate"]]:
        email = Email.parse(state["raw_email"])
        result = chat_json(
            llm,
            "You triage customer support emails for a B2B SaaS company. Classify the email.\n"
            f'Return {{"category": one of {list(CATEGORIES)}, "urgency": "low"|"normal"|"high", "summary": one sentence}}.',
            f"Subject: {email.subject}\n\n{email.body}",
        )
        category = result.get("category", "other")
        if category not in CATEGORIES:
            category = "other"
        goto = {"billing": "billing", "technical": "technical", "sales": "sales"}.get(category, "escalate")
        return Command(
            goto=goto,
            update={
                "email": email.__dict__,
                "category": category,
                "urgency": result.get("urgency", "normal"),
                "summary": result.get("summary", ""),
                "rewrites": 0,
            },
        )

    def _specialist(role: str, state: SupportState, context: str, sources: list[str]) -> dict[str, Any]:
        notes = llm.chat(
            f"You are the {role} specialist. Using ONLY the context, write short internal notes for the reply writer: "
            "the facts that answer the customer, what action we will take, and anything we must not promise.",
            f"Customer email:\n{state['email']['body']}\n\nContext:\n{context}",
        )
        return {"notes": notes, "sources": sources}

    def billing(state: SupportState) -> dict[str, Any]:
        record = customers.lookup(state["email"]["sender"])
        policy = kb.articles.get("billing-policy", "")
        context = f"Customer record:\n{json.dumps(record, indent=2) if record else 'NOT FOUND'}\n\n{policy}"
        return _specialist("billing", state, context, ["customer record", "billing-policy"])

    def technical(state: SupportState) -> dict[str, Any]:
        hits = kb.search(f"{state['email']['subject']} {state['email']['body']}")
        context = "\n\n".join(text for _, text in hits) or "No matching help article."
        return _specialist("technical support", state, context, [name for name, _ in hits])

    def sales(state: SupportState) -> dict[str, Any]:
        record = customers.lookup(state["email"]["sender"])
        context = kb.articles.get("pricing", "") + (f"\n\nExisting customer:\n{json.dumps(record)}" if record else "\n\nNew prospect.")
        return _specialist("sales", state, context, ["pricing"] + (["customer record"] if record else []))

    def escalate(state: SupportState) -> dict[str, Any]:
        return {"outcome": "escalated", "notes": "Outside the agents' scope; routed to a human."}

    def writer(state: SupportState) -> dict[str, Any]:
        feedback = state.get("review_feedback") or []
        revision = f"\n\nA reviewer rejected the last draft. Fix these problems:\n- " + "\n- ".join(feedback) if feedback else ""
        draft = llm.chat(
            "You write friendly, concise customer support replies (under 150 words). "
            "Use only facts from the specialist notes. Sign off as 'The Support Team'.",
            f"Customer: {state['email']['sender']}\nSubject: {state['email']['subject']}\n\n"
            f"Email:\n{state['email']['body']}\n\nSpecialist notes:\n{state['notes']}{revision}",
        )
        return {"draft": draft}

    def reviewer(state: SupportState) -> Command[Literal["writer", "human_approval"]]:
        verdict = chat_json(
            llm,
            "You are QA for support replies. Reject the draft if it states facts not in the notes, promises "
            "anything the notes don't allow, is rude, or doesn't answer the question. "
            'Return {"approved": true|false, "problems": [short strings]}.',
            f"Customer email:\n{state['email']['body']}\n\nNotes:\n{state['notes']}\n\nDraft:\n{state['draft']}",
        )
        if verdict.get("approved") or state.get("rewrites", 0) >= MAX_REWRITES:
            return Command(goto="human_approval", update={"review_feedback": verdict.get("problems", [])})
        return Command(
            goto="writer",
            update={"review_feedback": verdict.get("problems", []), "rewrites": state.get("rewrites", 0) + 1},
        )

    def human_approval(state: SupportState) -> Command[Literal["send", "discard"]]:
        decision = interrupt(
            {
                "to": state["email"]["sender"],
                "subject": f"Re: {state['email']['subject']}",
                "category": state["category"],
                "urgency": state["urgency"],
                "draft": state["draft"],
                "open_review_points": state.get("review_feedback", []),
                "sources": state.get("sources", []),
            }
        )
        action = decision.get("action")
        if action == "approve":
            return Command(goto="send", update={"decision": "approved", "final_reply": state["draft"]})
        if action == "edit":
            return Command(goto="send", update={"decision": "edited", "final_reply": decision["text"]})
        return Command(goto="discard", update={"decision": "rejected"})

    def send(state: SupportState) -> dict[str, Any]:
        outbox.append({"to": state["email"]["sender"], "subject": f"Re: {state['email']['subject']}", "body": state["final_reply"]})
        return {"outcome": "sent"}

    def discard(state: SupportState) -> dict[str, Any]:
        return {"outcome": "discarded"}

    g = StateGraph(SupportState)
    for name, fn in [
        ("supervisor", supervisor), ("billing", billing), ("technical", technical), ("sales", sales),
        ("escalate", escalate), ("writer", writer), ("reviewer", reviewer),
        ("human_approval", human_approval), ("send", send), ("discard", discard),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "supervisor")
    for specialist in ("billing", "technical", "sales"):
        g.add_edge(specialist, "writer")
    g.add_edge("writer", "reviewer")
    g.add_edge("escalate", END)
    g.add_edge("send", END)
    g.add_edge("discard", END)
    return g.compile(checkpointer=checkpointer)
