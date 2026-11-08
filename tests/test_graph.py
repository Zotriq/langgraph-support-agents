import sqlite3
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from support_agents.graph import MAX_REWRITES, build_graph
from support_agents.tools import Email, KnowledgeBase
from tests.fakes import ScriptedLLM

INBOX = Path(__file__).parents[1] / "examples" / "inbox"


def start(llm, email="duplicate_charge.txt", outbox=None, checkpointer=None, thread="t1"):
    app = build_graph(llm, outbox=outbox if outbox is not None else [], checkpointer=checkpointer or InMemorySaver())
    cfg = {"configurable": {"thread_id": thread}}
    app.invoke({"raw_email": (INBOX / email).read_text()}, cfg)
    return app, cfg


def pending(app, cfg):
    state = app.get_state(cfg)
    return [i.value for t in state.tasks for i in t.interrupts]


@pytest.mark.parametrize("category,email,expected_source", [
    ("billing", "duplicate_charge.txt", "customer record"),
    ("technical", "csv_import.txt", "csv-import"),
    ("sales", "enterprise_quote.txt", "pricing"),
])
def test_routes_to_specialist_and_pauses_for_approval(category, email, expected_source):
    llm = ScriptedLLM(category=category)
    app, cfg = start(llm, email)
    [request] = pending(app, cfg)
    assert request["category"] == category
    assert expected_source in request["sources"]
    assert request["draft"] == "Draft v1"
    assert app.get_state(cfg).next == ("human_approval",)


def test_billing_agent_sees_the_duplicate_invoice():
    llm = ScriptedLLM(category="billing")
    start(llm)
    [specialist_prompt] = [u for s, u in llm.calls if "billing specialist" in s]
    assert "INV-3389" in specialist_prompt and "duplicate charge" in specialist_prompt


def test_nothing_is_sent_without_approval_then_approve_sends():
    outbox = []
    app, cfg = start(ScriptedLLM(), outbox=outbox)
    assert outbox == []
    result = app.invoke(Command(resume={"action": "approve"}), cfg)
    assert result["outcome"] == "sent"
    assert outbox == [{"to": "sara@brightlane.co", "subject": "Re: Charged twice this month??", "body": "Draft v1"}]


def test_edit_sends_the_human_text():
    outbox = []
    app, cfg = start(ScriptedLLM(), outbox=outbox)
    app.invoke(Command(resume={"action": "edit", "text": "Refund issued."}), cfg)
    assert outbox[0]["body"] == "Refund issued."


def test_reject_sends_nothing():
    outbox = []
    app, cfg = start(ScriptedLLM(), outbox=outbox)
    result = app.invoke(Command(resume={"action": "reject"}), cfg)
    assert result["outcome"] == "discarded" and outbox == []


def test_reviewer_feedback_triggers_rewrite():
    llm = ScriptedLLM(reviews=[{"approved": False, "problems": ["promises a refund date not in notes"]}, {"approved": True, "problems": []}])
    app, cfg = start(llm)
    assert pending(app, cfg)[0]["draft"] == "Draft v2"
    assert "promises a refund date" in llm.prompts("You write")[1]


def test_rewrites_are_capped_and_open_points_shown_to_human():
    llm = ScriptedLLM(reviews=[])  # reviewer never approves
    app, cfg = start(llm)
    [request] = pending(app, cfg)
    assert len(llm.prompts("You write")) == MAX_REWRITES + 1
    assert request["open_review_points"] == ["still wrong"]


def test_unknown_category_escalates_without_drafting():
    llm = ScriptedLLM(category="legal-threat")
    app, cfg = start(llm)
    state = app.get_state(cfg)
    assert state.values["outcome"] == "escalated" and not state.next
    assert llm.prompts("You write") == []


def test_approval_survives_restart(tmp_path):
    db = tmp_path / "state.db"
    outbox = []
    start(ScriptedLLM(), outbox=outbox, checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)), thread="abc")

    # A new process: new graph object, same database.
    app2 = build_graph(ScriptedLLM(), outbox=outbox, checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    cfg = {"configurable": {"thread_id": "abc"}}
    assert pending(app2, cfg)[0]["draft"] == "Draft v1"
    app2.invoke(Command(resume={"action": "approve"}), cfg)
    assert outbox[0]["to"] == "sara@brightlane.co"


def test_email_parsing_and_kb_search():
    e = Email.parse((INBOX / "csv_import.txt").read_text())
    assert e.sender == "tom@harbortech.io" and e.subject == "weird characters after import"
    assert KnowledgeBase().search("import contacts excel characters")[0][0] == "csv-import"
