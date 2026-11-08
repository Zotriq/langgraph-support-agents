"""Run the agent team on an email, then approve / edit / reject later (state is saved in SQLite).

    support-agents run examples/inbox/duplicate_charge.txt
    support-agents pending
    support-agents approve <thread-id>
    support-agents edit <thread-id> reply.txt
    support-agents reject <thread-id>
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .graph import build_graph
from .llm import OpenAICompatibleLLM

OUTBOX = Path("outbox.jsonl")


class FileOutbox(list):
    """Appends sent replies to outbox.jsonl. Replace with Gmail/Zendesk/Help Scout in production."""

    def append(self, item):
        super().append(item)
        with OUTBOX.open("a") as fh:
            fh.write(json.dumps(item) + "\n")


def _app(db_path: str):
    load_dotenv()
    llm = OpenAICompatibleLLM(
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("LLM_BASE_URL"),
    )
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return build_graph(llm, outbox=FileOutbox(), checkpointer=SqliteSaver(conn)), conn


def _show(app, thread_id: str) -> None:
    state = app.get_state({"configurable": {"thread_id": thread_id}})
    pending = [i.value for t in state.tasks for i in t.interrupts]
    if pending:
        p = pending[0]
        print(f"\n⏸  Waiting for approval  [thread {thread_id}]")
        print(f"   To: {p['to']} | {p['subject']} | {p['category']}, {p['urgency']} urgency | sources: {', '.join(p['sources'])}")
        if p["open_review_points"]:
            print(f"   Reviewer notes: {'; '.join(p['open_review_points'])}")
        print("\n" + p["draft"] + "\n")
        print(f"→ support-agents approve {thread_id}   |   edit {thread_id} <file>   |   reject {thread_id}")
    else:
        v = state.values
        print(f"[thread {thread_id}] outcome: {v.get('outcome')} ({v.get('category')})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="support-agents")
    parser.add_argument("--db", default="support_agents.db")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("email_file", type=Path)
    sub.add_parser("pending")
    for name in ("approve", "reject"):
        sub.add_parser(name).add_argument("thread_id")
    edit = sub.add_parser("edit")
    edit.add_argument("thread_id")
    edit.add_argument("reply_file", type=Path)
    args = parser.parse_args()

    app, conn = _app(args.db)
    if args.cmd == "run":
        thread_id = uuid.uuid4().hex[:8]
        app.invoke({"raw_email": args.email_file.read_text()}, {"configurable": {"thread_id": thread_id}})
        _show(app, thread_id)
    elif args.cmd == "pending":
        threads = {row[0] for row in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")}
        for tid in sorted(threads):
            state = app.get_state({"configurable": {"thread_id": tid}})
            if state.next:
                _show(app, tid)
    else:
        resume = {"action": "approve"} if args.cmd == "approve" else {"action": "reject"}
        if args.cmd == "edit":
            resume = {"action": "edit", "text": args.reply_file.read_text()}
        config = {"configurable": {"thread_id": args.thread_id}}
        if not app.get_state(config).next:
            raise SystemExit(f"Thread {args.thread_id} is not waiting for approval.")
        app.invoke(Command(resume=resume), config)
        _show(app, args.thread_id)


if __name__ == "__main__":
    main()
