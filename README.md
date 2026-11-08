# LangGraph Support Agents

![tests](https://github.com/Zotriq/langgraph-support-agents/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-1C3C3C?style=flat-square&logo=langgraph&logoColor=white)
![Human in the loop](https://img.shields.io/badge/human--in--the--loop-approval-orange?style=flat-square)

A **multi-agent customer support team** built on LangGraph. A supervisor reads each incoming email and hands
it to the right specialist. A writer drafts the reply, and a QA agent sends weak drafts back for a rewrite.
**Nothing is sent until a human approves it.**

The approval step is durable: the graph pauses, its state is saved to SQLite, and a person can approve, edit or
reject the draft minutes or days later, from a different process or server.

## The agent team

```
                     ┌──────────────┐
  incoming email ──► │  supervisor  │  classifies: billing / technical / sales / other + urgency
                     └──────┬───────┘
          ┌─────────────────┼──────────────────┬───────────────┐
          ▼                 ▼                  ▼               ▼
   ┌────────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────┐
   │  billing   │   │  technical   │   │   sales    │   │  escalate  │ → human, no AI reply
   │ CRM lookup │   │  KB search   │   │  pricing   │   └────────────┘
   │ + policy   │   │              │   │ + account  │
   └─────┬──────┘   └──────┬───────┘   └─────┬──────┘
         └─────────────────┼─────────────────┘
                           ▼
                    ┌────────────┐
                    │   writer   │ ◄─────────┐
                    └─────┬──────┘           │ rejected (max 2 rewrites)
                          ▼                  │
                    ┌────────────┐           │
                    │  reviewer  │ ──────────┘  checks the draft only uses facts from the notes
                    └─────┬──────┘
                          ▼
                 ┌─────────────────┐
                 │ human_approval  │  ⏸ interrupt(): saved to SQLite, waits for a person
                 └───┬─────────┬───┘
            approve / edit   reject
                     ▼         ▼
                   send     discard
```

| Agent | What it does |
|---|---|
| **Supervisor** | Classifies the email and its urgency, then routes it with LangGraph `Command(goto=...)` |
| **Billing** | Looks up the customer and invoices, applies the refund policy |
| **Technical** | Searches the help-center articles for the answer |
| **Sales** | Uses pricing and the customer's current plan to answer quotes and upgrades |
| **Writer** | Drafts a short reply using only the specialist's notes |
| **Reviewer (QA)** | Rejects drafts that invent facts or promise things the notes don't allow, with reasons the writer must fix |
| **Human approval** | `interrupt()` pauses the graph with the draft, sources and any open review points |

## Try it

```bash
git clone https://github.com/Zotriq/langgraph-support-agents.git
cd langgraph-support-agents
pip install -e .
cp .env.example .env          # any OpenAI-compatible key: OpenAI, Groq, OpenRouter, Ollama

support-agents run examples/inbox/duplicate_charge.txt
```

```text
⏸  Waiting for approval  [thread 9a68a1d1]
   To: sara@brightlane.co | Re: Charged twice this month?? | billing, high urgency | sources: customer record, billing-policy

<the drafted reply appears here>

→ support-agents approve 9a68a1d1   |   edit 9a68a1d1 <file>   |   reject 9a68a1d1
```

Later, from any terminal:

```bash
support-agents pending                    # everything waiting for a human
support-agents approve 9a68a1d1           # send as drafted
support-agents edit 9a68a1d1 reply.txt    # send your edited version
support-agents reject 9a68a1d1            # send nothing
```

Approved replies go to `outbox.jsonl`. In production, swap that for Gmail, Zendesk, Help Scout or Intercom in
`send()`, and trigger `run` from a webhook or an n8n workflow.

### Sample emails

| File | Routed to | What the specialist finds |
|---|---|---|
| `duplicate_charge.txt` | billing | Invoice `INV-3389` is a duplicate charge from a billing retry; policy allows a full refund |
| `csv_import.txt` | technical | The "Ã©" characters are a UTF-8 encoding problem; fix is "Save As → CSV UTF-8" |
| `enterprise_quote.txt` | sales | 120 seats with SSO + audit logs needs the Enterprise plan; offer a sales call |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

12 tests with a scripted LLM, so they're fast and deterministic. They cover routing for every category, the billing
agent seeing the right invoice, the reviewer → writer rewrite loop and its cap, escalation, approve / edit /
reject, the guarantee that nothing is sent before approval, and **resuming an approval from a new process via the
SQLite checkpoint**.

## Project structure

```
support_agents/
  graph.py    the LangGraph state machine (agents, routing, review loop, interrupt)
  tools.py    email parsing, customer/billing lookup, knowledge-base search
  llm.py      OpenAI-compatible client + JSON helper
  cli.py      run / pending / approve / edit / reject, with SQLite checkpoints
data/         sample customer records and help-center articles
examples/     sample inbound emails
tests/        pytest suite
```

## License

MIT © Zohaib Tariq
