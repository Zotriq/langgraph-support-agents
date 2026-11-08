import json


class ScriptedLLM:
    """Plays every agent role. Each call is logged so tests can check what each agent saw."""

    def __init__(self, category="billing", reviews=None):
        self.category = category
        self.reviews = list(reviews) if reviews is not None else [{"approved": True, "problems": []}]
        self.calls = []

    def chat(self, system, user):
        self.calls.append((system, user))
        if system.startswith("You triage"):
            return json.dumps({"category": self.category, "urgency": "high", "summary": "test"})
        if system.startswith("You are the ") and "specialist" in system:
            return f"NOTES[{system.split('the ')[1].split(' specialist')[0]}] context={user[user.find('Context:'):][:80]!r}"
        if system.startswith("You write"):
            n = sum(1 for s, _ in self.calls if s.startswith("You write"))
            return f"Draft v{n}"
        if system.startswith("You are QA"):
            verdict = self.reviews.pop(0) if self.reviews else {"approved": False, "problems": ["still wrong"]}
            return "```json\n" + json.dumps(verdict) + "\n```"
        raise AssertionError(f"unexpected prompt: {system[:60]}")

    def prompts(self, prefix):
        return [u for s, u in self.calls if s.startswith(prefix)]
