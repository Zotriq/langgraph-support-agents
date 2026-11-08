"""Tools the specialist agents use: customer/billing lookup and knowledge-base search."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parents[1] / "data"


@dataclass
class Email:
    sender: str
    subject: str
    body: str

    @classmethod
    def parse(cls, raw: str) -> "Email":
        header, _, body = raw.strip().partition("\n\n")
        fields = dict(
            (k.strip().lower(), v.strip())
            for k, v in (line.split(":", 1) for line in header.splitlines() if ":" in line)
        )
        return cls(sender=fields.get("from", "").lower(), subject=fields.get("subject", ""), body=body.strip())


class CustomerDB:
    def __init__(self, path: Path = DATA_DIR / "customers.json"):
        self._customers = {c["email"]: c for c in json.loads(path.read_text())["customers"]}

    def lookup(self, email: str) -> dict[str, Any] | None:
        return self._customers.get(email.lower())


class KnowledgeBase:
    """Tiny keyword search over markdown articles. Swap for a vector store in production."""

    def __init__(self, folder: Path = DATA_DIR / "kb"):
        self.articles = {p.stem: p.read_text() for p in sorted(folder.glob("*.md"))}

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}

    def search(self, query: str, k: int = 2) -> list[tuple[str, str]]:
        q = self._tokens(query)
        scored = sorted(
            ((len(q & self._tokens(text)), name, text) for name, text in self.articles.items()),
            reverse=True,
        )
        return [(name, text) for score, name, text in scored[:k] if score > 0]
