"""Base classes and data structures for labeling functions (LFs).

* A token LF maps a tokenised sentence to one vote per token: an entity type,
  "O", or `None` (abstain).
* A relation LF maps an entity-pair candidate to the candidate's relation,
  `NO_RELATION`, or `None` (abstain).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from common.schema import NO_RELATION

TokenVote = Optional[str]


def find_subsequence(tokens: Sequence[str], pattern: Sequence[str]) -> list[int]:
    """Start indices of every occurrence of `pattern` in `tokens`."""
    size = len(pattern)
    if size == 0 or size > len(tokens):
        return []
    first = pattern[0]
    return [
        index
        for index in range(len(tokens) - size + 1)
        if tokens[index] == first and tuple(tokens[index : index + size]) == tuple(pattern)
    ]


class TokenLabelingFunction(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def apply(self, tokens: Sequence[str]) -> list[TokenVote]:
        """Return one vote per token."""

    def __call__(self, tokens: Sequence[str]) -> list[TokenVote]:
        votes = self.apply(tokens)
        if len(votes) != len(tokens):
            raise RuntimeError(f"LF '{self.name}' returned {len(votes)} votes for {len(tokens)} tokens")
        return votes

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


@dataclass(frozen=True)
class EntitySpan:
    start: int
    end: int
    type: str
    confidence: float = 1.0

    def text(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens[self.start : self.end])


@dataclass(frozen=True)
class RelationCandidate:
    tokens: tuple[str, ...]
    head: EntitySpan
    tail: EntitySpan
    relation: str  # the only relation the schema allows for this type pair

    @property
    def gap(self) -> range:
        """Token indices strictly between the two spans."""
        return range(min(self.head.end, self.tail.end), max(self.head.start, self.tail.start))

    @property
    def distance(self) -> int:
        return len(self.gap)


class RelationLabelingFunction(ABC):
    def __init__(self, name: str, weight: float = 1.0) -> None:
        self.name = name
        self.weight = float(weight)

    @abstractmethod
    def apply(self, candidate: RelationCandidate) -> str | None:
        """Return the candidate relation, NO_RELATION or None (abstain)."""

    def __call__(self, candidate: RelationCandidate) -> str | None:
        vote = self.apply(candidate)
        if vote not in (None, NO_RELATION, candidate.relation):
            raise RuntimeError(f"Relation LF '{self.name}' voted '{vote}' for a {candidate.relation} candidate")
        return vote
