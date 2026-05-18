"""This module implements the Citation object and @cite decorator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import TypeVar

T = TypeVar('T')


@dataclass
class Citation:
    """Represents a reference to a publication."""

    key: str  # most likely a doi number
    bibtex: str

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Citation):
            return NotImplemented

        return self.key == other.key


def cite(key: str, bibtex: str) -> Callable[[T], T]:
    """
    Attach citation metadata to a class.

    Can be stacked to attach multiple citations. Citations are inherited
    by subclasses and accessible via :func:`BasePass.get_citations`.

    Args:
        key: A unique identifier for the citation (e.g. a DOI).
        bibtex: A BibTeX string for the citation.
    """

    def decorator(cls: T) -> T:
        if '_cite_meta' not in cls.__dict__:
            setattr(cls, '_cite_meta', set())
        cls._cite_meta.add(Citation(key=key, bibtex=bibtex))  # type: ignore
        return cls

    return decorator
