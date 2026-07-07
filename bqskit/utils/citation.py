"""This module implements the Citation object and @cite decorator."""
from __future__ import annotations

from typing import Callable
from typing import TypeVar

T = TypeVar('T')


def cite(doi: str) -> Callable[[T], T]:
    """
    Attach citation metadata to a class.

    Can be stacked to attach multiple citations. Citations are inherited
    by subclasses and accessible via :func:`BasePass.get_citations`.

    Args:
        doi: A unique identifier for the citation (e.g. a DOI).
    """

    def decorator(cls: T) -> T:
        if '_cite_meta' not in cls.__dict__:
            setattr(cls, '_cite_meta', set())
        cls._cite_meta.add(doi)  # type: ignore
        return cls

    return decorator
