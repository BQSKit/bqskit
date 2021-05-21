"""This module implements the Language base class and LangException class."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


class LangException(Exception):
    """Exceptions related to language encoding and decoding."""


class Language(abc.ABC):
    """The Language base class."""

    @abc.abstractmethod
    def encode(self, circuit: Circuit) -> str:
        """Write `circuit` in this language."""

    @abc.abstractmethod
    def decode(self, source: str) -> Circuit:
        """Parse `source` into a circuit."""
