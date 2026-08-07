"""This module implements the MessageDirection enum."""
from __future__ import annotations

from enum import IntEnum


class MessageDirection(IntEnum):
    """The direction a message came from."""

    BELOW = 0
    """This describes messages coming from a child manager/worker."""

    ABOVE = 1
    """This describes messages coming from a parent server/manager."""

    CLIENT = 2
    """This describes messages coming from a client."""

    SIGNAL = 3
    """This describes internal directives created by a signal handler."""
