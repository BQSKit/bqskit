"""This module implements the RuntimeFuture class."""
from __future__ import annotations

from typing import Any


class RuntimeFuture:
    """
    An awaitable future.

    These future objects must be awaited in the same task that created them.
    They cannot be used as input to other tasks.
    """

    def __init__(self, mailbox_id: int) -> None:
        """Initialize a future tied to a local mailbox."""

        self.mailbox_id = mailbox_id
        """The mailbox id where this future's result will be shipped."""

        self._next_flag = False
        """If flag is set, then awaiting returns next result."""

    def __await__(self) -> Any:
        """
        Wait on a result to be delivered.

        Informs the event loop which mailbox this is waiting on.
        """
        if self._next_flag:
            return (yield ('BQSKIT_NEXT_ID', self.mailbox_id))

        return (yield ('BQSKIT_MAIL_ID', self.mailbox_id))

    def __getstate__(self) -> Any:
        """Prevent a future from being sent to another process."""
        raise NotImplementedError(
            'These future objects must be awaited in the same task that'
            ' created them. They cannot be inputted to other tasks.',
        )

    @property
    def _done(self) -> bool:
        """
        Return true if the future is ready.

        Warning: Polling on futures in a busy-wait loop can cause deadlock.
        It is best to await on futures rather than continuously polling.
        Use at your own risk.
        """
        from bqskit.runtime.worker import get_worker

        if self.mailbox_id not in get_worker()._mailboxes:
            return True

        return get_worker()._mailboxes[self.mailbox_id].ready
