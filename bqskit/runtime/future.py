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

    def __await__(self) -> Any:
        """
        Wait on a package to be delivered.

        Informs the event loop which mailbox this is waiting on.
        """
        return (yield ('BQSKIT_MAIL_ID', self.mailbox_id))

    def __getstate__(self) -> Any:
        """Prevent a future from being sent to another process."""
        raise NotImplementedError(
            'These future objects must be awaited in the same task that'
            ' created them. They cannot be inputted to other tasks.',
        )

    @property
    def done(self) -> bool:
        """Return true if the future is ready."""
        from bqskit.runtime.worker import get_worker

        if self.mailbox_id not in get_worker().mailboxes:
            return True

        return get_worker().mailboxes[self.mailbox_id][0] is True
