"""This module implements the TaskStatus enum type."""


from enum import Enum


class TaskStatus(Enum):
    """TaskStatus enum type."""
    WAITING = 0  # The task is waiting in a workqueue.
    RUNNING = 1  # The task is currently running.
    DONE = 2     # The task is finished.
    ERROR = 3    # The task encountered an error or does not exist.
