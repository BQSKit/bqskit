"""This module implements the various checkpointing passes."""
from __future__ import annotations

import logging
import pickle
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class SaveCheckpointPass(BasePass):
    """
    The SaveCheckpointPass class.

    The SaveCheckpointPass saves the current circuit as a data file.
    """

    def __init__(self, checkpoint_filename: str) -> None:
        """
        Constructor for the SaveCheckpointPass.

        Args:
            checkpoint_filename (str): Full path name for the checkpoint.
        """
        self.checkpoint_filename = checkpoint_filename

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see BasePass for more info."""
        with open(self.checkpoint_filename, 'wb') as f:
            pickle.dump((circuit, data), f)


class LoadCheckpointPass(BasePass):
    """
    The LoadCheckpointPass class.

    The LoadCheckpointPass loads the circuit from a data file.
    """

    def __init__(self, checkpoint_filename: str) -> None:
        """
        Constructor for the SaveCheckpointPass.

        Args:
            checkpoint_filename (str): Full path name for the checkpoint.
        """
        self.checkpoint_filename = checkpoint_filename

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see BasePass for more info."""
        with open(self.checkpoint_filename, 'rb') as f:
            checkpoint = pickle.load(f)
            circuit.become(checkpoint[0])
            data.clear()
            data.update(checkpoint[1])
