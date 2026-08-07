"""This module implements the DistributedPredicate class."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bqskit.passes.control.predicate import PassPredicate

if TYPE_CHECKING:
    from bqskit.compiler.passdata import PassData
    from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class DistributedPredicate(PassPredicate):
    """
    The DistributedPredicate class.

    The DistributedPredicate returns true if the targeted machine is distributed
    across multiple chips.
    """

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        return data.model.coupling_graph.is_distributed()
