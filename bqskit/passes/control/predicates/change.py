"""This module implements the ChangePredicate class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.ir.circuit import Circuit
from bqskit.passes.control.predicate import PassPredicate

_logger = logging.getLogger(__name__)


class ChangePredicate(PassPredicate):
    """
    The ChangePredicate class.

    The ChangePredicate returns true if the circuit has changed since the last
    call. On the first call, the predicate returns true.
    """

    key = 'ChangePredicate_circuit_hash'

    def get_truth_value(self, circuit: Circuit, data: dict[str, Any]) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""

        # If first call, record data and return true
        if self.key not in data:
            _logger.debug(f'Could not find {self.key} key.')
            data[self.key] = self.get_hash(circuit)
            return True

        # Otherwise, check for a change and return accordingly
        new_hash = self.get_hash(circuit)
        if data[self.key] == new_hash:
            _logger.debug('Hashes match; no change detected.')
            return False

        data[self.key] = new_hash
        _logger.debug('Hashes do not match; change detected.')
        return True

    def get_hash(self, circuit: Circuit) -> int:
        """Retreive hash associated with `circuit`."""
        _logger.debug('Calculating hash for circuit...')

        hashes: list[int] = []
        for op in circuit:
            hashes.append(hash(repr(op)))

            # Don't let the hash list grow too large.
            if len(hashes) >= 100:
                hashes = [hash(tuple(hashes))]

        hash_val = hash(tuple(hashes)) if len(hashes) > 1 else hashes[0]
        _logger.debug(f'Hash: {hash_val}')
        return hash_val
