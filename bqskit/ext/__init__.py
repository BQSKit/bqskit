"""
=============================================================
BQSKit Extensions (:mod:`bqskit.ext`)
=============================================================

.. currentmodule:: bqskit.ext

This subpackage provides integrations with other popular frameworks.
"""
from __future__ import annotations

try:
    import qiskit  # noqa
    import pytket  # noqa
    import cirq  # noqa
except ImportError as e:
    raise ImportError(
        '\n\nUnable to import bqskit.ext package.\n'
        'Ensure that bqskit was installed with extension support.\n'
        'To install bqskit with extension support try:\n'
        '\tpip install bqskit[ext]\n',
    ) from e


from bqskit.ext.qiskit.translate import bqskit_to_qiskit
from bqskit.ext.qiskit.translate import qiskit_to_bqskit


__all__ = [
    'bqskit_to_qiskit',
    'qiskit_to_bqskit',
]
