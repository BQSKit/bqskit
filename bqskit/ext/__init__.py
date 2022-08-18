"""
=============================================================
BQSKit Extensions (:mod:`bqskit.ext`)
=============================================================

.. currentmodule:: bqskit.ext

This subpackage provides integrations with other popular frameworks.
"""
from __future__ import annotations

try:
    import cirq  # noqa
    import pytket  # noqa
    import qiskit  # noqa
    import qutip  # noqa
except ImportError as e:
    raise ImportError(
        '\n\nUnable to import bqskit.ext package.\n'
        'Ensure that bqskit was installed with extension support.\n'
        'To install bqskit with extension support try:\n'
        '\tpip install bqskit[ext]\n',
    ) from e

from bqskit.ext.cirq.translate import bqskit_to_cirq
from bqskit.ext.cirq.translate import cirq_to_bqskit
from bqskit.ext.pytket.translate import bqskit_to_pytket
from bqskit.ext.pytket.translate import pytket_to_bqskit
from bqskit.ext.qiskit.models import model_from_backend
from bqskit.ext.qiskit.translate import bqskit_to_qiskit
from bqskit.ext.qiskit.translate import qiskit_to_bqskit
from bqskit.ext.qutip.translate import bqskit_to_qutip
from bqskit.ext.qutip.translate import qutip_to_bqskit
from bqskit.ext.supermarq import supermarq_program_communication
from bqskit.ext.supermarq import supermarq_critical_depth
from bqskit.ext.supermarq import supermarq_entanglement_ratio
from bqskit.ext.supermarq import supermarq_parallelism
from bqskit.ext.supermarq import supermarq_liveness


__all__ = [
    'bqskit_to_cirq',
    'cirq_to_bqskit',
    'bqskit_to_pytket',
    'pytket_to_bqskit',
    'model_from_backend',
    'bqskit_to_qiskit',
    'qiskit_to_bqskit',
    'bqskit_to_qutip',
    'qutip_to_bqskit',
    'supermarq_program_communication',
    'supermarq_critical_depth',
    'supermarq_entanglement_ratio',
    'supermarq_parallelism',
    'supermarq_liveness',
]
