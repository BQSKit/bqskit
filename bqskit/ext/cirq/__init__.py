"""This package contains integrations with the Cirq framework."""
from __future__ import annotations

from bqskit.ext.cirq.translate import bqskit_to_cirq
from bqskit.ext.cirq.translate import cirq_to_bqskit

__all__ = ['bqskit_to_cirq', 'cirq_to_bqskit']
