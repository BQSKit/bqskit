"""
This subpackage implements Routing algorithms.

Routing algorithms are responsible routing logical gates via physical gates.
"""
from __future__ import annotations

from bqskit.passes.mapping.routing.sabre import GeneralizedSabreRoutingPass

__all__ = ['GeneralizedSabreRoutingPass']
