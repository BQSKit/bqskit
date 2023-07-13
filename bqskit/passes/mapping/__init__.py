"""This package implements passes that perform qudit assignment."""
from __future__ import annotations

from bqskit.passes.mapping.apply import ApplyPlacement
from bqskit.passes.mapping.embed import EmbedAllPermutationsPass
from bqskit.passes.mapping.layout.pam import PAMLayoutPass
from bqskit.passes.mapping.layout.sabre import GeneralizedSabreLayoutPass
from bqskit.passes.mapping.placement.greedy import GreedyPlacementPass
from bqskit.passes.mapping.placement.trivial import TrivialPlacementPass
from bqskit.passes.mapping.routing.pam import PAMRoutingPass
from bqskit.passes.mapping.routing.sabre import GeneralizedSabreRoutingPass
from bqskit.passes.mapping.setmodel import SetModelPass
from bqskit.passes.mapping.topology import SubtopologySelectionPass

__all__ = [
    'GeneralizedSabreLayoutPass',
    'GreedyPlacementPass',
    'TrivialPlacementPass',
    'GeneralizedSabreRoutingPass',
    'SetModelPass',
    'ApplyPlacement',
    'PAMLayoutPass',
    'PAMRoutingPass',
    'EmbedAllPermutationsPass',
    'SubtopologySelectionPass',
]
