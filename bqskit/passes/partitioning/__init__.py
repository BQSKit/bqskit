"""This package implements partitioning passes."""
from __future__ import annotations

from bqskit.passes.partitioning.cluster import ClusteringPartitioner
from bqskit.passes.partitioning.greedy import GreedyPartitioner
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.partitioning.scan import ScanPartitioner

__all__ = [
    'ClusteringPartitioner',
    'GreedyPartitioner',
    'ScanPartitioner',
    'QuickPartitioner',
]
