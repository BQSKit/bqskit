from __future__ import annotations

from hypothesis import given

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import HGate
from bqskit.passes.search.generators import SeedLayerGenerator
from bqskit.utils.test.strategies import circuits


@given(circuits(2))
def test_seed_hash(circuit: Circuit) -> None:
    hash1 = SeedLayerGenerator.hash_structure(circuit)
    circuit.append_gate(HGate(), 0)
    hash2 = SeedLayerGenerator.hash_structure(circuit)
    assert hash1 != hash2
