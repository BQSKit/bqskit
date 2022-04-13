from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix
from bqskit.passes import ZXZXZDecomposition

def test_zxzxz_decomposition() -> None:
    for i in range(1000):
        test_utry = UnitaryMatrix.random(1)
        test_circuit = Circuit.from_unitary(test_utry)
        ZXZXZDecomposition().run(test_circuit)
        assert test_circuit.get_unitary().get_distance_from(test_utry) < 5e-8
