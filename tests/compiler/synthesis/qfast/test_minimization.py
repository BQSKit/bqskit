"""This test will ensure that the circuit's built by QFAST can be minimized successfully."""

from typing import Sequence

import numpy as np
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.opt.functions.hsd import HSDistanceFunction
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import PauliGate
from scipy.stats import unitary_group
import scipy.optimize as opt

def test_full_pauli_gate() -> None:
    circuit = Circuit(3)
    circuit.append_gate(PauliGate(3), [0, 1, 2])
    # cost = HSDistanceFunction(circuit, UnitaryMatrix(unitary_group.rvs(8)))
    # circuit.minimize(cost)
    # assert cost.get_cost(circuit.get_params()) < 1e-6

    M, dM = PauliGate(3).get_unitary_and_grad(circuit.get_params())
    MC, dMC = circuit.get_unitary_and_grad(circuit.get_params())
    assert np.allclose(M.get_numpy(), MC.get_numpy())
    assert np.allclose(dM, dMC)


    target_h = UnitaryMatrix(unitary_group.rvs(8)).get_dagger().get_numpy()
    dem = 8

    def get_cost_and_grad(params: Sequence[float]) -> tuple[float, np.ndarray]:
        """Return the cost and gradient given the input parameters."""
        M, dM = circuit.get_unitary_and_grad(params)
        MC, dMC = PauliGate(3).get_unitary_and_grad(params)
        assert np.allclose(M.get_numpy(), MC.get_numpy())
        assert np.allclose(dM, dMC)
        trace_prod = np.trace(target_h @ M.get_numpy())
        num = np.abs(trace_prod)
        obj = 1 - (num / dem)
        d_trace_prod = np.array([np.trace(target_h @ pM) for pM in dM])
        jacs = -(
            np.real(trace_prod) * np.real(d_trace_prod)
            + np.imag(trace_prod) * np.imag(d_trace_prod)
        )
        jacs *= dem / num
        return obj, jacs
    
    res = opt.minimize(
        get_cost_and_grad,
        circuit.get_params(),
        jac=True,
        method='L-BFGS-B',
    )

    print(res)
