from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.unitary_acc import VariableUnitaryGateAcc

from bqskit.ir.opt.instantiater import Instantiater
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarybuilderjax import UnitaryBuilderJax
from bqskit.qis.unitary.unitarymatrixjax import UnitaryMatrixJax
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

from scipy.stats import unitary_group


if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class QFactor_jax_batched_jit(Instantiater):
    """The QFactor batch circuit instantiater."""

    def __init__(
        self,
        diff_tol_a: float = 1e-12,
        diff_tol_r: float = 1e-6,
        dist_tol: float = 1e-10,
        max_iters: int = 100000,
        min_iters: int = 1000,
    ):

        if not isinstance(diff_tol_a, float) or diff_tol_a > 0.5:
            raise TypeError('Invalid absolute difference threshold.')

        if not isinstance(diff_tol_r, float) or diff_tol_r > 0.5:
            raise TypeError('Invalid relative difference threshold.')

        if not isinstance(dist_tol, float) or dist_tol > 0.5:
            raise TypeError('Invalid distance threshold.')

        if not isinstance(max_iters, int) or max_iters < 0:
            raise TypeError('Invalid maximum number of iterations.')

        if not isinstance(min_iters, int) or min_iters < 0:
            raise TypeError('Invalid minimum number of iterations.')

        self.diff_tol_a = diff_tol_a
        self.diff_tol_r = diff_tol_r
        self.dist_tol = dist_tol
        self.max_iters = max_iters
        self.min_iters = min_iters

    def instantiate(
        self,
        circuit,  # : Circuit,
        target: UnitaryMatrix | StateVector,
        x0,
    ):

        return self.instantiate_multistart(circuit, target, [x0])

    def instantiate_multistart(
        self,
        circuit,  # : Circuit,
        target: UnitaryMatrix | StateVector,
        starts: list[npt.NDArray[np.float64]],
    ):
        """Instantiate `circuit`, see Instantiater for more info."""
        target = UnitaryMatrixJax(target)
        amount_of_starts = len(starts)
        locations = tuple([op.location for op in circuit])
        gates = tuple([op.gate for op in circuit])
        biggest_gate_size = max((gate.num_qudits for gate in gates))
        
        
        untrys = []

        for gate in gates:
            size_of_untry = 2**gate.num_qudits
            untrys.append([_apply_padding_and_flatten(unitary_group.rvs(size_of_untry), gate, biggest_gate_size) for _ in range(amount_of_starts)])

                
                
        untrys = jnp.array(np.stack(untrys, axis=1))
        n = 40
        c1s = jnp.array([1] * amount_of_starts)
        it = 0
        it2 = 0
        best_start = 0

        while (True):
            c1s, untrys, plato_calc, reached_desired_distance = _sweep_jited_vmaped(target, locations, gates, untrys, n, c1s, self.dist_tol, self.diff_tol_a, self.diff_tol_r)
            
            it += n
            it2 +=1

            if it2 % 4 == 0:
                _logger.info(f'iteration: {it}, costs: {c1s}')
            
            # Termination conditions            
            if any(reached_desired_distance):

                best_start = reached_desired_distance.tolist().index(True)
                _logger.info(
                    f'Terminated: {it} c1 = {c1s} <= dist_tol.\n Best start is {best_start}',
                )
                break

            if it > self.min_iters:
                if all(plato_calc):
                    _logger.info(
                        f'Terminated: |c1 - c2| = '
                        ' <= diff_tol_a + diff_tol_r * |c1|.',
                    )
                    best_start = jnp.argmin(c1s)
                    break

                if it > self.max_iters:
                    _logger.info('Terminated: iteration limit reached.')
                    best_start = jnp.argmin(c1s)
                    break

        params = []
        for untry, gate in zip(untrys[best_start], gates):
            params.extend(gate.get_params(_remove_padding_and_create_matrix(untry, gate)))

        return np.array(params)

    @staticmethod
    def get_method_name() -> str:
        """Return the name of this method."""
        return 'qfactor_jax_batched_jit'

    @staticmethod
    def can_internaly_perform_multistart() -> bool:
        """Probes if the instantiater can internaly perform multistrat."""
        return True

    @staticmethod
    def is_capable(circuit) -> bool:
        """Return true if the circuit can be instantiated."""
        return all(
            isinstance(gate, (VariableUnitaryGateAcc, U3Gate, ConstantGate))
            for gate in circuit.gate_set
        )

    @staticmethod
    def get_violation_report(circuit) -> str:
        """
        Return a message explaining why `circuit` cannot be instantiated.

        Args:
            circuit (Circuit): Generate a report for this circuit.

        Raises:
            ValueError: If `circuit` can be instantiated with this
                instantiater.
        """

        invalid_gates = {
            gate
            for gate in circuit.gate_set
            if not isinstance(gate, (VariableUnitaryGateAcc, U3Gate, ConstantGate))
        }

        if len(invalid_gates) == 0:
            raise ValueError('Circuit can be instantiated.')

        return (
            'Cannot instantiate circuit with qfactor'
            ' because the following gates are not locally optimizable with jax: %s.'
            % ', '.join(str(g) for g in invalid_gates)
        )


def _initilize_circuit_tensor(
    target_num_qudits,
    target_radixes,
    locations,
    target_mat,
    untrys,
):

    target_untry_builder = UnitaryBuilderJax(
        target_num_qudits, target_radixes, target_mat.conj().T,
    )

    for loc, untry in zip(locations, untrys):
        target_untry_builder.apply_right(
            untry, loc, check_arguments=False,
        )

    return target_untry_builder

def _single_sweep(locations, gates,  amount_of_gates, target_untry_builder, untrys):
    # from right to left
    for k in reversed(range(amount_of_gates)):
        gate = gates[k]
        location = locations[k]
        untry = untrys[k]

        # Remove current gate from right of circuit tensor
        target_untry_builder.apply_right(
            untry, location, inverse=True, check_arguments=False
        )

        # Update current gate
        if gate.num_params > 0:
            env = target_untry_builder.calc_env_matrix(location)
            untry = gate.optimize(env, get_untry=True)
            untrys[k] = untry

            # Add updated gate to left of circuit tensor
        target_untry_builder.apply_left(
            untry, location, check_arguments=False,
        )

        # from left to right
    for k in range(amount_of_gates):
        gate = gates[k]
        location = locations[k]
        untry = untrys[k]

        # Remove current gate from left of circuit tensor
        target_untry_builder.apply_left(
            untry, location, inverse=True, check_arguments=False,
        )

        # Update current gate
        if gate.num_params > 0:
            env = target_untry_builder.calc_env_matrix(location)
            untry = gate.optimize(env, get_untry=True)
            untrys[k] = untry

            # Add updated gate to right of circuit tensor
        target_untry_builder.apply_right(
            untry, location, check_arguments=False,
        )

    return target_untry_builder, untrys

_single_sweep_jit = jax.jit(_single_sweep, static_argnums=(0, 1, 2))



def _apply_padding_and_flatten(untry, gate, max_gate_size):
    zero_pad_size = (2**max_gate_size)**2 - (2**gate.num_qudits)**2
    if zero_pad_size > 0:
        zero_pad = jnp.zeros(zero_pad_size)
        return jnp.concatenate((untry, zero_pad), axis=None)
    else:
        return jnp.array(untry.flatten())

def _remove_padding_and_create_matrix(untry, gate):
    len_of_matrix = 2**gate.num_qudits
    size_to_keep = len_of_matrix**2
    return untry[:size_to_keep].reshape((len_of_matrix, len_of_matrix))

def _sweep_circuit(target: UnitaryMatrix, locations, gates, untrys, n: int, c1, dist_tol, diff_tol_a, diff_tol_r):
    amount_of_gates = len(gates)
    untrys_as_matrixs = []
    for gate_index, gate in enumerate(gates):
        untrys_as_matrixs.append(
            UnitaryMatrixJax(
                _remove_padding_and_create_matrix(untrys[gate_index], gate), gate.radixes
            ),
        )

    untrys = untrys_as_matrixs
    target_untry_builder = _initilize_circuit_tensor(
        target.num_qudits, target.radixes, locations, target.numpy, untrys,
    )
    amount_of_qudits = target.num_qudits

    
    sweep_loop_body = lambda i,x: _single_sweep_jit(locations, gates,  amount_of_gates, target_untry_builder=x[0], untrys=x[1])
    target_untry_builder, untrys = jax.lax.fori_loop(0, n, sweep_loop_body, (target_untry_builder, untrys))


    c2 = c1
    dim = target_untry_builder.dim
    untry_res = target_untry_builder.tensor.reshape((dim, dim))
    c1 = jnp.abs(jnp.trace(untry_res))
    c1 = 1 - (c1 / (2 ** amount_of_qudits))

    plato_calc = jnp.abs(c1 - c2) <= diff_tol_a + diff_tol_r * jnp.abs(c1)
    reached_required_tol = c1 < dist_tol

    biggest_gate_size = max((gate.num_qudits for gate in gates))
    final_untrys_padded = jnp.array([_apply_padding_and_flatten(untry.numpy.flatten(), gate, biggest_gate_size) for untry, gate in zip(untrys, gates)])

    return c1, final_untrys_padded, plato_calc , reached_required_tol

_sweep_circuit_jited = jax.jit(_sweep_circuit, static_argnums=(1,2,4,6,7,8))

_sweep_jited_vmaped = jax.vmap(
        _sweep_circuit_jited, in_axes=(None, None, None, 0, None, 0, None, None, None),
        )


