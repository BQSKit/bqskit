
import logging
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import jax
import jax.numpy as jnp
from bqskit.ir.opt.instantiater import Instantiater
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)

class QFactor_jax_batched_jit(Instantiater):
    """The QFactor batch circuit instantiater."""

    def __init__(
        self,
        diff_tol_a: float = 1e-12 ,
        diff_tol_r: float = 1e-6,
        dist_tol: float = 1e-10,
        max_iters: int = 100000,
        min_iters: int = 1000,
    ):

        if not isinstance( diff_tol_a, float ) or diff_tol_a > 0.5:
            raise TypeError( "Invalid absolute difference threshold." )

        if not isinstance( diff_tol_r, float ) or diff_tol_r > 0.5:
            raise TypeError( "Invalid relative difference threshold." )

        if not isinstance( dist_tol, float ) or dist_tol > 0.5:
            raise TypeError( "Invalid distance threshold." )

        if not isinstance( max_iters, int ) or max_iters < 0:
            raise TypeError( "Invalid maximum number of iterations." )

        if not isinstance( min_iters, int ) or min_iters < 0:
            raise TypeError( "Invalid minimum number of iterations." )

        

        self.diff_tol_a = diff_tol_a
        self.diff_tol_r = diff_tol_r
        self.dist_tol   = dist_tol
        self.max_iters  = max_iters
        self.min_iters  = min_iters
        


    def instantiate(
        self,
        circuit,#: Circuit,
        target: UnitaryMatrix | StateVector,
        x0
    ) :

        return self.instantiate_multistart(circuit, target, [x0])
    
    
    def instantiate_multistart(
        self,
        circuit,#: Circuit,
        target: UnitaryMatrix | StateVector,
        starts: list[npt.NDArray[np.float64]]
    ) :
        """Instantiate `circuit`, see Instantiater for more info."""

        amount_of_starts = len(starts)
        starts = jnp.array(starts)      
        locations = tuple([op.location for op in circuit])
        gates = tuple([op.gate for op in circuit])
        

        untrys = [[] for _ in range(amount_of_starts)]
        param_index = 0
        for gate in gates:
            amount_of_params_in_gate = gate.num_params
            
            for start_index in range(amount_of_starts):                
                gparams = starts[start_index][param_index: param_index + amount_of_params_in_gate]
                untrys[start_index].append(gate.get_unitary(params=gparams, check_params=False, use_jax=True).numpy)
            
            param_index += amount_of_params_in_gate

        untrys = jnp.array(untrys)
        n = 40
        c1s = [0] * amount_of_starts
        c2s = [1] * amount_of_starts
        it = 0
        best_start = 0

        sweep_vmaped = jax.vmap(_sweep_circuit, in_axes = (None, None, None, 0, None))

        while(True):
            it += 1
            # Termination conditions
            if it*n > self.min_iters:

                if all([jnp.abs(c1 - c2) <= self.diff_tol_a + self.diff_tol_r * jnp.abs( c1 ) for c1, c2 in zip(c1s,c2s)]):
                    # diff = jnp.abs(c1 - c2)
                    _logger.info( f"Terminated: |c1 - c2| = "
                                " <= diff_tol_a + diff_tol_r * |c1|." )
                    best_start = np.argmin(c1s)
                    break

                if it*n > self.max_iters:
                    _logger.info( "Terminated: iteration limit reached." )
                    best_start = jnp.argmin(c1s)
                    break

            c2s = c1s
            c1s, untrys = sweep_vmaped(target, locations, gates, untrys, n)

            reached_desired_distance = [c1 <= self.dist_tol for c1 in c1s]
            if any(reached_desired_distance):
                
                best_start = reached_desired_distance.index(True)
                _logger.info( f"Terminated: {it} c1 = {c1s} <= dist_tol.\n Best start is {best_start}" )
                break

            if it % 4 == 0:
                _logger.info( f"iteration: {it*n}, costs: {c1s}" )


        params = []
        for untry, gate in zip(untrys[best_start], gates):
            params.extend(gate.get_params(untry))

        return np.array(params)

    @staticmethod
    def get_method_name() -> str:
        """Return the name of this method."""
        return 'qfactor_jax_batched_jit'

    @staticmethod
    def can_internaly_perform_multistart() -> bool:
        """Probes if the instantiater can internaly perform multistrat """
        return True
    
    @staticmethod
    def is_capable(circuit) -> bool:
        """Return true if the circuit can be instantiated."""
        return all(
            isinstance(gate, LocallyOptimizableUnitary)
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
            if not isinstance(gate, LocallyOptimizableUnitary)
        }

        if len(invalid_gates) == 0:
            raise ValueError('Circuit can be instantiated.')

        return (
            'Cannot instantiate circuit with qfactor'
            ' because the following gates are not locally optimizable: %s.'
            % ', '.join(str(g) for g in invalid_gates)
        )



def _initilize_circuit_tensor(
        target_num_qudits,
        target_radixes,
        locations,        
        target_mat,
        untrys
    ):

        use_jax = True
        target_untry_builder = UnitaryBuilder(target_num_qudits, target_radixes, target_mat.conj().T)
        
        for loc, untry in zip(locations, untrys):
            target_untry_builder.apply_right(untry, loc, check_arguments=False, use_jax=use_jax)
            
        return target_untry_builder

_initilize_circuit_tensor_jit = jax.jit(_initilize_circuit_tensor, static_argnums=(0,1,2))


def _single_sweep(locations, gates, untrys, amount_of_gates, target_untry_builder):
    # from right to left
    for k in reversed(range(amount_of_gates)):
        gate = gates[k]
        location = locations[k]
        untry = untrys[k]

            # Remove current gate from right of circuit tensor
        target_untry_builder.apply_right(untry , location, inverse = True, check_arguments = False, use_jax=True)

            # Update current gate
        if gate.num_params > 0:
            env = target_untry_builder.calc_env_matrix( location , use_jax=True)            
            untry =  gate.optimize(env, get_untry=True, use_jax=True)
            untrys[k] = untry
                

            # Add updated gate to left of circuit tensor
        target_untry_builder.apply_left( untry, location,  check_arguments = False, use_jax=True)


        # from left to right
    for k in range(amount_of_gates):
        gate = gates[k]
        location = locations[k]
        untry = untrys[k]
            
            # Remove current gate from left of circuit tensor
        target_untry_builder.apply_left( untry, location, inverse = True, check_arguments = False, use_jax=True)

            # Update current gate
        if gate.num_params > 0:
            env = target_untry_builder.calc_env_matrix(location, use_jax=True)            
            untry =  gate.optimize(env, get_untry=True, use_jax=True)
            untrys[k] = untry
            
            # Add updated gate to right of circuit tensor
        target_untry_builder.apply_right( untry, location,  check_arguments = False, use_jax=True)
    
    return target_untry_builder, untrys

_single_sweep_jit = jax.jit(_single_sweep, static_argnums=(0,1,3))
    
def _sweep_circuit(target:UnitaryMatrix, locations, gates, untrys, n:int):

    amount_of_gates = len(gates)
    untrys_as_matrixs = []
    for gate_index in range(amount_of_gates):
        untrys_as_matrixs.append(UnitaryMatrix(untrys[gate_index], gates[gate_index].radixes, check_arguments=False, use_jax=True))

    untrys = untrys_as_matrixs
    target_untry_builder = _initilize_circuit_tensor_jit(target.num_qudits, target.radixes, locations, target.numpy, untrys)
    amount_of_qudits = target.num_qudits

    for _ in range(n):        
        target_untry_builder, untrys = _single_sweep_jit(locations, gates, untrys, amount_of_gates, target_untry_builder)

    c1 = jnp.abs( jnp.trace( jnp.array(target_untry_builder.get_unitary(use_jax=True).numpy) ) )
    c1 = 1 - ( c1 / ( 2 ** amount_of_qudits ) )

    return c1, jnp.array([untry.numpy for untry in untrys])

