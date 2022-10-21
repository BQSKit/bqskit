import logging
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

# import jax
# import jax.numpy as jnp

from bqskit.ir.opt.instantiater import Instantiater
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class QFactor_einsum(Instantiater):
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
        


    @staticmethod
    def _initilize_circuit_tensor(
        target,
        locations,
        untrys,
        use_jax: bool = False,
    ):

        target_untry_builder = UnitaryBuilder(target.num_qudits, target.radixes, target.conj().T)
        
        for loc, untry in zip(locations, untrys):
            target_untry_builder.apply_right(untry, loc, check_arguments=False, use_jax=use_jax)
            
        return target_untry_builder

    def instantiate(
        self,
        circuit,#: Circuit,
        target: UnitaryMatrix | StateVector,
        x0: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Instantiate `circuit`, see Instantiater for more info."""


        params = np.array(x0)      
        locations = [op.location for op in circuit]
        gates = [op.gate for op in circuit]

        untrys = []
        param_index = 0
        for gate in gates:
            amount_of_params_in_gate = gate.num_params
            gparams = params[param_index: param_index + amount_of_params_in_gate]
            untrys.append(gate.get_unitary(params=gparams))
            param_index += amount_of_params_in_gate


        amount_of_gates = len(gates)
        amount_of_qudits = target.num_qudits
        target_untry_builder = QFactor_einsum._initilize_circuit_tensor(target, locations, untrys)


        c1 = 0
        c2 = 1
        it = 0

        while(True):
            it += 1
            # Termination conditions
            if it > self.min_iters:

                if np.abs(c1 - c2) <= self.diff_tol_a + self.diff_tol_r * np.abs( c1 ):
                    diff = np.abs(c1 - c2)
                    _logger.info( f"Terminated: |c1 - c2| = {diff}"
                                " <= diff_tol_a + diff_tol_r * |c1|." )
                    break

                if it > self.max_iters:
                    _logger.info( "Terminated: iteration limit reached." )
                    break

            # from right to left
            for k in reversed(range(amount_of_gates)):
                gate = gates[k]
                location = locations[k]
                untry = untrys[k]

                # Remove current gate from right of circuit tensor
                target_untry_builder.apply_right(untry , location, inverse = True, check_arguments = False)

                # Update current gate
                if amount_of_params_in_gate > 0:
                    env = target_untry_builder.calc_env_matrix( location )            
                    untry =  gate.optimize(env, get_untry=True)
                    untrys[k] = untry
                    

                # Add updated gate to left of circuit tensor
                target_untry_builder.apply_left( untry, location,  check_arguments = False)


            # from left to right
            for k in range(amount_of_gates):
                gate = gates[k]
                location = locations[k]
                untry = untrys[k]
                
                # Remove current gate from left of circuit tensor
                target_untry_builder.apply_left( untry, location, inverse = True, check_arguments = False)

                # Update current gate
                if gate.num_params > 0:
                    env = target_untry_builder.calc_env_matrix(location)            
                    untry =  gate.optimize(env, get_untry=True)
                    untrys[k] = untry
                
                # Add updated gate to right of circuit tensor
                target_untry_builder.apply_right( untry, location,  check_arguments = False)

            c2 = c1
            c1 = np.abs( np.trace( target_untry_builder.get_unitary() ) )
            c1 = 1 - ( c1 / ( 2 ** amount_of_qudits ) )
            
            if c1 <= self.dist_tol:
                _logger.info( f"Terminated: c1 = {c1} <= dist_tol." )
                break

            if it % 100 == 0:
                _logger.info( f"iteration: {it}, cost: {c1}" )

            if it % 40 == 0:
                target_untry_builder = QFactor_einsum._initilize_circuit_tensor(target, locations, untrys)

        params = []
        for untry, gate in zip(untrys, gates):
            params.extend(gate.get_params(untry))

        return np.array(params)


    @staticmethod
    # def is_capable(circuit: Circuit) -> bool:
    def is_capable(circuit) -> bool:
        """Return true if the circuit can be instantiated."""
        return all(
            isinstance(gate, LocallyOptimizableUnitary)
            for gate in circuit.gate_set
        )
    
    @staticmethod
    def get_violation_report(circuit) -> str:
    # def get_violation_report(circuit: Circuit) -> str:
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

    @staticmethod
    def get_method_name() -> str:
        """Return the name of this method."""
        return 'qfactor_einsum'