import logging
from typing import Any
from typing import TYPE_CHECKING

# import numpy as np
# import numpy.typing as npt

# import jax
import jax.numpy as jnp

from bqskit.ir.opt.instantiater import Instantiater
from bqskit.ir.opt.instantiaters.qfactor_einsum import QFactor_einsum
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class QFactor_jax(QFactor_einsum):
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
        """Instantiate `circuit`, see Instantiater for more info."""

        params = jnp.array(x0)      
        locations = [op.location for op in circuit]
        gates = [op.gate for op in circuit]

        untrys = []
        param_index = 0
        for gate in gates:
            amount_of_params_in_gate = gate.num_params
            gparams = params[param_index: param_index + amount_of_params_in_gate]
            untrys.append(gate.get_unitary(params=gparams, check_params=False, use_jax=True))
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

                if jnp.abs(c1 - c2) <= self.diff_tol_a + self.diff_tol_r * jnp.abs( c1 ):
                    diff = jnp.abs(c1 - c2)
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
            c1 = jnp.abs( jnp.trace( jnp.array(target_untry_builder.get_unitary().numpy) ) )
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

        return jnp.array(params)

    @staticmethod
    def get_method_name() -> str:
        """Return the name of this method."""
        return 'qfactor_jax'