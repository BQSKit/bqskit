bqskit.ir.Circuit
=================

.. currentmodule:: bqskit.ir

.. autoclass:: Circuit


   .. automethod:: __init__


   .. rubric:: Methods

   .. autosummary::

      ~Circuit.__init__
      ~Circuit.append
      ~Circuit.append_circuit
      ~Circuit.append_gate
      ~Circuit.append_qudit
      ~Circuit.batch_pop
      ~Circuit.batch_replace
      ~Circuit.become
      ~Circuit.check_parameters
      ~Circuit.check_region
      ~Circuit.check_valid_operation
      ~Circuit.clear
      ~Circuit.compress
      ~Circuit.copy
      ~Circuit.count
      ~Circuit.downsize_region
      ~Circuit.extend
      ~Circuit.extend_qudits
      ~Circuit.find_available_cycle
      ~Circuit.fold
      ~Circuit.format
      ~Circuit.freeze_param
      ~Circuit.from_file
      ~Circuit.from_operation
      ~Circuit.from_str
      ~Circuit.from_unitary
      ~Circuit.get_active_qudits
      ~Circuit.get_coupling_graph
      ~Circuit.get_dagger
      ~Circuit.get_depth
      ~Circuit.get_gate_set
      ~Circuit.get_grad
      ~Circuit.get_inverse
      ~Circuit.get_num_cycles
      ~Circuit.get_num_operations
      ~Circuit.get_num_params
      ~Circuit.get_operation
      ~Circuit.get_operations
      ~Circuit.get_parallelism
      ~Circuit.get_param
      ~Circuit.get_param_location
      ~Circuit.get_params
      ~Circuit.get_region
      ~Circuit.get_slice
      ~Circuit.get_statevector
      ~Circuit.get_unitary
      ~Circuit.get_unitary_and_grad
      ~Circuit.insert
      ~Circuit.insert_circuit
      ~Circuit.insert_gate
      ~Circuit.insert_qudit
      ~Circuit.instantiate
      ~Circuit.is_constant
      ~Circuit.is_cycle_in_range
      ~Circuit.is_cycle_unoccupied
      ~Circuit.is_differentiable
      ~Circuit.is_parameterized
      ~Circuit.is_point_idle
      ~Circuit.is_point_in_range
      ~Circuit.is_qubit_only
      ~Circuit.is_qudit_idle
      ~Circuit.is_qudit_in_range
      ~Circuit.is_qudit_only
      ~Circuit.is_qutrit_only
      ~Circuit.is_valid_region
      ~Circuit.minimize
      ~Circuit.normalize_point
      ~Circuit.operations
      ~Circuit.operations_with_cycles
      ~Circuit.point
      ~Circuit.pop
      ~Circuit.pop_cycle
      ~Circuit.pop_qudit
      ~Circuit.remove
      ~Circuit.remove_all
      ~Circuit.renumber_qudits
      ~Circuit.replace
      ~Circuit.replace_gate
      ~Circuit.replace_with_circuit
      ~Circuit.save
      ~Circuit.set_param
      ~Circuit.set_params
      ~Circuit.straighten
      ~Circuit.surround
      ~Circuit.unfold
      ~Circuit.unfold_all





   .. rubric:: Attributes

   .. autosummary::

      ~Circuit.dim
      ~Circuit.num_params
      ~Circuit.num_qudits
      ~Circuit.radixes
