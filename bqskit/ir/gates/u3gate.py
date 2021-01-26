import numpy as np

from bqskit.ir import Gate


class U3Gate(QubitGate):

    num_params = 3
    gate_size = 1

    def get_unitary(self, params):
        return self.calculate_u3(params)
