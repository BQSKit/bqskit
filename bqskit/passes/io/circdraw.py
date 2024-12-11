from bqskit.compiler import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ext import bqskit_to_qiskit
from qiskit import *

class CircuitDrawPass(BasePass):

    def __init__(
        self,
        marker: str = 'circuit',
    ) -> None:
        self.marker = marker

    async def run(self, circuit: Circuit, data: PassData):
        qc = bqskit_to_qiskit(circuit)
        qc.draw(output='mpl', filename=self.marker+'.jpg', scale=0.5, fold=-1)
