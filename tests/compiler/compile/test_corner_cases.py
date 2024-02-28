from __future__ import annotations

from bqskit.compiler.compile import compile
from bqskit.compiler.compiler import Compiler
from bqskit.ir.lang.qasm2 import OPENQASM2Language


def test_cry_identity_corner_case(compiler: Compiler) -> None:
    qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[5];
        creg meas[5];
        cry(0) q[0],q[2];
        cry(0) q[1],q[2];
    """
    circuit = OPENQASM2Language().decode(qasm)
    _ = compile(circuit, optimization_level=1, seed=10)
    # Should finish
