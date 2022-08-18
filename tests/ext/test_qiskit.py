from __future__ import annotations

from bqskit.ext import bqskit_to_qiskit
from bqskit.ext import qiskit_to_bqskit


class TestTranslate:

    # test starting in bqskit, going through qiskit, coming out same unitary
    # test starting in qiskit, going through bqskit, coming out same unitary
    # test starting in qiskit, compiling in bqskit, coming out same unitary
    # test starting in qiskit, synthesizing in bqskit, coming out same unitary
    # test synthesizing/constructing in bqskit, coming out same unitary
    # test qiskit same unitary as bqskit
