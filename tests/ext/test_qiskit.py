from bqskit.ext import qiskit_to_bqskit, bqskit_to_qiskit

class TestTranslate:
    
    # test starting in bqskit, going through qiskit, coming out same unitary
    # test starting in qiskit, going through bqskit, coming out same unitary
    # test starting in qiskit, compiling in bqskit, coming out same unitary
    # test starting in qiskit, synthesizing in bqskit, coming out same unitary
    # test synthesizing/constructing in bqskit, coming out same unitary
    # test qiskit same unitary as bqskit