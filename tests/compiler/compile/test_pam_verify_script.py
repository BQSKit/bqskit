from bqskit.compiler import Compiler
import test_pam_verify as tpv
import logging

compiler = Compiler()

logging.basicConfig(level=logging.INFO)

# Format
# tpv.test_pam_verify(compiler, 'path/to/qasm/file')

tpv.test_pam_verify(compiler, 'test/qft4.qasm')