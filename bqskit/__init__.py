from __future__ import annotations

import logging

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.compiler import TaskStatus
from bqskit.ir import Circuit
# Initialize Logging
_logger = logging.getLogger('bqskit')
_logger.setLevel(logging.CRITICAL)
_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_fmt = '%(levelname)-8s | %(message)s'
_formatter = logging.Formatter(_fmt)
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)
