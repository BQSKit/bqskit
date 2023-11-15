from __future__ import annotations

import os
from typing import Any

import pytest


@pytest.fixture(params=[1, 2, 3, 4])
def optimization_level(request: Any) -> int:
    """All valid optimization_levels for the `compile` function."""
    return request.param


if os.path.isdir(os.path.join(os.path.dirname(__file__), '_data')):
    params = os.listdir(os.path.join(os.path.dirname(__file__), '_data'))
else:
    params = []


@pytest.fixture(
    params=params,
    ids=lambda qasm_file: os.path.splitext(os.path.basename(qasm_file))[0],
)
def medium_qasm_file(request: Any) -> str:
    """Provide location of a medium qasm file."""
    cur_dir = os.path.dirname(__file__)
    path = os.path.join(cur_dir, '_data')
    return os.path.join(path, request.param)
