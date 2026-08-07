from __future__ import annotations

import os
from typing import Any

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test collected in this directory as slow."""
    here = os.path.dirname(__file__)
    for item in items:
        if str(item.path).startswith(here):
            item.add_marker(pytest.mark.slow)


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
