from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(params=[1, 2, 3])
def optimization_level(request: Any) -> int:
    """All valid optimization_levels for the `compile` function."""
    return request.param
