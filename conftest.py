"""Repo-root pytest configuration.

Lives at the repo root so the hook below applies
no matter what path pytest is invoked with.
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Deselect slow tests unless the run doesn't need filtering."""

    # an explicit -m expression was given; use it
    if config.option.markexpr:
        return

    # explicit path was passed in; run everything there
    if config.args_source is pytest.Config.ArgsSource.ARGS:
        return

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        (deselected if 'slow' in item.keywords else selected).append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
        print(
            f'\n{len(deselected)} slow test(s) excluded by default (no '
            'path or marker given). Pass a path (e.g. `pytest '
            'tests/compiler/compile`) or `-m "slow or not slow"` to '
            'include them.',
        )
