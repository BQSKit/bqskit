"""This module tests RuntimeTask functionality."""
from __future__ import annotations

import pytest

from bqskit.runtime.task import RuntimeTask


class AwaitableTestObject:
    def __await__(self) -> str:
        yield 'test1'
        assert (yield 'test2') == 'test_send'
        assert (yield 'test3') is None
        assert (yield 'test4') is None
        return 'test_return'


async def coro_test():
    assert (await AwaitableTestObject()) == 'test_return'
    return 'test_complete'


def test_uninit_task_step_errors() -> None:
    with pytest.raises(RuntimeError):
        task = RuntimeTask((coro_test, tuple(), {}), (0, 0, 0), 0, tuple())
        task.step()


def test_task_step() -> None:
    task = RuntimeTask((coro_test, tuple(), {}), (0, 0, 0), 0, tuple())
    task.start()
    assert task.step() == 'test1'
    assert task.step() == 'test2'
    task.send = 'test_send'
    assert task.step() == 'test3'
    task.send = None
    assert task.step() == 'test4'

    try:
        task.step()
    except StopIteration as e:
        assert e.value == 'test_complete'
    else:
        assert False, 'StopIteration next called.'


def not_coro_test() -> None:
    return 'test_complete'


def test_task_step_no_coro() -> None:
    task = RuntimeTask((not_coro_test, tuple(), {}), (0, 0, 0), 0, tuple())
    task.start()
    try:
        task.step()
    except StopIteration as e:
        assert e.value == 'test_complete'
    else:
        assert False, 'StopIteration next called.'


def test_task_descendant_of() -> None:
    task = RuntimeTask(
        (coro_test, tuple(), {}),
        (0, 0, 0),
        0,
        ((0, 0, 1), (0, 0, 2)),
    )
    assert task.is_descendant_of((0, 0, 0))
    assert task.is_descendant_of((0, 0, 1))
    assert task.is_descendant_of((0, 0, 2))
    assert not task.is_descendant_of((1, 0, 0))
