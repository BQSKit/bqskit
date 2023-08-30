"""This module tests the logging features supported by BQSKit Runtime."""
from __future__ import annotations

import logging
from io import StringIO

import pytest

from bqskit import enable_logging
from bqskit.compiler import BasePass
from bqskit.compiler import Compiler
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.runtime import get_runtime


async def nested1() -> None:
    logging.getLogger('bqskit').info('bqskit_level1')
    logging.getLogger('dummy2').info('dummy2_level1')
    await get_runtime().submit(nested2)


async def nested2() -> None:
    logging.getLogger('bqskit').info('bqskit_level2')
    logging.getLogger('dummy2').info('dummy2_level2')
    await get_runtime().submit(nested3)


async def nested3() -> None:
    logging.getLogger('bqskit').info('bqskit_level3')
    logging.getLogger('dummy2').info('dummy2_level3')


class TestInfoPass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        logging.getLogger('bqskit').info('bqskit_info')
        logging.getLogger('bqskit.dummy').info('bqskit_dummy_info')
        logging.getLogger('dummy2').info('dummy2_info')


class TestDebugPass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        logging.getLogger('bqskit').debug('bqskit_debug')
        logging.getLogger('bqskit.dummy').debug('bqskit_dummy_debug')
        logging.getLogger('dummy2').debug('dummy2_debug')


class TestNestedLogPass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        logging.getLogger('bqskit').info('bqskit_level0')
        logging.getLogger('dummy2').info('dummy2_level0')
        await get_runtime().submit(nested1)


def test_using_enable_logging(server_compiler: Compiler) -> None:
    enable_logging()
    logger = logging.getLogger('bqskit')
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    server_compiler.compile(Circuit(1), [TestInfoPass(), TestDebugPass()])
    log = stream.getvalue()
    assert 'bqskit_info' in log
    assert 'bqskit_dummy_info' in log
    assert 'dummy2_info' not in log
    assert 'bqskit_debug' not in log
    assert 'bqskit_dummy_debug' not in log
    assert 'dummy2_debug' not in log
    logger.removeHandler(handler)
    logger.setLevel(logging.WARNING)


def test_using_enable_logging_verbose(server_compiler: Compiler) -> None:
    enable_logging(True)
    logger = logging.getLogger('bqskit')
    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    server_compiler.compile(Circuit(1), [TestInfoPass(), TestDebugPass()])
    log = handler.stream.getvalue()
    assert 'bqskit_info' in log
    assert 'bqskit_dummy_info' in log
    assert 'dummy2_info' not in log
    assert 'bqskit_debug' in log
    assert 'bqskit_dummy_debug' in log
    assert 'dummy2_debug' not in log
    logger.removeHandler(handler)
    logger.setLevel(logging.WARNING)


def test_using_standard_logging(server_compiler: Compiler) -> None:
    logger = logging.getLogger('bqskit')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    server_compiler.compile(Circuit(1), [TestInfoPass(), TestDebugPass()])
    log = handler.stream.getvalue()
    assert 'bqskit_info' in log
    assert 'bqskit_dummy_info' in log
    assert 'dummy2_info' not in log
    assert 'bqskit_debug' in log
    assert 'bqskit_dummy_debug' in log
    assert 'dummy2_debug' not in log
    logger.removeHandler(handler)
    logger.setLevel(logging.WARNING)


def test_using_standard_logging_specific(server_compiler: Compiler) -> None:
    logger = logging.getLogger('bqskit.dummy')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    server_compiler.compile(Circuit(1), [TestInfoPass(), TestDebugPass()])
    log = handler.stream.getvalue()
    assert 'bqskit_info' not in log
    assert 'bqskit_dummy_info' in log
    assert 'dummy2_info' not in log
    assert 'bqskit_debug' not in log
    assert 'bqskit_dummy_debug' in log
    assert 'dummy2_debug' not in log
    logger.removeHandler(handler)
    logger.setLevel(logging.WARNING)


def test_using_external_logging(server_compiler: Compiler) -> None:
    logger = logging.getLogger('dummy2')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    server_compiler.compile(Circuit(1), [TestInfoPass(), TestDebugPass()])
    log = handler.stream.getvalue()
    assert 'bqskit_info' not in log
    assert 'bqskit_dummy_info' not in log
    assert 'dummy2_info' in log
    assert 'bqskit_debug' not in log
    assert 'bqskit_dummy_debug' not in log
    assert 'dummy2_debug' in log
    logger.removeHandler(handler)
    logger.setLevel(logging.WARNING)


@pytest.mark.parametrize('level', [-1, 0, 1, 2, 3, 4])
def test_limiting_nested_calls_enable_logging(
    server_compiler: Compiler,
    level: int,
) -> None:
    enable_logging()
    logger = logging.getLogger('bqskit')
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    server_compiler.compile(
        Circuit(1),
        [TestNestedLogPass()],
        max_logging_depth=level,
    )

    log = stream.getvalue()
    if level != -1:
        expected_prints = list(range(0, min(4, level + 1)))
    else:
        expected_prints = [0, 1, 2, 3]
    unexpected_prints = [x for x in [0, 1, 2, 3] if x not in expected_prints]
    assert all(f'bqskit_level{i}' in log for i in expected_prints)
    assert all(f'bqskit_level{i}' not in log for i in unexpected_prints)
    assert all(f'dummy2_level{i}' not in log for i in [0, 1, 2, 3])
    logger.removeHandler(handler)
    logger.setLevel(logging.WARNING)


@pytest.mark.parametrize('level', [-1, 0, 1, 2, 3, 4])
def test_limiting_nested_calls_external_logging(
    server_compiler: Compiler,
    level: int,
) -> None:
    logger = logging.getLogger('dummy2')
    logger.setLevel(logging.INFO)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    server_compiler.compile(
        Circuit(1),
        [TestNestedLogPass()],
        max_logging_depth=level,
    )

    log = stream.getvalue()
    if level != -1:
        expected_prints = list(range(0, min(4, level + 1)))
    else:
        expected_prints = [0, 1, 2, 3]
    unexpected_prints = [x for x in [0, 1, 2, 3] if x not in expected_prints]
    assert all(f'dummy2_level{i}' in log for i in expected_prints)
    assert all(f'dummy2_level{i}' not in log for i in unexpected_prints)
    assert all(f'bqskit_level{i}' not in log for i in [0, 1, 2, 3])
    logger.removeHandler(handler)
    logger.setLevel(logging.WARNING)
