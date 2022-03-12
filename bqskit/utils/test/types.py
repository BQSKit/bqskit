"""This module contains functions to generate strategies from annotations."""
from __future__ import annotations

import collections
import inspect
import sys
from itertools import chain
from itertools import combinations
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Sequence

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import complex_number_dtypes
from hypothesis.extra.numpy import floating_dtypes
from hypothesis.extra.numpy import from_dtype
from hypothesis.strategies import booleans
from hypothesis.strategies import complex_numbers
from hypothesis.strategies import data
from hypothesis.strategies import dictionaries
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import iterables
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import one_of
from hypothesis.strategies import SearchStrategy
from hypothesis.strategies import sets
from hypothesis.strategies import text
from hypothesis.strategies import tuples

from bqskit.utils.test.strategies import circuit_location_likes
from bqskit.utils.test.strategies import circuit_locations
from bqskit.utils.test.strategies import circuit_points
from bqskit.utils.test.strategies import circuit_regions
from bqskit.utils.test.strategies import circuits
from bqskit.utils.test.strategies import cycle_intervals
from bqskit.utils.test.strategies import everything_except
from bqskit.utils.test.strategies import gates
from bqskit.utils.test.strategies import operations
from bqskit.utils.test.strategies import unitaries
from bqskit.utils.test.strategies import unitary_likes


def _powerset(iterable: Iterable[Any]) -> Iterable[Any]:
    """
    Calculate the powerset of an iterable.

    Examples:

        >>> list(powerset([1,2,3]))
        ... [() (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)]

    References:

        https://stackoverflow.com/questions/18035595/powersets-in-python-using-
        itertools.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)))


def _split_generic_arguments(args: str) -> list[str]:
    """Split a generic's type arguments up."""
    comma_indices = []
    num_open_brackets = 0
    for i, char in enumerate(args):
        if char == '[':
            num_open_brackets += 1
        elif char == ']':
            num_open_brackets -= 1
        elif char == ',' and num_open_brackets == 0:
            comma_indices.append(i)

    if len(comma_indices) == 0:
        return [args]

    to_return: list[str] = []
    last_index = 0
    for comma_index in comma_indices:
        to_return.append(args[last_index: comma_index])
        last_index = comma_index + 1
    to_return.append(args[last_index:])
    return to_return


def type_annotation_to_valid_strategy(annotation: str) -> SearchStrategy[Any]:
    """Convert a type annotation into a hypothesis strategy."""
    strategies: list[SearchStrategy[Any]] = []

    annotation = annotation.replace('RealVector', 'Sequence[float]')

    for type_str in annotation.split('|'):
        type_str = type_str.strip()

        if type_str == 'None':
            strategies.append(just(None))

        elif type_str == 'int':
            strategies.append(integers())

        elif type_str == 'float':
            strategies.append(floats())
            strategies.append(floating_dtypes().flatmap(from_dtype))

        elif type_str == 'complex':
            strategies.append(complex_numbers())
            strategies.append(complex_number_dtypes().flatmap(from_dtype))

        elif type_str == 'bool':
            strategies.append(booleans())

        elif type_str == 'str':
            strategies.append(text())

        elif type_str == 'Any':
            strategies.append(just(None))

        elif type_str.lower().startswith('tuple'):
            inner_strategies = []
            for arg in _split_generic_arguments(type_str[6:-1]):
                inner_strategies.append(type_annotation_to_valid_strategy(arg))
            strategies.append(tuples(*inner_strategies))

        elif type_str.lower().startswith('dict'):
            args = _split_generic_arguments(type_str[5:-1])
            key_strat = type_annotation_to_valid_strategy(args[0])
            val_strat = type_annotation_to_valid_strategy(args[1])
            strategies.append(dictionaries(key_strat, val_strat))

        elif type_str.lower().startswith('mapping'):
            args = _split_generic_arguments(type_str[8:-1])
            key_strat = type_annotation_to_valid_strategy(args[0])
            val_strat = type_annotation_to_valid_strategy(args[1])
            strategies.append(dictionaries(key_strat, val_strat))

        elif type_str.lower().startswith('list'):
            arg_strat = type_annotation_to_valid_strategy(type_str[5:-1])
            strategies.append(lists(arg_strat))

        elif type_str.lower().startswith('set'):
            arg_strat = type_annotation_to_valid_strategy(type_str[4:-1])
            strategies.append(sets(arg_strat))

        elif type_str.lower().startswith('sequence'):
            arg_strat = type_annotation_to_valid_strategy(type_str[9:-1])
            strategies.append(lists(arg_strat))

        elif type_str.lower().startswith('iterable'):
            arg_strat = type_annotation_to_valid_strategy(type_str[9:-1])
            strategies.append(iterables(arg_strat))

        elif type_str.lower().startswith('intervallike'):
            strat = type_annotation_to_valid_strategy('Tuple[int, int]')
            strategies.append(strat)
            strategies.append(cycle_intervals())

        elif type_str.lower().startswith('cycleinterval'):
            strategies.append(cycle_intervals())

        elif type_str.lower().startswith('circuitpointlike'):
            strat = type_annotation_to_valid_strategy('Tuple[int, int]')
            strategies.append(strat)
            strategies.append(circuit_points())

        elif type_str.lower().startswith('circuitpoint'):
            strategies.append(circuit_points())

        elif type_str.lower().startswith('circuitregionlike'):
            strat = type_annotation_to_valid_strategy('dict[int, IntervalLike]')
            strategies.append(strat)
            strategies.append(circuit_regions())

        elif type_str.lower().startswith('circuitregion'):
            strategies.append(circuit_regions())

        elif type_str.lower().startswith('unitarylike'):
            strategies.append(unitary_likes())

        elif type_str.lower().startswith('unitarymatrix'):
            strategies.append(unitaries())

        elif type_str.lower().startswith('gate'):
            strategies.append(gates())

        elif type_str.lower().startswith('operation'):
            strategies.append(operations())

        elif type_str.lower().startswith('circuitlocationlike'):
            strategies.append(circuit_locations())

        elif type_str.lower().startswith('circuitlocation'):
            strategies.append(circuit_location_likes())

        elif type_str.lower().startswith('circuit'):
            strategies.append(circuits(max_gates=1))

        else:
            raise ValueError(f'Cannot generate strategy for type: {type_str}')

    return one_of(strategies)


def type_annotation_to_invalid_strategy(annotation: str) -> SearchStrategy[Any]:
    """Convert a type annotation into an invalid hypothesis strategy."""
    strategies: list[SearchStrategy[Any]] = []
    types_to_avoid: set[type] = set()
    tuple_valids: dict[int, set[SearchStrategy[Any]]] = {}
    tuple_invalids: dict[int, set[SearchStrategy[Any]]] = {}
    dict_key_valids: set[SearchStrategy[Any]] = set()
    dict_key_invalids: set[SearchStrategy[Any]] = set()
    dict_val_valids: set[SearchStrategy[Any]] = set()
    dict_val_invalids: set[SearchStrategy[Any]] = set()
    list_invalids: set[SearchStrategy[Any]] = set()
    set_invalids: set[SearchStrategy[Any]] = set()
    iterable_invalids: set[SearchStrategy[Any]] = set()

    annotation = annotation.replace('RealVector', 'Sequence[float]')

    for type_str in annotation.split('|'):
        type_str = type_str.strip()

        if type_str == 'None':
            types_to_avoid.add(type(None))

        elif type_str == 'int':
            types_to_avoid.add(int)
            types_to_avoid.add(np.byte)
            types_to_avoid.add(np.short)
            types_to_avoid.add(np.intc)
            types_to_avoid.add(np.longlong)
            types_to_avoid.add(np.int8)
            types_to_avoid.add(np.int16)
            types_to_avoid.add(np.int32)
            types_to_avoid.add(np.int64)

        elif type_str == 'float':
            types_to_avoid.add(float)
            types_to_avoid.add(np.half)
            types_to_avoid.add(np.single)
            types_to_avoid.add(np.double)
            types_to_avoid.add(np.longdouble)
            types_to_avoid.add(np.float32)
            types_to_avoid.add(np.float64)

        elif type_str == 'complex':
            types_to_avoid.add(complex)
            types_to_avoid.add(np.csingle)
            types_to_avoid.add(np.cdouble)
            types_to_avoid.add(np.clongdouble)
            types_to_avoid.add(np.complex64)
            types_to_avoid.add(np.complex128)

        elif type_str == 'bool':
            types_to_avoid.add(bool)
            types_to_avoid.add(np.bool_)

        elif type_str == 'str':
            types_to_avoid.add(str)

        elif type_str == 'Any':
            continue

        elif type_str.lower().startswith('tuple'):
            args = _split_generic_arguments(type_str[6:-1])
            if len(args) not in tuple_valids:
                tuple_valids[len(args)] = set()
                tuple_invalids[len(args)] = set()

            for arg in args:
                valid_strat = type_annotation_to_valid_strategy(arg)
                invalid_strat = type_annotation_to_invalid_strategy(arg)
                tuple_valids[len(args)].add(valid_strat)
                tuple_invalids[len(args)].add(invalid_strat)

            types_to_avoid.add(tuple)

        elif type_str.lower().startswith('dict'):
            args = _split_generic_arguments(type_str[5:-1])
            dict_key_valids.add(type_annotation_to_valid_strategy(args[0]))
            dict_key_invalids.add(type_annotation_to_valid_strategy(args[1]))
            dict_val_valids.add(type_annotation_to_invalid_strategy(args[0]))
            dict_val_invalids.add(type_annotation_to_invalid_strategy(args[1]))
            types_to_avoid.add(dict)
            types_to_avoid.add(map)

        elif type_str.lower().startswith('mapping'):
            args = _split_generic_arguments(type_str[8:-1])
            dict_key_valids.add(type_annotation_to_valid_strategy(args[0]))
            dict_key_invalids.add(type_annotation_to_valid_strategy(args[1]))
            dict_val_valids.add(type_annotation_to_invalid_strategy(args[0]))
            dict_val_invalids.add(type_annotation_to_invalid_strategy(args[1]))
            types_to_avoid.add(dict)
            types_to_avoid.add(map)

        elif type_str.lower().startswith('list'):
            arg_strat = type_annotation_to_invalid_strategy(type_str[5:-1])
            list_invalids.add(arg_strat)
            types_to_avoid.add(list)

        elif type_str.lower().startswith('set'):
            arg_strat = type_annotation_to_invalid_strategy(type_str[4:-1])
            set_invalids.add(arg_strat)
            types_to_avoid.add(set)
            types_to_avoid.add(collections.abc.MutableSet)

        elif type_str.lower().startswith('sequence'):
            arg_strat = type_annotation_to_invalid_strategy(type_str[9:-1])
            list_invalids.add(arg_strat)
            types_to_avoid.add(Sequence)
            types_to_avoid.add(list)
            types_to_avoid.add(tuple)
            types_to_avoid.add(bytearray)
            types_to_avoid.add(bytes)

        elif type_str.lower().startswith('iterable'):
            arg_strat = type_annotation_to_invalid_strategy(type_str[9:-1])
            iterable_invalids.add(arg_strat)
            types_to_avoid.add(Sequence)
            types_to_avoid.add(list)
            types_to_avoid.add(tuple)
            types_to_avoid.add(Iterable)
            types_to_avoid.add(set)
            types_to_avoid.add(frozenset)
            types_to_avoid.add(dict)
            types_to_avoid.add(str)
            types_to_avoid.add(bytearray)
            types_to_avoid.add(bytes)
            types_to_avoid.add(collections.abc.MutableSet)
            types_to_avoid.add(enumerate)
            types_to_avoid.add(map)
            types_to_avoid.add(range)
            types_to_avoid.add(reversed)

        elif type_str.lower().startswith('intervallike'):
            types_to_avoid.add(tuple)

        elif type_str.lower().startswith('cycleinterval'):
            continue

        elif type_str.lower().startswith('circuitpointlike'):
            types_to_avoid.add(tuple)

        elif type_str.lower().startswith('circuitpoint'):
            continue

        elif type_str.lower().startswith('circuitregionlike'):
            types_to_avoid.add(dict)

        elif type_str.lower().startswith('circuitregion'):
            continue

        elif type_str.lower().startswith('circuitlocationlike'):
            types_to_avoid.add(int)
            types_to_avoid.add(Sequence)
            types_to_avoid.add(Iterable)
            types_to_avoid.add(list)
            types_to_avoid.add(tuple)
            types_to_avoid.add(collections.abc.MutableSet)
            types_to_avoid.add(enumerate)
            types_to_avoid.add(range)
            types_to_avoid.add(reversed)

        elif type_str.lower().startswith('circuitlocation'):
            continue

        elif type_str.lower().startswith('unitarylike'):
            types_to_avoid.add(np.ndarray)

        elif type_str.lower().startswith('unitarymatrix'):
            continue

        elif type_str.lower().startswith('gate'):
            continue

        elif type_str.lower().startswith('operation'):
            continue

        elif type_str.lower().startswith('circuit'):
            continue

        else:
            raise ValueError(f'Cannot generate strategy for type: {type_str}')

    strategies.append(everything_except(tuple(types_to_avoid)))

    for tuple_len in tuple_valids:
        for valid_set in _powerset(list(range(tuple_len))):  # (), (0,), (1,)
            strategy_builder = []
            for i in range(tuple_len):
                if i in valid_set:
                    strat = one_of(list(tuple_valids[tuple_len]))
                    strategy_builder.append(strat)
                else:
                    strat = one_of(list(tuple_invalids[tuple_len]))
                    strategy_builder.append(strat)
            strategies.append(tuples(*strategy_builder))

    if len(dict_val_invalids) > 0:
        strategies.append(
            dictionaries(
                one_of(list(dict_key_valids)),
                one_of(list(dict_val_invalids)),
                min_size=1,
            ),
        )
        strategies.append(
            dictionaries(
                one_of(list(dict_key_invalids)),
                one_of(list(dict_val_valids)),
                min_size=1,
            ),
        )
        strategies.append(
            dictionaries(
                one_of(list(dict_key_invalids)),
                one_of(list(dict_val_invalids)),
                min_size=1,
            ),
        )

    if len(list_invalids) > 0:
        strategies.append(lists(one_of(list(list_invalids)), min_size=1))
    if len(set_invalids) > 0:
        strategies.append(sets(one_of(list(set_invalids)), min_size=1))
    if len(iterable_invalids) > 0:
        strategies.append(
            iterables(
                one_of(
                    list(iterable_invalids),
                ),
                min_size=1,
            ),
        )

    return one_of(strategies)


def invalid_type_test(
        func_to_test: Callable[..., Any],
        other_allowed_errors: list[type] = [],
) -> Callable[..., Callable[..., None]]:
    """
    Decorator to generate invalid type tests.

    A valid type test ensures that a function called with incorrect types
    does raise a TypeError.

    Examples:
        >>> class Foo:
        ...     def foo(self, x: int, y: int) -> None:
        ...         if not is_integer(x):
        ...             raise TypeError("")
        ...         if not is_integer(y):
        ...             raise TypeError("")

        >>> class TestFoo:
        ...     @invalid_type_test(Foo().foo)
        ...     def test_foo_invalid_type(self) -> None:
        ...         pass

        >>> @invalid_type_test(Foo().foo)
        ... def test_foo_invalid_type(self) -> None:
        ...     pass
    """
    if sys.version_info[0] == 3 and sys.version_info[1] < 9:
        return lambda x: x

    valids = []
    invalids = []
    for id, param in inspect.signature(func_to_test).parameters.items():
        if param.annotation == inspect._empty:
            raise ValueError(
                'Need type annotation to generate invalid type tests.',
            )

        valids.append(type_annotation_to_valid_strategy(param.annotation))
        invalids.append(type_annotation_to_invalid_strategy(param.annotation))

    strategies = []
    for valid_set in _powerset(list(range(len(valids)))):
        strategy_builder = []
        for i in range(len(valids)):
            if i in valid_set:
                strategy_builder.append(valids[i])
            else:
                strategy_builder.append(invalids[i])
        strategies.append(tuples(*strategy_builder))

    def inner(f: Callable[..., Any]) -> Callable[..., None]:
        if 'self' in inspect.signature(f).parameters:
            @pytest.mark.parametrize('strategy', strategies)
            @given(data=data())
            def invalid_type_test(self: Any, strategy: Any, data: Any) -> None:
                args = data.draw(strategy)
                with pytest.raises((TypeError,) + tuple(other_allowed_errors)):
                    func_to_test(*args)

            return invalid_type_test
        else:
            @pytest.mark.parametrize('strategy', strategies)
            @given(data=data())
            def invalid_type_test(strategy: Any, data: Any) -> None:
                args = data.draw(strategy)
                with pytest.raises((TypeError,) + tuple(other_allowed_errors)):
                    func_to_test(*args)

            return invalid_type_test

    return inner


def valid_type_test(
        func_to_test: Callable[..., Any],
) -> Callable[..., Callable[..., None]]:
    """
    Decorator to generate valid type tests.

    A valid type test ensures that a function called with correct types
    does not raise a TypeError.

    Examples:
        >>> class Foo:
        ...     def foo(self, x: int, y: int) -> None:
        ...         if not is_integer(x):
        ...             raise TypeError("")
        ...         if not is_integer(y):
        ...             raise TypeError("")

        >>> class TestFoo:
        ...     @valid_type_test(Foo().foo)
        ...     def test_foo_valid_type(self) -> None:
        ...         pass

        >>> @valid_type_test(Foo().foo)
        ... def test_foo_valid_type(self) -> None:
        ...     pass
    """
    if sys.version_info[0] == 3 and sys.version_info[1] < 9:
        return lambda x: x

    strategies = []
    for id, param in inspect.signature(func_to_test).parameters.items():
        if param.annotation == inspect._empty:
            raise ValueError(
                'Need type annotation to generate invalid type tests.',
            )

        strategies.append(type_annotation_to_valid_strategy(param.annotation))
    strategy = tuples(*strategies)

    def inner(f: Callable[..., Any]) -> Callable[..., None]:
        if 'self' in inspect.signature(f).parameters:
            @given(data=strategy)
            def valid_type_test(self: Any, data: Any) -> None:
                try:
                    func_to_test(*data)
                except TypeError:
                    assert False, 'Valid types caused TypeError.'
                except Exception:
                    pass

            return valid_type_test
        else:
            @given(data=strategy)
            def valid_type_test(data: Any) -> None:
                try:
                    func_to_test(*data)
                except TypeError:
                    assert False, 'Valid types caused TypeError.'
                except Exception:
                    pass

            return valid_type_test

    return inner
