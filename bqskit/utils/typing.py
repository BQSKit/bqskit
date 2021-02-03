"""This module implements many helper functions to check types."""


import logging

from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any, Optional


_logger = logging.getLogger( __name__ )


def is_iterable( test_variable: Iterable[Any] ) -> bool:
    """Returns true if test_variable is an iterable object."""
    try:
        iterator = iter(test_variable)
        return True
    except TypeError:
        _logger.debug("Invalid iterable.")
        return False


def is_sequence( test_variable: Sequence[Any] ) -> bool:
    """Returns true if test_variable is a sequence."""
    if isinstance( test_variable, Sequence ):
        return True
    else:
        _logger.debug("Invalid sequence.")
        return False


def is_valid_location(location: Iterable[int], num_qudits: Optional[int] = None) -> bool:
    """
    Determines if the sequence of qudits form a valid location.
    A valid location is a set of qubit indices (integers) that
    are greater than or equal to zero, and if num_qudits is specified, 
    less than num_qudits.

    Args:
        location (Iterable[int]): The location to check.

        num_qudits (Optional[int]): The total number of qudits.
            All qudit indices should be less than this. If None,
            don't check.
    
    Returns:
        (bool): True if the location is valid.
    """
    if not is_iterable( location ):
        return False
    
    if not all( [ isinstance( qudit, int ) for qudit in location ] ):
        _logger.debug( "Location is not a tuple of ints." )
        return False
    
    if len( location ) != len( set( location ) ):
        _logger.debug( "Location has duplicates." )
        return False
    
    if not all( [ qudit >= 0 for qudit in location ] ):
        _logger.debug( "Location invalid; qudit indices must be nonnegative." )
        return False

    if num_qudits is not None:
        if not all( [ qudit < num_qudits for qudit in location ] ):
            _logger.debug( "Location has an erroneously large qudit." )
            return False
    
    return True
    