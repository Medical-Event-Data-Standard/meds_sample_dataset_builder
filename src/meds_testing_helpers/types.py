from typing import Annotated, Any

import numpy as np
from annotated_types import Ge, Gt, Le

NUM = int | float
POSITIVE_NUM = Annotated[NUM, Gt(0)]
POSITIVE_INT = Annotated[int, Gt(0)]
NON_NEGATIVE_NUM = Annotated[NUM, Ge(0)]
NON_NEGATIVE_INT = Annotated[int, Ge(0)]
PROPORTION = Annotated[NUM, Ge(0), Le(1)]
POSITIVE_TIMEDELTA = Annotated[np.timedelta64, Gt(0)]


def is_NUM(x: Any) -> bool:
    """Check if x is a number.

    Examples:
        >>> is_NUM(1)
        True
        >>> is_NUM(1.0)
        True
        >>> is_NUM("a")
        False
    """
    return isinstance(x, int | float)


def is_POSITIVE_NUM(x: Any) -> bool:
    """Check if x is a positive number.

    Examples:
        >>> is_POSITIVE_NUM(1)
        True
        >>> is_POSITIVE_NUM(0)
        False
        >>> is_POSITIVE_NUM(1.0)
        True
        >>> is_POSITIVE_NUM("foo")
        False
    """
    return is_NUM(x) and x > 0


def is_POSITIVE_INT(x: Any) -> bool:
    """Check if x is a positive integer.

    Examples:
        >>> is_POSITIVE_INT(1)
        True
        >>> is_POSITIVE_INT(0)
        False
        >>> is_POSITIVE_INT(1.0)
        False
    """
    return isinstance(x, int) and x > 0


def is_NON_NEGATIVE_NUM(x: Any) -> bool:
    """Check if x is a non-negative number.

    Examples:
        >>> is_NON_NEGATIVE_NUM(1)
        True
        >>> is_NON_NEGATIVE_NUM(0)
        True
        >>> is_NON_NEGATIVE_NUM(-1)
        False
        >>> is_NON_NEGATIVE_NUM(1.0)
        True
    """
    return is_NUM(x) and x >= 0


def is_NON_NEGATIVE_INT(x: Any) -> bool:
    """Check if x is a non-negative integer.

    Examples:
        >>> is_NON_NEGATIVE_INT(1)
        True
        >>> is_NON_NEGATIVE_INT(0)
        True
        >>> is_NON_NEGATIVE_INT(-1)
        False
        >>> is_NON_NEGATIVE_INT(1.0)
        False
    """
    return isinstance(x, int) and x >= 0


def is_PROPORTION(x: Any) -> bool:
    """Check if x is a proportion (between 0 and 1 inclusive).

    Examples:
        >>> is_PROPORTION(1)
        True
        >>> is_PROPORTION(0)
        True
        >>> is_PROPORTION(0.5)
        True
        >>> is_PROPORTION(-1)
        False
        >>> is_PROPORTION("foo")
        False
    """
    return is_NUM(x) and 0 <= x <= 1


def is_POSITIVE_TIMEDELTA(x: Any) -> bool:
    """Check if x is a positive timedelta.

    Examples:
        >>> is_POSITIVE_TIMEDELTA(np.timedelta64(1, "s"))
        True
        >>> is_POSITIVE_TIMEDELTA(np.timedelta64(0, "s"))
        False
        >>> is_POSITIVE_TIMEDELTA(np.timedelta64(1, "m"))
        True
        >>> is_POSITIVE_TIMEDELTA(np.timedelta64(-1, "s"))
        False
        >>> is_POSITIVE_TIMEDELTA(1)
        False
    """
    return bool(isinstance(x, np.timedelta64) and (x > 0))
