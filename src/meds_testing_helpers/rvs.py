import abc
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import pytimeparse

from .types import (
    NON_NEGATIVE_NUM,
    POSITIVE_INT,
    POSITIVE_NUM,
    POSITIVE_TIMEDELTA,
    PROPORTION,
    is_NON_NEGATIVE_NUM,
    is_POSITIVE_INT,
    is_POSITIVE_NUM,
    is_POSITIVE_TIMEDELTA,
    is_PROPORTION,
)

T = TypeVar("T")


class Stringified(Generic[T], abc.ABC):
    @classmethod
    def _to_str(cls, x: Any) -> str:
        return str(x)

    @classmethod
    @abc.abstractmethod
    def _from_str(cls, x: T) -> Any:  # pragma: no cover
        raise NotImplementedError

    @property
    def _X(self) -> np.ndarray:
        return np.array([self._from_str(x) for x in self.X])

    @abc.abstractmethod
    def _validate(self):  # pragma: no cover
        raise NotImplementedError

    def __post_init__(self):
        if all(isinstance(x, str) for x in self.X):
            try:
                self._X
            except Exception as e:
                fails = []
                for x in self.X:
                    try:
                        self._from_str(x)
                    except Exception:
                        fails.append(x)

                if len(fails) > 5:
                    fails_str = ", ".join(fails[:5]) + ", ... (total: {len(fails)})"
                else:
                    fails_str = ", ".join(fails)
                raise ValueError(f"All elements should be convertible strings. Got: {fails_str}") from e
            self._validate(self._X)
        else:
            self._validate(self.X)

            str_X = []
            fails = []
            for x in self.X:
                try:
                    str_X.append(self._to_str(x))
                except Exception:
                    fails.append(x)

            if fails:
                if len(fails) > 5:
                    fails_str = ", ".join(str(x) for x in fails[:5]) + ", ... (total: {len(fails)})"
                else:
                    fails_str = ", ".join(str(x) for x in fails)

                raise ValueError(f"All elements should be convertible to strings. Got {fails_str}")
            else:
                self.X = str_X

        super().__post_init__()


@dataclass
class DiscreteGenerator:
    """A class to generate random numbers from a list of options with given frequencies.

    This is largely just for type safety and to ease specification of the various things that need to be
    sampled to generate a dataset.

    Attributes:
        X: The list of options to sample from.
        freq: The frequency of each option. If None, all options are equally weighted.

    Raises:
        ValueError: If the frequencies are not all positive, the lengths of X and freq are not equal, or no
            options are provided.

    Examples:
        >>> x = DiscreteGenerator([1, 2, 3], [1, 2, 3])
        >>> rng = np.random.default_rng(1)
        >>> x.rvs(10, rng)
        array([3, 3, 1, 3, 2, 2, 3, 2, 3, 1])
        >>> rng = np.random.default_rng(1)
        >>> x.rvs(10, rng)
        array([3, 3, 1, 3, 2, 2, 3, 2, 3, 1])
        >>> x.rvs(10, rng)
        array([3, 3, 2, 3, 2, 2, 1, 2, 2, 2])
        >>> rng = np.random.default_rng(1)
        >>> DiscreteGenerator([1, 2, 3]).rvs(10, rng)
        array([2, 3, 1, 3, 1, 2, 3, 2, 2, 1])
        >>> rng = np.random.default_rng(1)
        >>> DiscreteGenerator(['a', 'b', 'c']).rvs(10, rng)
        array(['b', 'c', 'a', 'c', 'a', 'b', 'c', 'b', 'b', 'a'], dtype='<U1')
        >>> DiscreteGenerator([1, 2], [-1, 1])
        Traceback (most recent call last):
            ...
        ValueError: All frequencies should be positive.
        >>> DiscreteGenerator([1, 2], [1, 2, 3])
        Traceback (most recent call last):
            ...
        ValueError: Equal numbers of frequencies and options must be provided. Got 3 and 2.
        >>> DiscreteGenerator([])
        Traceback (most recent call last):
            ...
        ValueError: At least one option should be provided. Got length 0.
    """

    X: list[Any]
    freq: list[NON_NEGATIVE_NUM] | None = None

    def __post_init__(self):
        if self.freq is None:
            self.freq = [1] * len(self.X)
        if not all(is_NON_NEGATIVE_NUM(f) for f in self.freq):
            raise ValueError("All frequencies should be positive.")
        if len(self.freq) != len(self.X):
            raise ValueError(
                "Equal numbers of frequencies and options must be provided. "
                f"Got {len(self.freq)} and {len(self.X)}."
            )
        if len(self.freq) == 0:
            raise ValueError("At least one option should be provided. Got length 0.")

    @property
    def _X(self) -> np.ndarray:
        return np.array(self.X)

    @property
    def p(self) -> np.ndarray:
        return np.array(self.freq) / sum(self.freq)

    def rvs(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self._X, size=size, p=self.p, replace=True)


class DatetimeGenerator(Stringified[np.datetime64], DiscreteGenerator):
    """A class to generate random datetimes.

    This merely applies type-checking to the DiscreteGenerator class.

    Attributes:
        X: The list of datetimes to sample from.
        freq: The frequency of each option. If None, all options are equally weighted.

    Raises:
        ValueError: In addition to the base class errors, if any of the options are not datetimes.

    Examples:
        >>> rng = np.random.default_rng(1)
        >>> DatetimeGenerator([np.datetime64("2021-01-01"), np.datetime64("2022-02-02")]).rvs(10, rng)
        array(['2022-02-02', '2022-02-02', '2021-01-01', '2022-02-02',
               '2021-01-01', '2021-01-01', '2022-02-02', '2021-01-01',
               '2022-02-02', '2021-01-01'], dtype='datetime64[D]')
        >>> DatetimeGenerator([1, 2])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be datetimes. Got [1, 2].
    """

    X: list[str | np.datetime64]

    @classmethod
    def _from_str(cls, x: str) -> np.datetime64:
        return np.datetime64(x)

    def _validate(self, X: list[Any]):
        if not all(isinstance(x, np.datetime64) for x in X):
            fails = [x for x in X if not isinstance(x, np.datetime64)]
            raise ValueError(f"All elements should be datetimes. Got {fails}.")


class PositiveTimeDeltaGenerator(Stringified[np.timedelta64], DiscreteGenerator):
    """A class to generate random positive timedeltas.

    This merely applies type-checking to the DiscreteGenerator class.

    Attributes:
        X: The list of timedeltas to sample from.
        freq: The frequency of each option. If None, all options are equally weighted.

    Raises:
        ValueError: In addition to the base class errors, if any of the options are not positive timedeltas.

    Examples:
        >>> rng = np.random.default_rng(1)
        >>> PositiveTimeDeltaGenerator([np.timedelta64(1, "s"), np.timedelta64(2, "s")]).rvs(10, rng)
        array([2, 2, 1, 2, 1, 1, 2, 1, 2, 1], dtype='timedelta64[s]')
        >>> PositiveTimeDeltaGenerator([1, 2])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be positive timedeltas. Got [1, 2].
        >>> PositiveTimeDeltaGenerator([np.timedelta64(1, "s"), np.timedelta64(-1, "s")])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be positive timedeltas. Got [np.timedelta64(-1,'s')].
    """

    X: list[POSITIVE_TIMEDELTA | str]

    def _validate(self, X: list[Any]):
        if not all(is_POSITIVE_TIMEDELTA(x) for x in X):
            fails = [x for x in X if not is_POSITIVE_TIMEDELTA(x)]
            raise ValueError(f"All elements should be positive timedeltas. Got {fails}.")

    @classmethod
    def _to_str(cls, x: np.timedelta64) -> str:
        as_sec = x.astype("timedelta64[s]") / np.timedelta64(1, "s")
        return f"{as_sec}s"

    @classmethod
    def _from_str(cls, x: str) -> np.timedelta64:
        return np.timedelta64(int(pytimeparse.parse(x) * 1e9), "ns").astype("timedelta64[s]")


class ProportionGenerator(DiscreteGenerator):
    """A class to generate random proportions.

    This merely applies type-checking to the DiscreteGenerator class.

    Attributes:
        X: The list of proportions to sample from.
        freq: The frequency of each option. If None, all options are equally weighted.

    Raises:
        ValueError: In addition to the base class errors, if any of the proportions are not numbers between 0
            and 1.

    Examples:
        >>> rng = np.random.default_rng(1)
        >>> ProportionGenerator([0, 1, 0.3]).rvs(10, rng)
        array([1. , 0.3, 0. , 0.3, 0. , 1. , 0.3, 1. , 1. , 0. ])
        >>> ProportionGenerator([1, 2])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be proportions (numbers between 0 and 1 inclusive).
        >>> ProportionGenerator(["a"])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be proportions (numbers between 0 and 1 inclusive).
    """

    X: list[PROPORTION]

    def __post_init__(self):
        if not all(is_PROPORTION(x) for x in self.X):
            raise ValueError("All elements should be proportions (numbers between 0 and 1 inclusive).")
        super().__post_init__()


class PositiveNumGenerator(DiscreteGenerator):
    """A class to generate random positive numbers.

    This merely applies type-checking to the DiscreteGenerator class.

    Attributes:
        X: The list of positive numbers to sample from.
        freq: The frequency of each option. If None, all options are equally weighted.

    Raises:
        ValueError: In addition to the base class errors, if any of the options are not positive numbers.

    Examples:
        >>> rng = np.random.default_rng(1)
        >>> PositiveNumGenerator([1, 2, 3.0]).rvs(10, rng)
        array([2., 3., 1., 3., 1., 2., 3., 2., 2., 1.])
        >>> PositiveNumGenerator([1, -1, 2])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be positive numbers.
        >>> PositiveNumGenerator([0])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be positive numbers.
        >>> PositiveNumGenerator(["a"])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be positive numbers.
    """

    X: list[POSITIVE_NUM]

    def __post_init__(self):
        if not all(is_POSITIVE_NUM(x) for x in self.X):
            raise ValueError("All elements should be positive numbers.")
        super().__post_init__()


class PositiveIntGenerator(PositiveNumGenerator):
    """A class to generate random positive integers.

    This merely applies type-checking to the DiscreteGenerator class.

    Attributes:
        X: The list of positive integers to sample from.
        freq: The frequency of each option. If None, all options are equally weighted.

    Raises:
        ValueError: In addition to the base class errors, if any of the options are not positive integers.

    Examples:
        >>> rng = np.random.default_rng(1)
        >>> PositiveIntGenerator([1, 2, 3]).rvs(10, rng)
        array([2, 3, 1, 3, 1, 2, 3, 2, 2, 1])
        >>> PositiveIntGenerator([0.1])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be integers.
    """

    X: list[POSITIVE_INT]

    def __post_init__(self):
        if not all(is_POSITIVE_INT(x) for x in self.X):
            raise ValueError("All elements should be integers.")
        super().__post_init__()
