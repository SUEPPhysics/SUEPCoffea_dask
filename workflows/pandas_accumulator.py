import pandas as pd
from coffea.processor.accumulator import AccumulatorABC


class pandas_accumulator(AccumulatorABC):
    """An appendable pandas table
    Parameters
    ----------
        value : pandas.DataFrame
            The identity value array, which should be an empty ndarray
            with the desired row shape. The column dimension will correspond to
            the first index of `value` shape.
    Examples
    --------
    If a set of accumulators is defined as::
        a = pandas_accumulator(pd.DataFrame())
        b = pandas_accumulator(pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}))
        b = pandas_accumulator(pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]}))
    then:
    >>> a + b
    column_accumulator(array([1., 2., 3.]))
    >>> c + b + a
    column_accumulator(array([4., 5., 6., 1., 2., 3.]))
    """

    def __init__(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("pandas_accumulator only works with pandas DataFrames")
        self._empty = pd.DataFrame()
        self._value = value

    def __repr__(self):
        return "pandas_accumulator(\n%r\n)" % self.value

    def identity(self):
        return pandas_accumulator(self._empty)

    def add(self, other):
        if not isinstance(other, pandas_accumulator):
            raise ValueError("pandas_accumulator cannot be added to %r" % type(other))
        if other._empty.shape != self._empty.shape:
            raise ValueError(
                "Cannot add two column_accumulator objects of dissimilar shape (%r vs %r)"
                % (self._empty.shape, other._empty.shape)
            )
        self._value = pd.concat((self._value, other._value))

    def loc(self, indices, key, value):
        self._value.loc[indices, key] = value

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError("Column name must be a string not %r." % type(key))
        self._value[key] = value

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise ValueError("Column name must be a string not %r." % type(key))
        if key not in self._value.keys():
            raise KeyError(f"Key {key} does not exist in accumulator")
        return self._value[key]

    @property
    def value(self):
        """The current value of the column
        Returns a numpy array where the first dimension is the column dimension
        """
        return self._value
