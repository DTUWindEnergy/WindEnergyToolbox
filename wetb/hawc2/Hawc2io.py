"""Deprecated NumPy-compatible interface for reading HAWC2 output."""

import warnings

from wetb.hawc2.Hawc2output import Hawc2Output


class ReadHawc2(Hawc2Output):
    """Deprecated adapter preserving the original NumPy return format."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ReadHawc2 is deprecated; use Hawc2Output instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        if result is None:
            return None
        return result.to_numpy()
