"""Search and filter HAWC2 sensor metadata."""

import numpy as np
import pandas as pd


class SensorSearch(object):
    """Search sensor IDs and their available metadata.

    Sensors can be filtered by name, unit, description, ID, HTC input, and
    label. Name, unit, and description are searchable when supplied during
    initialization. HTC input and label are only searchable when HTC metadata
    is available. Labels are extracted from the text following ``#`` in each
    HTC input.

    Filters on different metadata fields use AND logic. A sensor must match
    every supplied field. Multiple search terms within one field use OR logic;
    a sensor only needs to match one of those terms for that field.

    Text matching is case-insensitive, ignores spaces, and accepts partial
    matches.

    Examples
    --------
    Match a name containing "tower" AND a unit containing "kNm":

    >>> sensors(name="tower", unit="kNm")

    Match a name containing either "tower" OR "blade":

    >>> sensors(name=["tower", "blade"])

    Combine OR terms in one field with an AND condition in another:

    >>> sensors(name=["tower", "blade"], desc="load")

    This means::

        (name contains "tower" OR name contains "blade")
        AND description contains "load"

    Search an extracted HTC label, when HTC input was supplied:

    >>> sensors(label="tower_base")

    The result is a pandas DataFrame containing the matching IDs and all
    available sensor metadata.

    Raises
    ------
    ValueError
        If a requested metadata field was not supplied during initialization.
    """

    def __init__(self, names=None, units=None, desc=None, htc=None):
        metadata = {
            column: np.atleast_1d(values).tolist()
            for column, values in [
                ("name", names),
                ("unit", units),
                ("desc", desc),
                ("htc", htc),
            ]
            if values is not None
        }

        if not metadata:
            raise ValueError("No sensor metadata was provided")

        n_lst = [len(v) for v in metadata.values()]

        if not np.all(n_lst[0] == np.array(n_lst)):
            raise ValueError("Inputs must have same length or be None")

        if "htc" in metadata:
            metadata["label"] = [
                str(value).split("#", 1)[1].strip()
                if "#" in str(value)
                else ""
                for value in metadata["htc"]
            ]

        n_sensor = n_lst[0]

        self.df = pd.DataFrame({
            "id": np.arange(n_sensor),
            **metadata,
        })

    def __call__(
        self,
        name=None,
        unit=None,
        desc=None,
        htc=None,
        label=None,
        id=None,
    ):
        """Return matching sensors as a metadata DataFrame.

        Arguments correspond to the ``Name``, ``Unit``, ``Description``,
        ``htc``, ``Label``, and ``id`` columns. Omitted arguments are ignored.
        An empty DataFrame is returned when no sensors match.
        """
        if id is None:
            sensor_df = self.df
        else:
            sensor_df = self.df.set_index("id", drop=False).loc[np.atleast_1d(id)]

        matches = pd.Series(True, index=sensor_df.index)

        for column, value in [
            ("name", name),
            ("unit", unit),
            ("desc", desc),
            ("htc", htc),
            ("label", label),
        ]:
            if value is None:
                continue
            if column not in sensor_df:
                raise ValueError(f"{column} metadata is not available")
            matches &= self._contains_any(sensor_df[column], value)

        return sensor_df.loc[matches].reset_index(drop=True)


    def get_sensor_id(self, **kwargs):
        """Return matching channel IDs as a one-dimensional integer array."""
        return self(**kwargs)["id"].to_numpy(dtype=int)

    @staticmethod
    def _normalize(value):
        return str(value).strip().lower().replace(" ", "").lstrip("#")

    @classmethod
    def _contains_any(cls, column, values):
        # normalize searching for one or multiple values
        search_values = np.atleast_1d(values)
        search_texts = [cls._normalize(value) for value in search_values]

        # for each value / search tag see if there's a match
        # ie tags: ['tower base', 'pitch1 angle'] becomes ['towerbase'] ['pitch1angle']
        def value_contains_search_text(value):
            value = cls._normalize(value)
            for search_text in search_texts:
                if search_text in value:
                    return True
            return False

        return column.map(value_contains_search_text)
