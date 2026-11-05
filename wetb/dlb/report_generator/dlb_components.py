from wetb.dlb.report_generator.dlb_utils import get_component_ids

class Component:
    """Configure one named group of sensors in a generated report.

    A component defines which report sections to create, how to locate its
    sensors in every input dataset, and the Wöhlner exponent to use for its
    fatigue outputs.  The report resolves the configured ``match`` rule for
    each loaded dataset before it generates tables and figures.

    Parameters
    ----------
    name : str
        Unique component identifier. It is used in output paths and, unless
        ``label`` is supplied, as the display name.
    sections : {"all", str, iterable of str}
        Sections requested for the component. ``"all"`` expands to every
        section supported by :class:`DLBReport`; duplicate explicit names are
        removed while preserving their supplied order.
    fatigue_m : number or iterable of numbers
        Wohler exponent used by equivalent-load and fatigue outputs.
    match : str or dict
        Sensor-seach tag. A string is matched against both sensor names
        and descriptions. A dictionary may provide ``"name"`` and/or
        ``"description"`` independently; each value may be a string or an
        iterable of strings.
    label : str, optional
        Display name for tables and report headings. Defaults to ``name``.

    Attributes
    ----------
    name : str
        Component identifier.
    label : str
        Display name.
    enabled_sections : list of str
        Normalized requested section names.
    fatigue_m : number or iterable of numbers
        Configured Wohler exponent or exponents.
    match : dict
        Canonical name/description matcher.
    sensor_ids : dict[str, list[int]]
        Sensor indexes resolved per dataset. This is populated by
        :meth:`resolve_sensor_ids` when the report is generated.

    Notes
    -----
    Matching is case-insensitive, ignores spaces, and uses substring matching.
    A sensor selected by either a supplied name or description criterion is
    included.
    """

    def __init__(self, name, sections, fatigue_m, match, label=None):
        if fatigue_m is None:
            raise ValueError("Component fatigue_m must be defined.")
        if match is None:
            raise ValueError("Component match must be defined.")
        self.name = str(name)
        self.label = str(label or name)
        self.enabled_sections = self._normalize_sections(sections)
        self.fatigue_m = fatigue_m
        self.match = self._normalize_match(match)
        # Dataset keys map to sensor indexes because datasets can have different
        # sensor orderings or omit sensors entirely.
        self.sensor_ids = {}

    @staticmethod
    def _normalize_sections(section_names):
        """Return unique requested section names.

        Parameters
        ----------
        section_names : str or iterable of str
            Requested report sections, or ``"all"``.

        Returns
        -------
        list of str
            Requested section names in their supplied order.
        """
        if section_names is None:
            raise ValueError("Component sections must be defined.")
        if section_names == "all":
            return ["all"]
        if isinstance(section_names, str):
            section_names = [section_names]
        sections = []
        for section_name in section_names:
            if section_name not in sections:
                sections.append(section_name)
        return sections

    @staticmethod
    def _normalize_match(match):
        """Convert a component sensor matcher to its canonical mapping.

        Parameters
        ----------
        match : str or dict
            Sensor-name and/or description matching criteria.

        Returns
        -------
        dict
            Mapping containing supported match criteria.
        """
        if isinstance(match, str):
            return {"name": match, "description": match}
        elif isinstance(match, dict):
            return {key: match[key] for key in ("name", "description") if key in match}
        raise ValueError("Component match must be either a string or a dict.")

    def resolve_sensor_ids(self, datasets, dataset_keys):
        """Resolve and store component sensor indexes for each available dataset.

        Parameters
        ----------
        datasets : dict
            Loaded datasets keyed by report dataset name.
        dataset_keys : iterable of str
            Dataset names for which indexes should be stored.
        """
        self.sensor_ids = {}
        for dataset_key in dataset_keys:
            if dataset_key not in datasets:
                self.sensor_ids[dataset_key] = []
                continue

            dataset = datasets[dataset_key]
            self.sensor_ids[dataset_key] = get_component_ids(self.match, dataset)


