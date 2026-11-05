						

from pathlib import Path
import xarray as xr

from wetb.dlb.report_generator.dlb_components import Component
from wetb.dlb.dlb_postprocs import (
    add_coords_from_filename,
)
from wetb.dlb.report_generator.dlb_latex import (
    make_component_table,
    make_psf_table,
    make_sensor_table,
    write_main_latex_report,
)
from wetb.dlb.report_generator.dlb_markdown import write_markdown_report
from wetb.dlb.report_generator.sections.directional import (
    DirectionalEquivalentSection,
    DirectionalExtremeSection,
)
from wetb.dlb.report_generator.sections.equivalent import EquivalentLoadsSection
from wetb.dlb.report_generator.sections.extreme import ExtremeLoadsSection
from wetb.dlb.report_generator.sections.statistics import StatisticsSection
from wetb.dlb.report_generator.dlb_utils import normalize_dlc_filter
from wetb.dlb.report_generator.dlb_utils import get_component_ids

DEFAULT_DATASET_FILENAMES = {
    'statistics': 'statistics.nc', 'extreme': 'extreme_loads.nc',
    'equivalent': 'equivalent_loads.nc',
    'dlb_extreme': 'DLB_extreme_loads.nc',
    'dlb_equivalent': 'DLB_equivalent_loads.nc',
    'damage': 'DLB_damage_by_dlc_wsp.nc',
    'directional_extreme': 'directional_extreme_loads.nc',
    'directional_equivalent': 'directional_equivalent_loads.nc',
    'dlb_directional_extreme': 'DLB_directional_extreme_loads.nc',
    'dlb_directional_equivalent': 'DLB_directional_equivalent_loads.nc',
}

def get_safety_factors(data):
    """Return factors from the first stored ``psf(dlc)`` coordinate.

    Precomputed extreme results must store their partial safety factors as a
    separate coordinate. Missing coordinates return ``None``.
    """
    psf = data.get("dlb_extreme_psf")
    if psf is None:
        psf = data.get("dlb_directional_extreme_psf")
    if psf is None:
        return None

    safety_factors = {}
    for dlc, factor in zip(psf.dlc.values, psf.values):
        dlc_key = f"DLC{str(dlc).removeprefix('DLC')}"
        safety_factors[dlc_key] = float(factor)
    return safety_factors

def load_report_array(file_path, per_simulation):
    """Read the primary report array and an optional ``psf`` coordinate."""
    dataset = xr.open_dataset(file_path)
    variables = list(dataset.data_vars)
    # Report files are constrained to one payload variable;
    # coordinates carry all associated sensor and case metadata.
    if len(variables) != 1:
        raise ValueError(f'Expected one data variable in {file_path}, found {variables}')

    # get psf from extreme and remove from data
    array = dataset[variables[0]]
    psf = dataset.coords.get('psf')
    unrelated_coords = [
        name for name, coordinate in array.coords.items()
        if not set(coordinate.dims).issubset(array.dims)
    ]
    if unrelated_coords:
        array = array.drop_vars(unrelated_coords)

    # add dlc, wsp and wdir as coordinates to data if not done already
    if per_simulation and not all(key in array.coords for key in ('dlc', 'wsp', 'wdir')):
        array = add_coords_from_filename(
            array,
            ['dlc', 'wsp', 'wdir'],
            r'DLC(\w+?)_wsp(\d{2})_wdir(\d{3})',
            [str, float, float],
        )
    # ensure python format string
    for coordinate in ('sensor_unit', 'sensor_description'):
        if coordinate in array.coords:
            array = array.assign_coords({coordinate: array[coordinate].astype(str)})
    return array, psf


def load_data(files, dlcs=None):
    """Try to load required datasets and warn instead of failing on missing files."""
    datasets = {}
    selected_dlcs = normalize_dlc_filter(dlcs)
    for dataset_key, file_path in files.items():
        
        try:
            data_array, psf = load_report_array(file_path, dataset_key not in {
                'dlb_extreme', 'dlb_equivalent', 'damage',
                'dlb_directional_extreme', 'dlb_directional_equivalent'})
            if selected_dlcs is not None and dataset_key not in {
                'dlb_extreme', 'dlb_equivalent', 'dlb_directional_equivalent'} and "dlc" in data_array.coords:
                data_array = data_array.where(
                    data_array["dlc"].astype(str).isin(selected_dlcs),
                    drop=True,
                )
            datasets[dataset_key] = data_array
            # Keep the independent safety-factor coordinate beside its source
            # array so it is available for the report-wide factor table.
            if psf is not None:
                datasets[f'{dataset_key}_psf'] = psf
            if selected_dlcs is not None and dataset_key in {
                'dlb_extreme', 'dlb_equivalent', 'dlb_directional_equivalent'}:
                print(f"WARNING: DLC filter cannot change precomputed {dataset_key} results; verify their upstream selection.")
        except FileNotFoundError:
            print(f"WARNING: Missing input file: {file_path}")
        except OSError as exc:
            print(f"WARNING: Could not load {file_path}: {exc}")

    return datasets


def make_front_matter(components, data, output_dir):
    """Generate report-wide tables that are not tied to one specific section."""
    make_component_table(components, output_dir=output_dir, filename="component_table.tex")
    report_dlcs = []
    # Pick the first per-simulation dataset with DLC coordinates. Its sequence
    # determines the display order of the report-wide safety-factor table.
    for dataset_key in (
        "statistics",
        "extreme",
        "equivalent",
        "directional_extreme",
        "directional_equivalent",
        "dlb_directional_extreme",
    ):
        if dataset_key in data and "dlc" in data[dataset_key].coords:
            report_dlcs = list(
                dict.fromkeys(str(dlc) for dlc in data[dataset_key]["dlc"].values)
            )
            break

    # make psf table    
    safety_factors = get_safety_factors(data)
    if safety_factors is not None and not report_dlcs:
        report_dlcs = [
            dlc.removeprefix('DLC') if dlc.upper().startswith('DLC') else dlc
            for dlc in safety_factors
        ]
    if safety_factors is not None:
        make_psf_table(report_dlcs, safety_factors, output_dir=output_dir, filename="psf_table.tex")
    else:
        print(
            "WARNING: No stored partial safety factors were found; omitting "
            "the safety factor table."
        )

    # make sensor table
    sensor_table_data = None
    sensor_table_ids = []
    sensor_table_components = []

    for dataset_key in ["statistics", "extreme", "equivalent"]:
        if dataset_key not in data:
            continue # if data sets w sensors are not available, skip
        
        candidate_ids = []
        candidate_components = []
        for component_name, component in components.items():
            component_ids = component.sensor_ids.get(dataset_key, [])
            candidate_ids.extend(component_ids)
            candidate_components.extend([component_name] * len(component_ids))

        if candidate_ids:
            # One source dataset is sufficient because component sensor IDs are
            # resolved separately for each dataset type.
            sensor_table_data = data[dataset_key]
            sensor_table_ids = candidate_ids
            sensor_table_components = candidate_components
            break

    if sensor_table_data is None:
        print(
            "WARNING: Skipping sensor table because no suitable statistics, extreme, "
            "or equivalent dataset with sensor IDs could be loaded."
        )
        return

    make_sensor_table(
        sensor_table_data,
        sensor_table_ids,
        sensor_components=sensor_table_components,
        special_name=None,
        output_dir=output_dir,
        unit_coord="sensor_unit",
    )


class DLBReport:
    """Configure and generate a component-load report from DLB datasets.

    The report accepts one baseline case or a baseline/comparison pair. Calling
    :meth:`generate` loads the available datasets, resolves each component's
    sensors, generates requested section outputs, and writes LaTeX and/or
    Markdown report entry points.

    Parameters
    ----------
    components : iterable of Component
        Components to include. Component names must be unique.
    output_dir : str or pathlib.Path
        Destination directory for generated figures, tables, and report files.
    report_title : str
        Title written into the generated reports.
    cases : dict[str, str | pathlib.Path | dict[str, str | pathlib.Path]]
        One or two named cases. A case may be a directory containing files with
        :data:`DEFAULT_DATASET_FILENAMES`, or a mapping from report dataset keys
        (for example ``"statistics"`` or ``"dlb_extreme"``) to arbitrary file
        paths. Omit optional dataset keys when their output is not required.
    output_formats : str or iterable of {"latex", "markdown"}, optional
        Output entry points to write. Defaults to ``("latex",)``.
    main_tex_filename : str, optional
        Name of the generated LaTeX entry document. Defaults to ``"main.tex"``.
    markdown_filename : str, optional
        Name of the generated Markdown document. Defaults to ``"report.md"``.
    dlcs : None, "all", scalar, or iterable, optional
        DLC selection for datasets retaining a DLC coordinate. The filter does
        not alter precomputed aggregated DLB results.
    fatigue_dlc_list : iterable, optional
        DLC labels included in fatigue-contribution plots.

    Attributes
    ----------
    components : dict[str, Component]
        Components keyed by their names.
    sections : dict[str, ReportSection]
        Available report sections keyed by section name.
    case_files : dict[str, dict[str, pathlib.Path]]
        Normalized input file paths for each case.
    case_data : dict[str, dict[str, xarray.DataArray]]
        Loaded input arrays, populated by :meth:`load_cases` or
        :meth:`generate`.
    reference_case : str or None
        Name of the first configured case after the data has been loaded.

    Notes
    -----
    Two-case reports generate comparison outputs. Their datasets used by the
    requested sections must be present in both cases and have compatible sensor
    metadata and component selections.
    """

    section_types = (
        StatisticsSection,
        ExtremeLoadsSection,
        EquivalentLoadsSection,
        DirectionalExtremeSection,
        DirectionalEquivalentSection,
    )

    def __init__(
        self,
        *,
        components,
        output_dir,
        report_title,
        cases,
        output_formats=("latex",),
        main_tex_filename="main.tex",
        markdown_filename="report.md",
        dlcs="all",
        fatigue_dlc_list=("12", "24", "64"),
    ):
        self.output_dir = Path(output_dir)
        self.report_title = str(report_title)
        self.output_formats = self._validate_output_formats(output_formats)
        self.main_tex_filename = str(main_tex_filename)
        self.markdown_filename = str(markdown_filename)
        self.dlcs = dlcs
        self.fatigue_dlc_list = list(fatigue_dlc_list)
        self.sections = {}
        for section_type in self.section_types:
            section = section_type(self)
            self.sections[section.key] = section

        self.components = {}
        for component in components:
            if not isinstance(component, Component):
                raise TypeError("components must contain Component objects.")
            if component.name in self.components:
                raise ValueError(f"Duplicate component name: {component.name}")

            if component.enabled_sections == ["all"]:
                component.enabled_sections = list(self.sections)

            unknown_sections = set(component.enabled_sections) - set(self.sections)
            if unknown_sections:
                raise ValueError(
                    f"Unknown section names: {sorted(unknown_sections)}. "
                    f"Valid options: {sorted(self.sections)}"
                )
            self.components[component.name] = component
        self.case_files = self._normalize_cases(cases)
        self.case_data = {}
        self.reference_case = None

    @staticmethod
    def _validate_output_formats(output_formats):
        """Validate output formats and remove duplicate selections.

        """
        if isinstance(output_formats, str):
            output_formats = [output_formats]

        formats = []
        for output_format in output_formats:
            normalized_format = str(output_format).lower()
            formats.append(normalized_format)

        supported_formats = {"latex", "markdown"}

        if any(fmt not in supported_formats for fmt in formats):
            raise ValueError(f"output_formats not supported.")
        if not formats:
            raise ValueError("output_formats must not be empty.")
        return list(dict.fromkeys(formats))


    @staticmethod
    def _normalize_cases(cases):
        """Normalize report case files depending on wether a folder or specific file paths have been given.

        Parameters
        ----------
        cases : dict
            One or two named case directories or dataset mappings, ie folder or full paths.

        Returns
        -------
        dict
            Case names mapped to known dataset keys and ``Path`` objects.

        Example
        -------
        cases = {
            "Baseline": "path/to/baseline/postproc",
            "Updated": "path/to/updated/postproc",
        }
        or
        cases = {
            "Baseline": {
                "statistics": Path(".../statistics.nc"),
                "extreme": Path(".../extreme_loads.nc"),
                ...
            },}
        """
        
        if not isinstance(cases, dict):
            raise ValueError(
                "cases must be a dictionary such as "
                "{'Baseline': Path('postproc')}, or "
                "{'Baseline': {'statistics': Path('statistics.nc'), ...}}. ")
        normalized_cases = {}
        # Convert a folder path into the standard paths expected for one case.
        for name, paths in cases.items():
            if isinstance(paths, (str, Path)):
                case_folder = Path(paths)
                paths = {}
                for dataset_key, filename in DEFAULT_DATASET_FILENAMES.items():
                    paths[dataset_key] = case_folder / filename

            if not isinstance(paths, dict):
                raise ValueError(
                    f"Case {name!r} must be either a directory path or a "
                    "dictionary of dataset paths."
                )
            unknown = set(paths) - set(DEFAULT_DATASET_FILENAMES)
            if unknown:
                raise ValueError(f"Unknown dataset names: {sorted(unknown)}")
            normalized_paths = {}
            for dataset_key, path in paths.items():
                if path is not None:
                    normalized_paths[dataset_key] = Path(path)
            normalized_cases[str(name)] = normalized_paths
        if len(normalized_cases) not in (1, 2):
            raise ValueError("cases must contain either 1 or 2 cases.")
        return normalized_cases

    def generate(self):
        """Generate all selected outputs and return this report instance."""
        self.load_cases()
        # Sensor IDs depend on the loaded reference coordinates and must be
        # resolved before front matter or individual sections use them.
        self.resolve_component_ids()
        self.generate_front_matter()
        self.generate_sections()
        self.write_reports()
        self.print_summary()
        return self

    def load_cases(self):
        """Load each configured case and choose the first as the reference case."""
        self.case_data = {}
        # Load each case separately so its name stays attached to its datasets
        # for later comparison and report generation.
        for case_name, files in self.case_files.items():
            loaded_data = load_data(files, dlcs=self.dlcs)
            self.case_data[case_name] = loaded_data

        if len(self.case_data) == 2:
            # Comparison figures require matching data and sensor ordering.
            self.validate_comparison_cases()

        # Set first case as reference
        self.reference_case = next(iter(self.case_files))

    def validate_comparison_cases(self):
        """Validate inputs needed by the configured comparison sections.
        """

        reference_name, reference_data = next(iter(self.case_data.items()))
        comparison_name, comparison_data = list(self.case_data.items())[1]
        
        requested_sections = set()
        for component in self.components.values():
            for section_key in component.enabled_sections:
                requested_sections.add(section_key)

        required_dataset_keys = set()

        for section_key in requested_sections:
            section = self.sections[section_key]
            required_dataset_keys.update(section.dataset_keys)

        reference_keys = set()
        for dataset_key in reference_data:
            if dataset_key in required_dataset_keys:
                reference_keys.add(dataset_key)

        comparison_keys = set()
        for dataset_key in comparison_data:
            if dataset_key in required_dataset_keys:
                comparison_keys.add(dataset_key)
        # check datasets are present for both cases before comparison.
        if reference_keys != comparison_keys:
            missing = sorted(reference_keys - comparison_keys)
            unexpected = sorted(comparison_keys - reference_keys)
            details = []
            if missing:
                details.append(f"missing from {comparison_name}: {missing}")
            if unexpected:
                details.append(f"missing from {reference_name}: {unexpected}")
            raise ValueError(
                "Comparison cases must provide the same datasets required by "
                f"the requested sections {sorted(requested_sections)}; "
                + "; ".join(details)
            )
        # Compare metadata only for datasets that participate in output.
        for dataset_key in sorted(reference_keys):
            reference = reference_data[dataset_key]
            comparison = comparison_data[dataset_key]
            has_sensors = "sensor_name" in reference.coords
            if has_sensors != ("sensor_name" in comparison.coords):
                raise ValueError(
                    f"Comparison dataset '{dataset_key}' has inconsistent sensor metadata."
                )
            if not has_sensors:
                continue

            for coordinate in ("sensor_name", "sensor_description", "sensor_unit"):
                reference_coordinate = reference.coords.get(coordinate)
                comparison_coordinate = comparison.coords.get(coordinate)
                if (reference_coordinate is None) != (comparison_coordinate is None):
                    raise ValueError(
                        f"Comparison dataset '{dataset_key}' has inconsistent "
                        f"{coordinate} metadata."
                    )
                if (reference_coordinate is not None
                        and not reference_coordinate.identical(comparison_coordinate)):
                    raise ValueError(
                        f"Comparison cases have different {coordinate} ordering "
                        f"for dataset '{dataset_key}'."
                    )

            for component_name, component in self.components.items():
                reference_ids = get_component_ids(component.match, reference)
                comparison_ids = get_component_ids(component.match, comparison)
                if reference_ids != comparison_ids:
                    raise ValueError(
                        f"Component '{component_name}' selects different sensors "
                        f"in comparison dataset '{dataset_key}'."
                    )

    def resolve_component_ids(self):
        """ Find and store every component's sensor indexes in the reference case for the different data sets."""
        for component in self.components.values():
            component.resolve_sensor_ids(
                self.case_data[self.reference_case],
                tuple(DEFAULT_DATASET_FILENAMES),
            )

    def generate_front_matter(self):
        """Generate report-wide component, safety-factor, and sensor tables."""
        make_front_matter(
            self.components,
            self.case_data[self.reference_case],
            self.output_dir,
        )

    def generate_sections(self):
        """Generate available requested sections for each configured component."""
        for component_name, component in self.components.items():
            for section_key in component.enabled_sections:
                section = self.sections[section_key]
                if section.is_available():
                    section.generate(component_name, component)

    def write_reports(self):
        """Write the configured LaTeX and Markdown report entry points."""
        if "latex" in self.output_formats:
            write_main_latex_report(
                output_dir=self.output_dir,
                components=self.components,
                sections=self.sections,
                filename=self.main_tex_filename,
                title=self.report_title,
            )
        if "markdown" in self.output_formats:
            write_markdown_report(
                output_dir=self.output_dir,
                components=self.components,
                filename=self.markdown_filename,
                title=self.report_title,
            )

    def print_summary(self):
        """Print the generated section names for every report component."""
        for component_name, component in self.components.items():
            print(
                f"Generated {component_name} sections: "
                f"{', '.join(component.enabled_sections)}"
            )


