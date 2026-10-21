from pathlib import Path

import numpy as np
from dlb_models import AnalysisConfig, ComponentConfig, DLBCase, ReportConfig
from dlb_postprocs import *
from dlb_plots import *
from dlb_latex import *
from dlb_reporting import *
from dlb_utils import component_path_name, get_component_ids, safety_factor_list


# Section setup. Keys are the only supported user-facing section names.
SECTION_CONFIG = {
    "statistics": {"requires": ["statistics"], "file": "statistics.nc"},
    "extreme": {"requires": ["extreme"], "file": "extreme_loads.nc"},
    "equivalent": {"requires": ["equivalent"], "file": "equivalent_loads.nc"},
    "directional_extreme": {"requires": ["directional_extreme"], "file": "directional_extreme_loads.nc"},
    "directional_equivalent": {"requires": ["directional_equivalent", "equivalent"], "file": "directional_equivalent_loads.nc"},
}

ID_DATASET_KEYS = ["statistics", "equivalent", "directional_equivalent", "directional_extreme", "extreme"]

def section_lookup():
    """Return lookup for supported section names."""
    return {section_name: section_name for section_name in SECTION_CONFIG}


def section_requirements(section_name):
    """Return the dataset keys required by one section."""
    return SECTION_CONFIG[section_name]["requires"]


def dataset_file(dataset_key):
    """Return the postproc filename for a dataset key."""
    try:
        return SECTION_CONFIG[dataset_key]["file"]
    except KeyError as exc:
        raise KeyError(f"No dataset file configured for '{dataset_key}'.") from exc


def normalize_sections(section_names):
    """Convert user-provided section names into the internal canonical names."""
    if section_names == "all":
        return list(SECTION_CONFIG)
    if isinstance(section_names, str):
        section_names = [section_names]

    lookup = section_lookup()
    normalized = []
    for name in section_names:
        key = lookup.get(name)
        if key is None:
            raise ValueError(
                f"Unknown section '{name}'. Valid options: {sorted(lookup)}"
            )
        if key not in normalized:
            normalized.append(key)
    return normalized


def normalize_component_config(component_name, config):
    """Normalize compact and legacy component config into internal keys."""
    return ComponentConfig.from_mapping(component_name, config, normalize_sections)


def normalize_components(components):
    """Normalize all user component inputs once at startup."""
    return {
        component_name: normalize_component_config(component_name, config)
        for component_name, config in components.items()
    }


def required_datasets(enabled_sections):
    """Return the dataset keys needed for the selected report sections."""
    required = list(dict.fromkeys(
        dataset_key
        for section in enabled_sections
        for dataset_key in section_requirements(section)
    ))

    if "statistics" not in required:
        required.append("statistics")

    return required


def resolve_component_sections(components):
    """Return normalized enabled sections for each component."""
    return {
        component_name: component.enabled_sections
        for component_name, component in components.items()
    }


def union_enabled_sections(component_sections):
    """Return the ordered union of all sections enabled by any component."""
    return list(dict.fromkeys(
        section
        for sections in component_sections.values()
        for section in sections
    ))


def normalize_cases(report_config):
    """Return configured postproc cases and enforce one- or two-case reports."""
    if isinstance(report_config, ReportConfig):
        return {
            name: DLBCase(name=name, postproc_dir=postproc_dir)
            for name, postproc_dir in report_config.cases.items()
        }
    report = ReportConfig.from_mapping(report_config)
    return normalize_cases(report)


def normalize_dlc_filter(dlcs):
    """Return a string list of selected DLCs, or None for all DLCs."""
    if dlcs in (None, "all"):
        return None
    if isinstance(dlcs, (str, int, float)):
        return [str(dlcs)]
    return [str(dlc) for dlc in dlcs]


def filter_data_by_dlc(data_array, dlcs):
    """Filter an xarray object by DLC when a DLC coordinate is available."""
    dlcs = normalize_dlc_filter(dlcs)
    if dlcs is None or "dlc" not in data_array.coords:
        return data_array
    return data_array.where(data_array["dlc"].astype(str).isin(dlcs), drop=True)


def selected_fatigue_dlcs(analysis_config):
    """Return fatigue DLCs that remain after the optional global DLC filter."""
    fatigue_dlcs = normalize_dlc_filter(analysis_config.fatigue_dlc_list) or []
    selected_dlcs = normalize_dlc_filter(analysis_config.dlcs)

    if selected_dlcs is None:
        return fatigue_dlcs

    selected = set(selected_dlcs)
    return [dlc for dlc in fatigue_dlcs if dlc in selected]


def get_report_dlcs(data):
    """Return the unique filtered DLCs present in the loaded report data."""
    for dataset_key in ["statistics", "extreme", "equivalent", "directional_extreme", "directional_equivalent"]:
        if dataset_key in data and "dlc" in data[dataset_key].coords:
            dlcs = [str(dlc) for dlc in data[dataset_key]["dlc"].values]
            return list(dict.fromkeys(dlcs))
    return []


def resolve_safety_factor_list(analysis_config):
    """Return configured partial safety factors, or the default list."""
    configured = analysis_config.safety_factor_list
    if configured is None:
        return safety_factor_list
    return configured


def load_data(postproc_dir, enabled_sections, dlcs=None):
    """Try to load required datasets and warn instead of failing on missing files."""
    datasets = {}

    for dataset_key in required_datasets(enabled_sections):
        file_path = postproc_dir / dataset_file(dataset_key)
        try:
            datasets[dataset_key] = filter_data_by_dlc(
                nc_to_dataarray(file_path),
                dlcs,
            )
        except FileNotFoundError:
            print(f"WARNING: Missing input file: {file_path}")
        except OSError as exc:
            print(f"WARNING: Could not load {file_path}: {exc}")

    return datasets


def load_case_data(cases, enabled_sections, dlcs=None):
    """Load required datasets for every configured postproc case."""
    case_data = {}
    for case_name, case in cases.items():
        case.datasets = load_data(case.postproc_dir, enabled_sections, dlcs=dlcs)
        case_data[case_name] = case.datasets
    return case_data


def reference_case_name(cases):
    """Return the first configured case name."""
    return next(iter(cases))

def resolve_component_ids(components, data):
    """
    Resolve sensor IDs for each component and each report section.

    The script keeps these IDs under component["ids"] so the later report
    generators can stay simple and only focus on writing tables/plots.
    """
    resolved = {}

    for component_name, config in components.items():
        component = config
        component.ids = {
            dataset_key: get_component_ids(component.match.as_dict(), data[dataset_key])
            if dataset_key in data else []
            for dataset_key in ID_DATASET_KEYS
        }

        resolved[component_name] = component

    return resolved


def build_weight_list(da_equivalent_loads, weight_params):
    """Build the shared weight list once for equivalent-load calculations."""
    return get_weight_list(
        da_equivalent_loads,
        weight_params.n_years,
        weight_params.Vin,
        weight_params.Vout,
        weight_params.Vr,
        weight_params.Vref,
        weight_params.Vstep,
    )


def build_equivalent_load_cases(case_data, weight_params):
    """Return weighted equivalent loads for every case with equivalent data."""
    return {
        case_name: get_DLB_eq_loads(
            case_datasets["equivalent"],
            build_weight_list(case_datasets["equivalent"], weight_params),
        )
        for case_name, case_datasets in case_data.items()
        if "equivalent" in case_datasets
    }


def filter_fatigue_dlcs(data_array, fatigue_dlcs):
    """Return data filtered to selected fatigue DLCs."""
    return data_array.where(data_array["dlc"].isin(fatigue_dlcs), drop=True)


def build_directional_equivalent_cases(case_data, weight_params, fatigue_dlcs):
    """Return weighted directional equivalent loads for each complete case."""
    return {
        case_name: get_DLB_eq_loads(
            eq_loads=filter_fatigue_dlcs(case_datasets["directional_equivalent"], fatigue_dlcs),
            weight_list=build_weight_list(case_datasets["equivalent"], weight_params),
        )
        for case_name, case_datasets in case_data.items()
        if "directional_equivalent" in case_datasets and "equivalent" in case_datasets
    }


def expand_m_values(m_value, sensor_ids):
    """Return one fatigue exponent per sensor ID."""
    if isinstance(m_value, int):
        return np.full(len(sensor_ids), m_value)
    return np.asarray(m_value)


def sensor_name_and_slug(data_array, sensor_id):
    """Return the original sensor name together with a filesystem-safe slug."""
    name = str(data_array["sensor_name"][sensor_id].values)
    slug = name.strip().lower().replace(" ", "_").replace(":", "")
    return name, slug


def component_paths(output_dir, component_name):
    """Return filesystem-safe component name and base output path."""
    component_path = component_path_name(component_name)
    return component_path, output_dir / component_path


def clear_matching_files(folder, pattern):
    """Delete stale generated files matching one pattern."""
    for stale_file in folder.glob(pattern):
        stale_file.unlink()


def section_table_folder(base_path, section_name, cases, case_name):
    """Return the output table folder for a section and case."""
    folder = base_path / "latex_tables" / section_name
    if len(cases) == 2:
        folder = folder / component_path_name(case_name)
    return folder


def case_dataset_map(cases, dataset_key):
    """Return case_name -> dataset_key data for all cases containing the dataset."""
    return {
        case_name: case_data[dataset_key]
        for case_name, case_data in cases.items()
        if dataset_key in case_data
    }


def make_front_matter(components, data, output_dir, analysis_config):
    """Generate report-wide tables that are not tied to one specific section."""
    make_component_table(components, output_dir=output_dir, filename="component_table.tex")
    report_dlcs = get_report_dlcs(data)
    dlc_table_path = output_dir / "dlc_table.tex"
    if dlc_table_path.exists():
        dlc_table_path.unlink()
    make_psf_table(
        report_dlcs,
        resolve_safety_factor_list(analysis_config),
        output_dir=output_dir,
        filename="psf_table.tex",
    )

    sensor_table_data = None
    sensor_table_ids = []
    sensor_table_components = []

    for dataset_key, id_key in [
        ("statistics", "statistics"),
        ("extreme", "extreme"),
        ("equivalent", "equivalent"),
    ]:
        if dataset_key not in data:
            continue

        candidate_ids = []
        candidate_components = []
        for component_name, component in components.items():
            candidate_ids.extend(component.ids[id_key])
            candidate_components.extend([component_name] * len(component.ids[id_key]))

        if candidate_ids:
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


def generate_statistics_section(component_name, component, data, output_dir, cases=None):
    """Generate statistics tables and plots for one component."""
    if cases is None:
        cases = {"v1": data}

    component_path, base_path = component_paths(output_dir, component_name)
    reference_case = reference_case_name(cases)
    reference_data = cases[reference_case]
    sensor_ids = component.ids["statistics"]
    if not sensor_ids:
        return

    for case_name, case_data in cases.items():
        case_path = component_path_name(case_name)
        table_folder = section_table_folder(base_path, "stats", cases, case_name)
        clear_matching_files(table_folder, "*.tex")

        dlc_statistics_tables_to_latex_file(
            case_data["statistics"],
            sensor_input=sensor_ids,
            caption_prefix=f"Statistics loads per DLC for {component_name} ({case_name})",
            label_prefix=f"tab:statistics_loads_{component_path}_{case_path}",
            output_path=str(table_folder / f"{component_path}.tex"),
        )

    for sensor_id in sensor_ids:
        name, slug = sensor_name_and_slug(reference_data["statistics"], sensor_id)

        if len(cases) == 1:
            plot_dlc_statistics(
                reference_data["statistics"],
                sensor_input=[sensor_id],
                sensor_title=[component_path],
                save_dir=str(base_path / "figures" / "stats" / f"{component_path}_{slug}"),
                show=False,
            )
        else:
            plot_dlc_statistics_cases(
                case_dataset_map(cases, "statistics"),
                sensor_input=[sensor_id],
                sensor_title=[component_path],
                save_dir=str(base_path / "figures" / "stats" / "comparison" / f"{component_path}_{slug}"),
                show=False,
            )


def generate_extreme_section(component_name, component, data, output_dir, cases=None):
    """Generate extreme-load tables and plots for one component."""
    if cases is None:
        cases = {"v1": data}

    sensor_ids = component.ids["extreme"]
    if not sensor_ids:
        return

    reference_case = reference_case_name(cases)
    reference_data = cases[reference_case]
    selected = reference_data["extreme"].isel(sensor_name=sensor_ids)
    component_path, base_path = component_paths(output_dir, component_name)

    for local_idx, sensor_id in enumerate(sensor_ids):
        name, slug = sensor_name_and_slug(selected, local_idx)

        for case_name, case_data in cases.items():
            case_selected = case_data["extreme"].isel(sensor_name=sensor_ids)
            extreme_matrix = get_DLB_extreme_loads(case_selected)
            case_path = component_path_name(case_name)
            table_folder = section_table_folder(base_path, "extreme", cases, case_name)

            dlc_extreme_loads_to_latex(
                extreme_matrix,
                sensor_input=[local_idx],
                caption=f"Extreme loads per DLC for {component_name} {name} ({case_name})",
                label=f"tab:extreme_loads_{component_path}_{slug}_{case_path}",
                output_path=str(table_folder / f"{component_path}_{slug}.tex"),
            )
            diagonal_extreme_loads_to_latex(
                case_selected,
                sensor_input=local_idx,
                caption=f"Diagonal extreme loads and source files for {component_name} {name} ({case_name})",
                label=f"tab:diagonal_extreme_sources_{component_path}_{slug}_{case_path}",
                output_path=str(table_folder / f"{component_path}_{slug}_diagonal_sources.tex"),
            )

        if len(cases) == 1:
            plot_dlc_extreme(
                selected,
                sensor_input=[local_idx],
                sensor_title=[component_path],
                save_dir=str(base_path / "figures" / "extreme"),
                show=False,
            )
        else:
            plot_dlc_extreme(
                case_dataset_map(cases, "extreme"),
                sensor_input=[sensor_id],
                sensor_title=[component_path],
                save_dir=str(base_path / "figures" / "extreme"),
                show=False,
            )


def generate_equivalent_section(
    component_name,
    component,
    data,
    output_dir,
    dlb_eq_loads,
    dlb_eq_load_cases=None,
    analysis_config=None,
):
    """Generate equivalent-load tables and plots for one component."""
    sensor_ids = component.ids["equivalent"]
    if not sensor_ids:
        return

    m_values = expand_m_values(component.fatigue_m, sensor_ids)
    component_path, base_path = component_paths(output_dir, component_name)

    DLB_equivalent_loads_to_latex(
        dlb_eq_load_cases if dlb_eq_load_cases and len(dlb_eq_load_cases) > 1 else dlb_eq_loads,
        sensor_input=sensor_ids,
        m_values=m_values,
        component=f"{component_name} ",
        caption=f"Equivalent loads for {component_name}",
        label=f"tab:Equivalent_loads_{component_path}",
        output_path=str(base_path / "latex_tables" / "eq_load" / f"{component_path}.tex"),
    )

    plot_dlb_equivalent_bar(
        dlb_eq_load_cases or dlb_eq_loads,
        sensor_input=sensor_ids,
        m_value=m_values,
        component=component_name,
        save_dir=str(base_path / "figures" / "equivalent" / "summary"),
        show=False,
    )

    equivalent_weight_list = build_weight_list(
        data["equivalent"],
        analysis_config.weight_params,
    )

    summary_dir = base_path / "figures" / "equivalent" / "summary"
    for stale_pattern in [
        "*_damage_contribution_m*.png",
        "*_wind_speed_damage_*.png",
    ]:
        clear_matching_files(summary_dir, stale_pattern)

    plot_fatigue_summary_damage(
        data["equivalent"],
        equivalent_weight_list,
        sensor_input=sensor_ids,
        m_values=m_values,
        fatigue_dlcs=selected_fatigue_dlcs(analysis_config),
        wind_speed_dlc="12",
        component=component_name,
        save_dir=summary_dir,
        show=False,
    )

    for local_idx, sensor_id in enumerate(sensor_ids):
        _, slug = sensor_name_and_slug(data["equivalent"], sensor_id)
        short_term_dir = base_path / "figures" / "equivalent" / f"{component_path}_{slug}"
        short_term_dir.mkdir(parents=True, exist_ok=True)
        clear_matching_files(short_term_dir, "*.png")
        plot_dlb_equivalent(
            data["equivalent"],
            sensor_input=[sensor_id],
            m_value=[m_values[local_idx]],
            component=component_name,
            save_dir=str(short_term_dir),
            show=False,
        )


def generate_directional_extreme_section(
    component_name, component, output_dir, dlb_directional_extreme
):
    """Generate directional extreme envelopes for one component."""
    sensor_ids = component.ids["directional_extreme"]
    if not sensor_ids:
        return

    component_path, base_path = component_paths(output_dir, component_name)
    m_values = expand_m_values(component.fatigue_m, sensor_ids)

    for local_idx, sensor_id in enumerate(sensor_ids):
        plot_load_envl(
            dlb_directional_extreme,
            sensor_input=[sensor_id],
            m_value=m_values[local_idx],
            component=component_name,
            save_dir=str(base_path / "figures" / "directional_extreme"),
            show=False,
        )


def generate_directional_equivalent_section(
    component_name,
    component,
    output_dir,
    dlb_directional_equivalent,
    dlb_directional_equivalent_cases=None,
):
    """Generate directional equivalent envelopes for one component."""
    sensor_ids = component.ids["directional_equivalent"]
    if not sensor_ids:
        return

    component_path, base_path = component_paths(output_dir, component_name)
    m_values = expand_m_values(component.fatigue_m, sensor_ids)
    figure_dir = base_path / "figures" / "directional_equivalent"
    figure_dir.mkdir(parents=True, exist_ok=True)

    if dlb_directional_equivalent_cases and len(dlb_directional_equivalent_cases) == 2:
        comparison_dir = figure_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        clear_matching_files(comparison_dir, "*.png")
        for local_idx, sensor_id in enumerate(sensor_ids):
            plot_dir_load_envl_cases(
                dlb_directional_equivalent_cases,
                sensor_input=[sensor_id],
                m_value=[m_values[local_idx]],
                component=component_name,
                save_dir=str(comparison_dir),
                show=False,
            )
        return

    clear_matching_files(figure_dir, "*.png")

    plot_dir_load_envl(
        dlb_directional_equivalent,
        sensor_input=sensor_ids,
        m_value=m_values,
        component=component_name,
        save_dir=str(figure_dir),
        show=False,
    )


def missing_section_data(section_name, data):
    """Return required dataset keys missing from one loaded data mapping."""
    return [key for key in section_requirements(section_name) if key not in data]


def section_is_available(section_name, data):
    """Check whether all datasets needed by a section are available."""
    missing = missing_section_data(section_name, data)
    if missing:
        print(
            f"WARNING: Skipping section '{section_name}' because required data is missing: "
            + ", ".join(missing)
        )
        return False
    return True


def section_is_available_for_cases(section_name, case_data):
    """Check whether all datasets needed by a section are available for every case."""
    available = True
    for case_name, data in case_data.items():
        missing = missing_section_data(section_name, data)
        if missing:
            print(
                f"WARNING: Skipping section '{section_name}' for case '{case_name}' "
                "because required data is missing: " + ", ".join(missing)
            )
            available = False

    return available

class DLBReport:
    """Coordinate loading data and generating a DLB LaTeX report."""

    def __init__(self, report_config, analysis_config, components):
        self.report_config = (
            report_config
            if isinstance(report_config, ReportConfig)
            else ReportConfig.from_mapping(report_config)
        )
        self.analysis_config = (
            analysis_config
            if isinstance(analysis_config, AnalysisConfig)
            else AnalysisConfig.from_mapping(analysis_config)
        )
        self.components = normalize_components(components)

        self.output_dir = self.report_config.output_dir
        self.cases = normalize_cases(self.report_config)
        self.component_sections = {}
        self.enabled_sections = []
        self.case_data = {}
        self.reference_case = None
        self.data = None

        self.weight_list = None
        self.dlb_eq_loads = None
        self.dlb_eq_load_cases = None
        self.dlb_directional_extreme = None
        self.dlb_directional_equivalent = None
        self.dlb_directional_equivalent_cases = None

    def generate(self):
        self.resolve_sections()
        self.load_cases()
        self.resolve_component_ids()
        self.generate_front_matter()
        self.precompute()
        self.generate_sections()
        self.write_main_latex()
        self.print_summary()
        return self

    def resolve_sections(self):
        self.component_sections = resolve_component_sections(self.components)
        self.enabled_sections = union_enabled_sections(self.component_sections)

    def load_cases(self):
        self.case_data = load_case_data(
            self.cases,
            self.enabled_sections,
            dlcs=self.analysis_config.dlcs,
        )
        self.reference_case = reference_case_name(self.cases)
        self.data = self.case_data[self.reference_case]

    def resolve_component_ids(self):
        self.components = resolve_component_ids(self.components, self.data)

    def generate_front_matter(self):
        make_front_matter(
            self.components,
            self.data,
            self.output_dir,
            self.analysis_config,
        )

    def precompute(self):
        if "equivalent" in self.data:
            self.weight_list = build_weight_list(
                self.data["equivalent"],
                self.analysis_config.weight_params,
            )
            self.dlb_eq_loads = get_DLB_eq_loads(
                self.data["equivalent"],
                self.weight_list,
            )
            self.dlb_eq_load_cases = build_equivalent_load_cases(
                self.case_data,
                self.analysis_config.weight_params,
            )

        if "directional_extreme" in self.data:
            self.dlb_directional_extreme = get_DLB_directional_extreme_loads(
                self.data["directional_extreme"],
                all_values=True,
            )
            self.dlb_directional_extreme = self.dlb_directional_extreme.groupby("dlc").max(dim=("group"))

        fatigue_dlcs = selected_fatigue_dlcs(self.analysis_config)
        if "directional_equivalent" in self.data and self.weight_list is not None and fatigue_dlcs:
            self.dlb_directional_equivalent = get_DLB_eq_loads(
                eq_loads=filter_fatigue_dlcs(self.data["directional_equivalent"], fatigue_dlcs),
                weight_list=self.weight_list,
            )
            self.dlb_directional_equivalent_cases = build_directional_equivalent_cases(
                self.case_data,
                self.analysis_config.weight_params,
                fatigue_dlcs,
            )
        elif "directional_equivalent" in self.enabled_sections and not fatigue_dlcs:
            print(
                "WARNING: Skipping directional equivalent section because no fatigue DLCs "
                "remain after applying analysis_config.dlcs."
            )

    def generate_sections(self):
        for component_name, component in self.components.items():
            sections = component.enabled_sections

            if "statistics" in sections and section_is_available_for_cases("statistics", self.case_data):
                generate_statistics_section(
                    component_name,
                    component,
                    self.data,
                    self.output_dir,
                    cases=self.case_data,
                )

            if "extreme" in sections and section_is_available_for_cases("extreme", self.case_data):
                generate_extreme_section(
                    component_name,
                    component,
                    self.data,
                    self.output_dir,
                    cases=self.case_data,
                )

            if "equivalent" in sections and section_is_available("equivalent", self.data):
                generate_equivalent_section(
                    component_name,
                    component,
                    self.data,
                    self.output_dir,
                    self.dlb_eq_loads,
                    self.dlb_eq_load_cases,
                    self.analysis_config,
                )

            if "directional_extreme" in sections and section_is_available("directional_extreme", self.data):
                generate_directional_extreme_section(
                    component_name,
                    component,
                    self.output_dir,
                    self.dlb_directional_extreme,
                )

            if "directional_equivalent" in sections and section_is_available_for_cases("directional_equivalent", self.case_data):
                generate_directional_equivalent_section(
                    component_name,
                    component,
                    self.output_dir,
                    self.dlb_directional_equivalent,
                    self.dlb_directional_equivalent_cases,
                )

    def write_main_latex(self):
        write_main_latex_report(
            output_dir=self.output_dir,
            components=self.components,
            enabled_sections=self.enabled_sections,
            filename=self.report_config.main_tex_filename,
            title=self.report_config.report_title,
        )

    def print_summary(self):
        for component_name, sections in self.component_sections.items():
            print(f"Generated {component_name} sections: {', '.join(sections)}")


def generate_report(report_config, analysis_config, components):
    """Generate a DLB report from mapping or dataclass inputs and return the report object."""
    return DLBReport(report_config, analysis_config, components).generate()

