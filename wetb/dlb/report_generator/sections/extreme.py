import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from wetb.dlb.report_generator.dlb_utils import finish_subplot_grid, make_subplot_grid
from wetb.dlb.report_generator.dlb_latex import (
    extreme_load_bar_explanation_latex,
    fit_table_latex_to_page,
)
from wetb.dlb.report_generator.dlb_utils import (
    chunked,
    ensure_output_dir,
    latex_escape,
    latex_filename,
    normalize_sensor_ids,
    write_text_file,
)
from wetb.dlb.report_generator.sections.base import ReportSection


def component_sensor_caption(component_name, sensor_name):
    """Return a sensor caption that omits a duplicated component prefix."""
    component_label = " ".join(str(component_name).split())
    sensor_label = " ".join(str(sensor_name).split())
    if sensor_label.casefold().startswith(component_label.casefold()):
        return sensor_label
    return f"{component_label} {sensor_label}"


NO_SEED_AVERAGE_DLCS = {"14", "15", "23", "31", "32", "33", "41", "42"}


def _seed_from_extreme_filenames(filenames):
    """Extract seed numbers from filenames containing '_sxxxx'."""
    seeds = pd.Series(filenames.astype(str)).str.extract(r"_s(\d+)", expand=False)
    if seeds.isna().any():
        missing = filenames[seeds.isna().to_numpy()]
        raise ValueError(f"Could not extract seed from filename(s): {missing[:3]}")
    return seeds.astype(int).to_numpy()


def _dlc_seed_average_extreme(da_sensor, driver, load, metric):
    """Return one aggregated extreme value per DLC, averaging seeds where applicable."""
    required_coords = ["dlc", "wsp", "wdir"]
    missing_coords = [coord for coord in required_coords if coord not in da_sensor.coords]
    if missing_coords:
        raise ValueError(
            "plot_dlc_extreme requires coordinates "
            f"{', '.join(required_coords)} on the filename dimension. "
            f"Missing: {', '.join(missing_coords)}"
        )

    df = da_sensor.sel(driver=driver, load=load).to_dataframe(name="value").reset_index()
    df["dlc"] = df["dlc"].astype(str)

    seeded_df = df[~df["dlc"].isin(NO_SEED_AVERAGE_DLCS)].copy()
    no_seed_df = df[df["dlc"].isin(NO_SEED_AVERAGE_DLCS)].copy()
    wind_cases = []

    if not seeded_df.empty:
        seeded_df["seed"] = _seed_from_extreme_filenames(
            seeded_df["filename"].astype(str).to_numpy()
        )
        grouped_seed = seeded_df.groupby(["dlc", "wsp", "wdir", "seed"], as_index=False)["value"]
        if metric == "max":
            value_by_seed = grouped_seed.max()
        elif metric == "min":
            value_by_seed = grouped_seed.min()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        wind_cases.append(
            value_by_seed.groupby(["dlc", "wsp", "wdir"], as_index=False)["value"]
            .mean()
        )

    if not no_seed_df.empty:
        if metric == "max":
            wind_cases.append(
                no_seed_df.groupby(["dlc", "wsp", "wdir"], as_index=False)["value"]
                .max()
            )
        elif metric == "min":
            wind_cases.append(
                no_seed_df.groupby(["dlc", "wsp", "wdir"], as_index=False)["value"]
                .min()
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

    if not wind_cases:
        return pd.DataFrame(columns=["dlc", "value"])

    values_by_wind_case = pd.concat(wind_cases, ignore_index=True)

    if metric == "max":
        idx = values_by_wind_case.groupby("dlc")["value"].idxmax()
        by_dlc = values_by_wind_case.loc[idx, ["dlc", "value", "wsp", "wdir"]]
    elif metric == "min":
        idx = values_by_wind_case.groupby("dlc")["value"].idxmin()
        by_dlc = values_by_wind_case.loc[idx, ["dlc", "value", "wsp", "wdir"]]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    by_dlc["dlc_sort"] = pd.to_numeric(by_dlc["dlc"], errors="coerce")
    by_dlc = by_dlc.sort_values(["dlc_sort", "dlc"]).drop(columns="dlc_sort")
    return by_dlc


def _format_wind_annotation_value(value):
    """Format wind metadata compactly for bar annotations."""
    if pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.1f}"


def _annotations_overlap(annotations, renderer):
    """Return whether any rendered annotation bounding boxes overlap."""
    boxes = [annotation.get_window_extent(renderer) for annotation in annotations]
    return any(
        first.overlaps(second)
        for index, first in enumerate(boxes)
        for second in boxes[index + 1:]
    )


def _extreme_group_label(page_tasks):
    """Return a compact base-load range label for one grouped extreme figure."""
    bases = []
    for base, *_ in page_tasks:
        if base not in bases:
            bases.append(base)

    if not bases:
        return "group"
    if len(bases) == 1:
        return bases[0]
    return f"{bases[0]}_to_{bases[-1]}"


def plot_dlc_extreme(
    da: xr.DataArray,
    sensor_input,
    sensor_title=None,
    sort_sensors: bool = True,
    save_dir: str | None = None,
    show: bool = True,
    plots_per_file: int = 6,
):
    """
    Plot DLC extremes after seed averaging.

    For each sensor and each base load, first average values across seeds for
    each DLC/wind-speed/wind-direction case. Then plot one bar per DLC for:
        - the maximum of the averaged *_max values
        - the minimum of the averaged *_min values
    """

    case_data = da if isinstance(da, dict) else {"": da}
    case_names = list(case_data)
    first_da = case_data[case_names[0]]
    sensors = normalize_sensor_ids(sensor_input, sort_sensors)
    first_da_sel = first_da.isel(sensor_name=sensors)

    if sensor_title is None:
        sensor_title = [str(first_da_sel.sensor_name.values[i]) for i in range(len(sensors))]

    base_drivers = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    for sensor_idx, sensor_id in enumerate(sensors):
        da_ref_sensor = first_da.isel(sensor_name=sensor_id)
        plot_tasks = []

        base_index = -1
        for base in base_drivers:
            driver_max = f"{base}_max"
            driver_min = f"{base}_min"
            base_index += 1

            available_drivers = da_ref_sensor.driver.values
            # A dataset may omit unsupported base loads; skip only those plots
            # rather than failing the whole component report.
            if driver_max not in available_drivers or driver_min not in available_drivers:
                continue

            plot_tasks.append((base, base_index, driver_max, "max"))
            plot_tasks.append((base, base_index, driver_min, "min"))

        save_dir = ensure_output_dir(save_dir)

        for group_idx, page_tasks in enumerate(chunked(plot_tasks, plots_per_file), start=1):
            fig, axes = make_subplot_grid()
            annotation_groups = []

            for ax, (base, base_index, driver, metric) in zip(axes, page_tasks):
                values_by_case = {}
                all_dlcs = []
                annotations = []

                for case_name, da_case in case_data.items():
                    da_sensor = da_case.isel(sensor_name=sensor_id)
                    values = _dlc_seed_average_extreme(da_sensor, driver, base, metric)
                    values_by_case[case_name] = values.set_index("dlc")
                    all_dlcs.extend(values["dlc"].tolist())

                # Use the union so every comparison case has a stable DLC slot;
                # missing values appear as gaps rather than shifted bars.
                dlcs = list(dict.fromkeys(all_dlcs))
                dlcs = sorted(dlcs, key=lambda value: (pd.to_numeric(value, errors="coerce"), str(value)))
                x = np.arange(len(dlcs))
                width = 0.8 / len(case_names)

                for case_idx, case_name in enumerate(case_names):
                    case_values = values_by_case[case_name]
                    heights = [
                        case_values.loc[dlc, "value"] if dlc in case_values.index else np.nan
                        for dlc in dlcs
                    ]
                    offset = (case_idx - (len(case_names) - 1) / 2) * width
                    label = case_name if case_name else None
                    bars = ax.bar(x + offset, heights, width=width, label=label)
                    for bar, dlc, height in zip(bars, dlcs, heights):
                        if pd.isna(height) or dlc not in case_values.index:
                            continue
                        wsp = _format_wind_annotation_value(case_values.loc[dlc, "wsp"])
                        wdir = _format_wind_annotation_value(case_values.loc[dlc, "wdir"])
                        annotation = f"wsp:{wsp}\nwdir:{wdir}"
                        va = "bottom" if height >= 0 else "top"
                        y_offset = 3 if height >= 0 else -3
                        annotations.append(ax.annotate(
                            annotation,
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, y_offset),
                            textcoords="offset points",
                            ha="center",
                            va=va,
                            rotation=90,
                            fontsize=10,
                        ))

                labels = [f"DLC {dlc}" for dlc in dlcs]

                ax.set_xlabel("DLC")
                ax.set_ylabel(base +" ["+str(da_ref_sensor.sensor_unit.values[base_index] + "]"))
                metric_label = "Max of seed-averaged max" if metric == "max" else "Min of seed-averaged min"
                ax.set_title(f"{da_ref_sensor.sensor_name.values} - {base} {metric_label}", fontsize=14)
                ax.set_xticks(x, labels, rotation=45, ha="right")
                # Leave room for the enlarged wind-case tags above or below
                # bars without letting them collide with titles or x labels.
                ax.margins(y=0.4)
                ax.grid(True, axis="y")
                if len(case_names) > 1:
                    ax.legend(fontsize=12)
                annotation_groups.append((ax, annotations))

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for ax, annotations in annotation_groups:
                if _annotations_overlap(annotations, renderer):
                    # Vertical labels are compact, but switch to one line when
                    # their rendered bounding boxes would overlap.
                    for annotation in annotations:
                        annotation.set_text(annotation.get_text().replace("\n", ", "))
                    ax.margins(y=0.5)

            save_path = None
            if save_dir is not None:
                group_label = _extreme_group_label(page_tasks)
                fname = f"{str(sensor_title[sensor_idx]).replace(' ', '_').replace(':', '').replace('/', '_')}_extreme_{group_label}.png"
                save_path = save_dir / fname

            finish_subplot_grid(fig, axes, len(page_tasks), save_path=save_path, show=show)

class ExtremeLoadsSection(ReportSection):
    """Generate extreme-load tables and figures for report components."""
    key = "extreme"
    raw_dataset_key = "extreme"
    precomputed_dataset_key = "dlb_extreme"
    psf_dataset_key = "dlb_extreme_psf"
    dataset_keys = (raw_dataset_key, precomputed_dataset_key, psf_dataset_key)
    latex_title = "Extreme Loads"
    latex_table_subdirectory = "latex_tables/extreme"
    latex_figure_subdirectory = "figures/extreme"

    def latex_figure_intro(self):
        """Explain the extreme-load bar plots before including them."""
        return extreme_load_bar_explanation_latex()

    def is_available(self):
        """Return whether raw or precomputed extreme loads are available."""
        reference_data = self.report.case_data[self.report.reference_case]
        return (
            self.raw_dataset_key in reference_data
            or self.precomputed_dataset_key in reference_data
        )

    @staticmethod
    def _highlight_diagonal(dataframe, float_format):
        """Format a square load matrix with bold diagonal values for LaTeX."""
        display = dataframe.copy().astype(object)
        for row_label in display.index:
            row_base = str(row_label).replace("\\_", "_").split("_", 1)[0]
            for column in display.columns:
                column_name = str(column).replace("\\_", "_")
                value = f"{display.loc[row_label, column]:{float_format[1:]}}"
                display.loc[row_label, column] = (
                    f"\\cellcolor{{gray!20}}{value}"
                    if row_base in {"Fx", "Fy", "Fz", "Mx", "My", "Mz"} and column_name == row_base
                    else value
                )
        return display

    @classmethod
    def dlc_extreme_loads_to_latex(
        cls, loads, sensor_input, caption="DLC extreme loads summary",
        label="tab:dlc_extreme", float_format="%.3f", output_path=None,
    ):
        """Render the precomputed extreme-load matrix as a LaTeX table."""
        selected = loads[sensor_input] if sensor_input is not None else loads
        dataframe = selected[0].to_pandas()
        dataframe.index = [latex_escape(index) for index in dataframe.index]
        dataframe.columns = [latex_escape(column) for column in dataframe.columns]
        dataframe = cls._highlight_diagonal(dataframe, float_format)
        dataframe.index = [rf"\textbf{{{index}}}" for index in dataframe.index]
        latex = fit_table_latex_to_page(dataframe.to_latex(
            bold_rows=False, caption=caption, label=label, escape=False,
        ))
        if output_path is not None:
            write_text_file(output_path, latex)
        return latex

    @staticmethod
    def diagonal_extreme_loads_to_latex(
        extreme_data, sensor_input, caption="Diagonal extreme loads source files",
        label="tab:diagonal_extreme_sources", float_format="%.3f", output_path=None,
    ):
        """Render diagonal extreme values and their source filenames as LaTeX."""
        selected = extreme_data.isel(sensor_name=sensor_input)
        loads = set(map(str, selected.coords["load"].values))
        drivers = set(map(str, selected.coords["driver"].values))
        rows = []
        for base in ("Fx", "Fy", "Fz", "Mx", "My", "Mz"):
            for suffix, reducer in (("max", "argmax"), ("min", "argmin")):
                driver = f"{base}_{suffix}"
                if base not in loads or driver not in drivers:
                    continue
                values = selected.sel(load=base, driver=driver)
                if not values.filename.size:
                    continue
                index = int(getattr(values, reducer)(dim="filename").item())
                rows.append({"Load": latex_escape(base), "Driver": latex_escape(driver),
                             "Value": values.isel(filename=index).item(),
                             "DLC": latex_escape(values.dlc.values[index] if "dlc" in values.coords else ""),
                             "File": latex_filename(values.filename.values[index])})
        dataframe = pd.DataFrame(rows)
        if dataframe.empty:
            return dataframe
        latex = fit_table_latex_to_page(dataframe.to_latex(
            index=False, float_format=float_format, caption=caption, label=label,
            escape=False, longtable=False,
        ))
        if output_path is not None:
            write_text_file(output_path, latex)
        return dataframe

    def generate(self, component_name, component):
        """Generate extreme-load tables and figures for one component."""
        raw_sensor_ids = component.sensor_ids.get(self.raw_dataset_key, [])
        matrix_sensor_ids = component.sensor_ids.get(self.precomputed_dataset_key, [])
        sensor_ids = raw_sensor_ids or matrix_sensor_ids
        source_key = (
            self.raw_dataset_key
            if raw_sensor_ids else self.precomputed_dataset_key
        )
        if not sensor_ids:
            return

        component_path = self.component_path(component_name)
        base_path = self.report.output_dir / component_path
        cases = self.report.case_data
        reference_data = self.report.case_data[self.report.reference_case]
        selected = reference_data[source_key].isel(sensor_name=sensor_ids)

        for case_name in cases:
            table_folder = base_path / "latex_tables" / "extreme"
            if len(cases) > 1:
                table_folder /= self.component_path(case_name)
            self.clear_matching_files(table_folder, "*.tex")

        for local_idx, sensor_id in enumerate(sensor_ids):
            name = str(selected["sensor_name"][local_idx].values)
            slug = name.strip().lower().replace(" ", "_").replace(":", "")
            caption_target = component_sensor_caption(component_name, name)

            for case_name, case_data in cases.items():
                case_path = self.component_path(case_name)
                table_folder = base_path / "latex_tables" / "extreme"
                if len(cases) > 1:
                    table_folder /= case_path
                extreme_matrix = case_data.get(self.precomputed_dataset_key)
                case_selected = case_data.get(self.raw_dataset_key)

                if extreme_matrix is not None:
                    self.dlc_extreme_loads_to_latex(
                        extreme_matrix, sensor_input=[sensor_id],
                        caption=f"Extreme loads for {caption_target} ({case_name})",
                        label=f"tab:extreme_loads_{component_path}_{slug}_{case_path}",
                        output_path=str(table_folder / f"{component_path}_{slug}.tex"),
                    )
                elif case_selected is not None:
                    self.diagonal_extreme_loads_to_latex(
                        case_selected.isel(sensor_name=[sensor_id]), sensor_input=0,
                        caption=(f"Diagonal extreme loads and source files for "
                                 f"{caption_target} ({case_name})"),
                        label=f"tab:diagonal_extreme_sources_{component_path}_{slug}_{case_path}",
                        output_path=str(table_folder / f"{component_path}_{slug}_diagonal_sources.tex"),
                    )

            if source_key != self.raw_dataset_key:
                continue
            if len(cases) == 1:
                plot_dlc_extreme(selected, sensor_input=[local_idx],
                                 sensor_title=[component_path],
                                 save_dir=str(base_path / "figures" / "extreme"), show=False)
            else:
                plot_dlc_extreme(
                    {
                        case_name: case_data[self.raw_dataset_key]
                        for case_name, case_data in cases.items()
                    },
                    sensor_input=[sensor_id], sensor_title=[component_path],
                    save_dir=str(base_path / "figures" / "extreme"), show=False,
                )
