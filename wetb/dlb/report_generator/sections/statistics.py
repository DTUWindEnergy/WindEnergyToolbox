import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from wetb.dlb.report_generator.dlb_latex import (
    fit_table_latex_to_page,
    include_png_files_by_folder,
)
from wetb.dlb.report_generator.dlb_utils import finish_subplot_grid, make_subplot_grid
from wetb.dlb.report_generator.sections.base import ReportSection
from wetb.dlb.report_generator.dlb_utils import (
    chunked,
    ensure_output_dir,
    latex_escape,
    latex_filename,
    normalize_sensor_ids,
    write_text_file,
)


def plot_dlc_statistics(da: xr.DataArray, sensor_input, sensor_title=None,
                        sort_sensors=True, save_dir=None, show=True, plots_per_file=6):
    """Plot per-DLC statistics for selected sensors."""
    sensors = normalize_sensor_ids(sensor_input, sort_sensors)
    da_sel = da.isel(sensor_name=sensors)
    if sensor_title is None:
        sensor_title = [str(da_sel.sensor_name.values[i]) for i in range(len(sensors))]
    for sensor in range(len(sensors)):
        da_sensor = da_sel.isel(sensor_name=sensor)
        # Each task represents one DLC. Page the tasks before plotting so a
        # component with many DLCs does not create unreadable figure grids.
        tasks = [(dlc, da_sensor.where(da_sensor.dlc == dlc, drop=True))
                 for dlc in np.unique(da_sel.dlc.values)]
        tasks = [(dlc, values) for dlc, values in tasks if values.filename.size]
        output_dir = ensure_output_dir(save_dir)
        for page in chunked(tasks, plots_per_file):
            fig, axes = make_subplot_grid(n_panels=len(page))
            for ax, (dlc, values) in zip(axes, page):
                # Wind speed is stored per simulation, so sort it explicitly
                # before connecting statistics with a visual series.
                values = values.sortby("wsp")
                for statistic, marker, color in (("mean", "o", "red"), ("min", "v", "blue"), ("max", "^", "black")):
                    ax.plot(values.wsp.values, values.sel(statistic=statistic).values,
                            linestyle="", marker=marker, color=color, label=statistic.title())
                ax.set(xlabel="Wind Speed [m/s]", ylabel=values.sensor_unit.values,
                       title=f"{sensor_title[sensor]} {values.sensor_name.values} - DLC {dlc}")
                ax.legend(fontsize=8); ax.grid(True)
            save_path = None
            if output_dir is not None:
                first, last = page[0][0], page[-1][0]
                name = str(da_sensor.sensor_name.values).replace(" ", "_").replace(":", "").replace("/", "_")
                save_path = output_dir / f"{sensor_title[sensor]}_{name}_DLC{first}_to_DLC{last}.png"
            finish_subplot_grid(fig, axes, len(page), save_path=save_path, show=show)


def plot_dlc_statistics_cases(case_data, sensor_input, sensor_title=None,
                              sort_sensors=True, save_dir=None, show=True, plots_per_file=6):
    """Overlay per-DLC statistics for validated comparison cases."""
    case_names = list(case_data)
    first_da = case_data[case_names[0]]
    # Comparison-case validation guarantees that this reference supplies
    # compatible sensor labels and units for all cases.
    sensors = normalize_sensor_ids(sensor_input, sort_sensors)
    if sensor_title is None:
        sensor_title = [str(first_da.sensor_name.values[i]) for i in sensors]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    all_dlcs = sorted({str(dlc) for da in case_data.values()
                       for dlc in np.unique(da.isel(sensor_name=sensors).dlc.values)})
    for sensor_idx, sensor_id in enumerate(sensors):
        output_dir = ensure_output_dir(save_dir)
        for page in chunked(all_dlcs, plots_per_file):
            fig, axes = make_subplot_grid(n_panels=len(page))
            for ax, dlc in zip(axes, page):
                plotted = False
                for case_idx, (case_name, da) in enumerate(case_data.items()):
                    values = da.isel(sensor_name=sensor_id).where(da.dlc == dlc, drop=True)
                    if not values.filename.size:
                        continue
                    values = values.sortby("wsp")
                    for statistic, marker in (("mean", "o"), ("min", "v"), ("max", "^")):
                        # Colour denotes case; marker denotes the statistic,
                        # allowing both dimensions to remain distinguishable.
                        ax.plot(values.wsp.values, values.sel(statistic=statistic).values,
                                linestyle="", marker=marker, color=colors[case_idx % len(colors)],
                                label=f"{case_name} {statistic}")
                    plotted = True
                if plotted:
                    reference = first_da.isel(sensor_name=sensor_id)
                    ax.set(xlabel="Wind Speed [m/s]", ylabel=reference.sensor_unit.values,
                           title=f"{sensor_title[sensor_idx]} {reference.sensor_name.values} - DLC {dlc}")
                    ax.legend(fontsize=8); ax.grid(True)
                else:
                    ax.axis("off")
            save_path = None
            if output_dir is not None:
                first, last = page[0], page[-1]
                name = str(first_da.sensor_name.values[sensor_id])
                name = name.replace(" ", "_").replace(":", "").replace("/", "_")
                save_path = output_dir / (
                    f"{sensor_title[sensor_idx]}_{name}_DLC{first}_to_DLC{last}.png"
                )
            finish_subplot_grid(
                fig, axes, len(page), save_path=save_path, show=show)


class StatisticsSection(ReportSection):
    """Generate DLC statistics tables and figures for report components."""
    key = "statistics"
    dataset_key = "statistics"
    dataset_keys = (dataset_key,)
    latex_title = "Statistics"
    latex_table_subdirectory = "latex_tables/stats"
    latex_figure_subdirectory = "figures/stats"

    def render_latex_figures(self, figure_folder, relative_to=None):
        """Group statistics figures by their source folder in LaTeX."""
        return include_png_files_by_folder(figure_folder, relative_to)

    def is_available(self):
        """Return whether statistics input data is available."""
        return self.dataset_key in self.report.case_data[self.report.reference_case]

    @staticmethod
    def dlc_statistics_to_latex(
        da, sensor_input, caption="DLC statistics summary", label="tab:dlc_stats",
        float_format="%.3f", output_path=None,
    ):
        """Render governing max/min statistics for selected sensors as LaTeX."""
        rows = []
        values = da.isel(sensor_name=sensor_input).to_dataframe("value").reset_index()
        for (dlc, sensor_name), group in values.groupby(["dlc", "sensor_name"], sort=True):
            maximum = group[group.statistic == "max"]
            minimum = group[group.statistic == "min"]
            if maximum.empty or minimum.empty:
                continue
            max_idx, min_idx = maximum.value.idxmax(), minimum.value.idxmin()
            # The governing values and their source filenames must come from
            # the same rows, hence index the grouped frame rather than its
            # independently reduced numeric arrays.
            rows.append({
                "DLC": dlc, "Sensor": latex_escape(sensor_name),
                "max of max": maximum.loc[max_idx, "value"],
                "min of min": minimum.loc[min_idx, "value"],
                "file of max": latex_filename(maximum.loc[max_idx, "filename"]),
                "file of min": latex_filename(minimum.loc[min_idx, "filename"]),
            })
        latex = fit_table_latex_to_page(pd.DataFrame(rows).to_latex(
            index=False, float_format=float_format, caption=caption, label=label,
            escape=False, longtable=False,
        ))
        if output_path is not None:
            write_text_file(output_path, latex)
        return latex

    @classmethod
    def dlc_statistics_tables_to_latex_file(
        cls, da, sensor_input, caption_prefix="DLC statistics summary",
        label_prefix="tab:dlc_stats", float_format="%.3f", output_path=None,
    ):
        """Render one statistics table for every selected sensor."""
        tables = []
        for sensor_id in sensor_input:
            name = str(da.sensor_name.values[sensor_id])
            slug = name.replace(" ", "_").replace(":", "").replace("/", "_")
            tables.append(cls.dlc_statistics_to_latex(
                da, [sensor_id], f"{caption_prefix} {name}",
                f"{label_prefix}_{slug}", float_format,
            ))
        latex = "\n\n".join(tables)
        if output_path is not None:
            write_text_file(output_path, latex)
        return latex

    def generate(self, component_name, component):
        """Generate statistics tables and figures for one component."""
        sensor_ids = component.sensor_ids.get(self.dataset_key, [])
        if not sensor_ids:
            return

        component_path = self.component_path(component_name)
        base_path = self.report.output_dir / component_path
        cases = self.report.case_data

        for case_name, case_data in cases.items():
            case_path = self.component_path(case_name)
            table_folder = base_path / "latex_tables" / "stats"
            if len(cases) > 1:
                # Preserve each case's table as a separate input artifact;
                # comparison is represented by the generated plot instead.
                table_folder /= case_path
            self.clear_matching_files(table_folder, "*.tex")
            self.dlc_statistics_tables_to_latex_file(
                case_data[self.dataset_key], sensor_input=sensor_ids,
                caption_prefix=f"Statistics loads per DLC for {component_name} ({case_name})",
                label_prefix=f"tab:statistics_loads_{component_path}_{case_path}",
                output_path=str(table_folder / f"{component_path}.tex"),
            )

        reference_data = self.report.case_data[self.report.reference_case]
        for sensor_id in sensor_ids:
            name = str(reference_data[self.dataset_key]["sensor_name"][sensor_id].values)
            slug = name.strip().lower().replace(" ", "_").replace(":", "")
            if len(cases) == 1:
                plot_dlc_statistics(
                    reference_data[self.dataset_key], sensor_input=[sensor_id],
                    sensor_title=[component_path],
                    save_dir=str(base_path / "figures" / "stats" / f"{component_path}_{slug}"),
                    show=False,
                )
            else:
                plot_dlc_statistics_cases(
                    {case_name: case_data[self.dataset_key] for case_name, case_data in cases.items()},
                    sensor_input=[sensor_id], sensor_title=[component_path],
                    save_dir=str(base_path / "figures" / "stats" / "comparison" /
                                 f"{component_path}_{slug}"),
                    show=False,
                )
