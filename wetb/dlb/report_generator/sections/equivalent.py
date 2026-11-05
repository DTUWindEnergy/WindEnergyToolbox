import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from wetb.dlb.report_generator.dlb_latex import (
    detect_force,
    fit_table_latex_to_page,
    include_equivalent_png_files,
    latex_escape,
)
from wetb.dlb.report_generator.dlb_utils import finish_figure, finish_subplot_grid, make_subplot_grid
from wetb.dlb.report_generator.dlb_utils import (
    chunked, ensure_output_dir, normalize_m_values, normalize_sensor_ids, write_text_file,
)
from wetb.dlb.report_generator.sections.base import ReportSection


def _equivalent_group_label(page_tasks):
    """Return a compact DLC range label for one grouped equivalent-load figure."""
    dlcs = [str(dlc_group[0]) for dlc_group in page_tasks if len(dlc_group)]

    if not dlcs:
        return "group"
    if len(dlcs) == 1:
        return str(dlcs[0]) if str(dlcs[0]).upper().startswith("DLC") else f"DLC{dlcs[0]}"
    first_dlc = str(dlcs[0]) if str(dlcs[0]).upper().startswith("DLC") else f"DLC{dlcs[0]}"
    last_dlc = str(dlcs[-1]) if str(dlcs[-1]).upper().startswith("DLC") else f"DLC{dlcs[-1]}"
    return f"{first_dlc}_to_{last_dlc}"


def plot_dlb_equivalent_bar(
    dlb_equivalent_loads,
    sensor_input,
    m_value,
    component=None,
    sort_sensors: bool = True,
    save_dir: str | None = None,
    show: bool = True,
):
    """
    Plot long-term DEL summary bars for selected sensors.

    dlb_equivalent_loads can be either one weighted equivalent-load DataArray or
    a dict of case_name -> weighted equivalent-load DataArray for comparison.
    """
    if isinstance(dlb_equivalent_loads, dict):
        case_loads = dlb_equivalent_loads
    else:
        # Use a synthetic label to give the single-case and comparison paths
        # the same grouped-bar implementation.
        case_loads = {"DEL": dlb_equivalent_loads}

    sensors = normalize_sensor_ids(sensor_input, sort_sensors)
    m_values = normalize_m_values(m_value, sensors)

    reference = next(iter(case_loads.values()))
    selected = reference.isel(sensor_name=sensors)
    units = [
        unit for unit in np.unique(selected.sensor_unit.values)
        if str(unit) != "nan"
    ]

    # Loads with different physical units cannot share a meaningful y-axis.
    # Each unit therefore receives a separate row of grouped bars.

    save_dir = ensure_output_dir(save_dir)
    if save_dir is not None:
        title = component or "Equivalent Loads"
        safe_component = str(title).replace(" ", "_").replace(":", "")
        for stale_file in save_dir.glob(f"{safe_component}_DEL_summary_*.png"):
            stale_file.unlink()

    if not units:
        return

    title = component or "Equivalent Loads"
    case_names = list(case_loads.keys())
    max_labels = max(
        len([
            sensor_id for sensor_id in sensors
            if str(reference.sensor_unit.values[sensor_id]) == str(unit)
        ])
        for unit in units
    )
    fig_height = max(4.0 * len(units), 4.5)
    fig_width = max(8, max_labels * 1.2)
    fig, axes = plt.subplots(
        len(units),
        1,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    for ax, unit in zip(axes.ravel(), units):
        unit_sensor_positions = [
            pos for pos, sensor_id in enumerate(sensors)
            if str(reference.sensor_unit.values[sensor_id]) == str(unit)
        ]
        if not unit_sensor_positions:
            continue

        labels = [
            str(reference.sensor_name.values[sensors[pos]])
            for pos in unit_sensor_positions
        ]
        x = np.arange(len(labels))
        width = min(0.8 / max(len(case_names), 1), 0.35)

        for case_idx, case_name in enumerate(case_names):
            da_case = case_loads[case_name]
            values = []

            for pos in unit_sensor_positions:
                sensor_id = sensors[pos]
                m_val = m_values[pos]
                value = da_case.isel(sensor_name=sensor_id).sel(m=m_val).item()
                values.append(value)

            # Centre the case bars around each sensor tick regardless of the
            # number of compared report cases.
            offset = (case_idx - (len(case_names) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, label=str(case_name))

        ax.set_xlabel("Sensor name")
        ax.set_ylabel(str(unit))
        ax.set_title(f"{str(unit)}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        if len(case_names) > 1:
            ax.legend()

    fig.suptitle(f"{title} DEL summary")
    fig.tight_layout()

    save_path = None
    if save_dir is not None:
        fname = f"{safe_component}_DEL_summary.png"
        save_path = save_dir / fname

    finish_figure(fig, save_path=save_path, show=show, close_all=True)


def plot_fatigue_summary_damage(
    damage,
    sensor_input,
    m_values,
    fatigue_dlcs=None,
    wind_speed_dlc=None,
    component=None,
    save_dir=None,
    show=True,
):
    """Save one page-sized figure per sensor with DLC pie and wind-speed bars."""
    if fatigue_dlcs is not None:
        damage = damage.sel(dlc=damage.dlc.astype(str).isin([str(d) for d in fatigue_dlcs]))
    if not damage.sizes['dlc']:
        return []
    dlc_by_label = {str(dlc): dlc for dlc in damage.dlc.values}
    if wind_speed_dlc is None or str(wind_speed_dlc) not in dlc_by_label:
        wind_speed_dlc = next(iter(dlc_by_label))
    damage_by_dlc = damage.sum('wsp')
    damage_by_wsp = damage.sel(dlc=dlc_by_label[str(wind_speed_dlc)])

    sensors = list(sensor_input)
    m_values = normalize_m_values(m_values, sensors, name="m_values")
    save_dir = ensure_output_dir(save_dir)

    output_paths = []
    dlc_values = [str(dlc) for dlc in damage_by_dlc.dlc.values]
    pie_labels = [str(dlc) if str(dlc).upper().startswith("DLC") else f"DLC{dlc}" for dlc in dlc_values]
    wsp_values = damage_by_wsp.wsp.values

    for sensor_id, m_value in zip(sensors, m_values):
        sensor_dlc_damage = damage_by_dlc.isel(sensor_name=sensor_id).sel(m=m_value)
        pie_values = np.asarray(sensor_dlc_damage.values, dtype=float)
        total = pie_values.sum()
        if not np.isfinite(total) or total <= 0:
            continue

        sensor_wsp_damage = damage_by_wsp.isel(sensor_name=sensor_id).sel(m=m_value)
        bar_values = np.asarray(sensor_wsp_damage.values, dtype=float)
        if not np.any(np.isfinite(bar_values)):
            continue

        percentages = pie_values / total * 100
        sensor_name = str(damage.sensor_name.values[sensor_id])
        title_prefix = f"{component} " if component else ""

        fig, (ax_pie, ax_bar) = plt.subplots(
            2,
            1,
            figsize=(8.0, 10.5),
            gridspec_kw={"height_ratios": [1.05, 1.0]},
        )
        wedges, _ = ax_pie.pie(percentages, startangle=90)
        legend_labels = [
            f"{label}: {percentage:.1f}%"
            for label, percentage in zip(pie_labels, percentages)
        ]
        ax_pie.legend(
            wedges,
            legend_labels,
            title="Load cases",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
        )
        ax_pie.set_title(f"{title_prefix}{sensor_name} damage contribution, m={m_value}")
        ax_pie.axis("equal")

        ax_bar.bar(wsp_values, bar_values, width=1.2)
        ax_bar.set_xlabel("Wind speed [m/s]")
        ax_bar.set_ylabel("Fatigue damage contribution")
        ax_bar.set_title(f"{str(wind_speed_dlc) if str(wind_speed_dlc).upper().startswith('DLC') else f'DLC{wind_speed_dlc}'} damage by wind speed")
        ax_bar.grid(axis="y", alpha=0.3)
        ax_bar.set_xticks(wsp_values)

        fig.tight_layout()

        save_path = None
        if save_dir is not None:
            filename = (
                f"{str(title_prefix + sensor_name).replace(' ', '_').replace(':', '').replace('/', '_')}"
                f"_fatigue_damage_summary_m{str(m_value).replace(' ', '_').replace(':', '').replace('/', '_')}.png"
            )
            save_path = save_dir / filename
            output_paths.append(save_path)

        finish_figure(fig, save_path=save_path, show=show)

    return output_paths


def plot_fatigue_summary_damage_cases(
    case_data, sensor_input, m_values, fatigue_dlcs=None, wind_speed_dlc=None,
    component=None, save_dir=None, show=True,
):
    """Compare fatigue-damage contributions across report cases."""
    case_names = list(case_data)
    reference = case_data[case_names[0]]
    if fatigue_dlcs is not None:
        # Apply the user-facing fatigue filter before summing over wind-speed
        # bins so excluded DLCs cannot contribute to pie-chart percentages.
        case_data = {
            name: damage.sel(dlc=damage.dlc.astype(str).isin([str(d) for d in fatigue_dlcs]))
            for name, damage in case_data.items()
        }
        reference = case_data[case_names[0]]
    if not reference.sizes["dlc"]:
        return []

    reference_dlcs = {str(dlc): dlc for dlc in reference.dlc.values}
    if wind_speed_dlc is None or str(wind_speed_dlc) not in reference_dlcs:
        # The first available DLC is a robust fallback for reduced datasets.
        wind_speed_dlc = next(iter(reference_dlcs))

    sensors = list(sensor_input)
    m_values = normalize_m_values(m_values, sensors, name="m_values")
    save_dir = ensure_output_dir(save_dir)
    output_paths = []
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for sensor_id, m_value in zip(sensors, m_values):
        figure_data = []
        for case_name, damage in case_data.items():
            dlc_by_label = {str(dlc): dlc for dlc in damage.dlc.values}
            if not damage.sizes["dlc"] or str(wind_speed_dlc) not in dlc_by_label:
                continue
            by_dlc = damage.sum("wsp").isel(sensor_name=sensor_id).sel(m=m_value)
            pie_values = np.asarray(by_dlc.values, dtype=float)
            total = pie_values.sum()
            by_wsp = damage.sel(dlc=dlc_by_label[str(wind_speed_dlc)]).isel(
                sensor_name=sensor_id).sel(m=m_value)
            bar_values = np.asarray(by_wsp.values, dtype=float)
            if not np.isfinite(total) or total <= 0 or not np.any(np.isfinite(bar_values)):
                continue
            figure_data.append((case_name, by_dlc.dlc.values, pie_values / total * 100,
                                by_wsp.wsp.values, bar_values))
        if not figure_data:
            continue

        sensor_name = str(reference.sensor_name.values[sensor_id])
        title_prefix = f"{component} " if component else ""
        fig = plt.figure(figsize=(5 * len(figure_data), 10.5))
        grid = fig.add_gridspec(2, len(figure_data), height_ratios=[1.05, 1.0])
        for index, (case_name, dlcs, percentages, _, _) in enumerate(figure_data):
            ax_pie = fig.add_subplot(grid[0, index])
            wedges, _ = ax_pie.pie(percentages, startangle=90)
            labels = [
                f"{str(dlc) if str(dlc).upper().startswith('DLC') else f'DLC{dlc}'}: {percentage:.1f}%"
                for dlc, percentage in zip(dlcs, percentages)
            ]
            ax_pie.legend(wedges, labels, title="Load cases", loc="center left",
                          bbox_to_anchor=(1.0, 0.5), frameon=False)
            ax_pie.set_title(str(case_name))
            ax_pie.axis("equal")

        ax_bar = fig.add_subplot(grid[1, :])
        wsp_values = sorted({float(wsp) for _, _, _, wsps, _ in figure_data for wsp in wsps})
        width = 0.8 / len(figure_data)
        for index, (case_name, _, _, wsps, values) in enumerate(figure_data):
            values_by_wsp = dict(zip(map(float, wsps), values))
            heights = [values_by_wsp.get(wsp, np.nan) for wsp in wsp_values]
            offset = (index - (len(figure_data) - 1) / 2) * width
            ax_bar.bar(np.arange(len(wsp_values)) + offset, heights, width=width,
                       color=colors[index % len(colors)], label=str(case_name))
        dlc_label = (str(wind_speed_dlc) if str(wind_speed_dlc).upper().startswith("DLC")
                     else f"DLC{wind_speed_dlc}")
        ax_bar.set(xlabel="Wind speed [m/s]", ylabel="Fatigue damage contribution",
                   title=f"{dlc_label} damage by wind speed")
        ax_bar.set_xticks(np.arange(len(wsp_values)), wsp_values)
        ax_bar.grid(axis="y", alpha=0.3)
        ax_bar.legend()
        fig.suptitle(f"{title_prefix}{sensor_name} damage contribution, m={m_value}")
        fig.tight_layout()

        if save_dir is not None:
            filename = (
                f"{str(title_prefix + sensor_name).replace(' ', '_').replace(':', '').replace('/', '_')}"
                f"_fatigue_damage_comparison_m{m_value}.png"
            )
            save_path = save_dir / filename
            output_paths.append(save_path)
        else:
            save_path = None
        finish_figure(fig, save_path=save_path, show=show)
    return output_paths


def plot_dlb_equivalent(
    da: xr.DataArray,
    sensor_input,
    m_value,
    component=None,
    sort_sensors: bool = True,
    save_dir: str | None = None,
    show: bool = True,
    plots_per_file: int = 6,
):
    """
    Plot driver extremes vs wind speed.

    For each sensor and each base load (Fx, Fy, Fz, Mx, My, Mz),
    create figures with up to 4 DLCs plotted in the same axes.

    Each DLC contributes two curves:
        - *_max
        - *_min
    """

    sensors = normalize_sensor_ids(sensor_input, sort_sensors)
    m_value = normalize_m_values(m_value, sensors)
    
    da_sel = da.isel(sensor_name=sensors)
    m_idx = np.searchsorted(da_sel.m.values, m_value)

    for sensor in range(len(sensors)):
        da_sensor = da_sel.isel(sensor_name=sensor, m = m_idx[sensor])
        plot_tasks = []

        dlcs = np.unique(da_sensor.dlc.values)
        dlcs = np.sort(dlcs)

        dlc_in_plot = 1 # number of dlcs plotted in the figure

        n_pages = (len(dlcs) + dlc_in_plot-1) // dlc_in_plot


        for page in range(n_pages):
            dlc_group = dlcs[page * dlc_in_plot : (page + 1) * dlc_in_plot]
            plot_tasks.append(dlc_group)

        save_dir = ensure_output_dir(save_dir)

        for group_idx, page_tasks in enumerate(chunked(plot_tasks, plots_per_file), start=1):
            fig, axes = make_subplot_grid(n_panels=len(page_tasks))

            for ax, dlc_group in zip(axes, page_tasks):
                for dlc in dlc_group:
                    da_dlc = da_sensor.where(da_sensor.dlc == dlc, drop=True)

                    if da_dlc.filename.size == 0:
                        continue

                    da_dlc = da_dlc.sortby("wsp")
                    wsp_vals = da_dlc.wsp.values

                    ax.plot(
                        wsp_vals,
                        da_dlc,
                        marker="o",
                        linestyle="",
                        label=f"DLC {dlc}",
                    )

                sensor_name = str(da_sensor.sensor_name.values)
                if component is None:
                    title = (
                        str(da_sensor.sensor_description.values)
                        if "sensor_description" in da_sensor.coords
                        else sensor_name
                    )
                else:
                    title = f"{component} {sensor_name}"


                ax.set_xlabel("Wind Speed [m/s]")
                ax.set_ylabel(
                    f"Short term Eq Load [{str(da_sensor.sensor_unit.values)}]"
                )
                ax.set_title(
                    f"{title} - (DLC {dlc_group[0]})",
                    fontsize=14,
                )
                ax.grid(True)
                ax.legend(fontsize=12)

            save_path = None
            if save_dir is not None:
                group_label = _equivalent_group_label(page_tasks)
                fname = f"{str(title).replace(' ', '_').replace(':', '').replace('/', '_')}_equivalent_{group_label}.png"
                save_path = save_dir / fname

            finish_subplot_grid(fig, axes, len(page_tasks), save_path=save_path, show=show)


def plot_dlb_equivalent_cases(
    case_data, sensor_input, m_value, component=None, sort_sensors=True,
    save_dir=None, show=True, plots_per_file=6,
):
    """Overlay short-term equivalent loads from multiple report cases."""
    case_names = list(case_data)
    reference = case_data[case_names[0]]
    sensors = normalize_sensor_ids(sensor_input, sort_sensors)
    m_values = normalize_m_values(m_value, sensors)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    save_dir = ensure_output_dir(save_dir)

    for sensor_id, m_val in zip(sensors, m_values):
        reference_sensor = reference.isel(sensor_name=sensor_id).sel(m=m_val)
        dlcs = sorted({str(dlc) for data in case_data.values()
                       for dlc in np.unique(data.isel(sensor_name=sensor_id).dlc.values)})
        plot_tasks = [[dlc] for dlc in dlcs]
        title = (f"{component} {reference_sensor.sensor_name.values}"
                 if component is not None else str(reference_sensor.sensor_name.values))

        for page_tasks in chunked(plot_tasks, plots_per_file):
            fig, axes = make_subplot_grid(n_panels=len(page_tasks))
            for ax, dlc_group in zip(axes, page_tasks):
                dlc = dlc_group[0]
                for case_index, (case_name, data) in enumerate(case_data.items()):
                    values = data.isel(sensor_name=sensor_id).sel(m=m_val)
                    values = values.where(values.dlc.astype(str) == dlc, drop=True)
                    if not values.filename.size:
                        continue
                    values = values.sortby("wsp")
                    ax.plot(values.wsp.values, values.values, marker="o", linestyle="",
                            color=colors[case_index % len(colors)], label=str(case_name))
                ax.set(
                    xlabel="Wind Speed [m/s]",
                    ylabel=f"Short term Eq Load [{reference_sensor.sensor_unit.values}]",
                )
                ax.set_title(f"{title} - (DLC {dlc})", fontsize=14)
                ax.grid(True)
                ax.legend(fontsize=12)

            save_path = None
            if save_dir is not None:
                group_label = _equivalent_group_label(page_tasks)
                filename = (
                    f"{str(title).replace(' ', '_').replace(':', '').replace('/', '_')}"
                    f"_equivalent_{group_label}.png"
                )
                save_path = save_dir / filename
            finish_subplot_grid(fig, axes, len(page_tasks), save_path=save_path, show=show)




class EquivalentLoadsSection(ReportSection):
    """Generate lifetime and short-term equivalent-load report outputs."""
    key = "equivalent"
    short_term_dataset_key = "equivalent"
    precomputed_dataset_key = "dlb_equivalent"
    damage_dataset_key = "damage"
    dataset_keys = (
        short_term_dataset_key,
        precomputed_dataset_key,
        damage_dataset_key,
    )
    latex_title = "Equivalent Loads"
    latex_table_subdirectory = "latex_tables/eq_load"
    latex_figure_subdirectory = "figures/equivalent"

    def render_latex_figures(self, figure_folder, relative_to=None):
        """Group equivalent-load figures by sensor and comparison output."""
        return include_equivalent_png_files(figure_folder, relative_to)

    def is_available(self):
        """Return whether any equivalent-load input is available."""
        reference_data = self.report.case_data[self.report.reference_case]
        return any(dataset_key in reference_data for dataset_key in self.dataset_keys)

    @staticmethod
    def dlb_equivalent_loads_to_latex(
        equivalent_loads, sensor_input, m_values, component="", special_name=None,
        output_path=None, float_format="%.3f", caption="DLB Custom Sensor Ranking",
        label="tab:dlb_custom",
    ):
        """Render selected lifetime equivalent loads as a LaTeX table."""
        if isinstance(equivalent_loads, dict):
            case_datasets, reference_loads = equivalent_loads, next(iter(equivalent_loads.values()))
        else:
            case_datasets, reference_loads = {None: equivalent_loads}, equivalent_loads
        if len(sensor_input) != len(m_values):
            raise ValueError("selected_sensors and m_values must have same length")
        if special_name is not None and len(special_name) != len(sensor_input):
            raise ValueError("special_name must match length of selected_sensors")

        rows = []
        for position, (index, m_value) in enumerate(zip(sensor_input, m_values)):
            original_name = str(reference_loads.sensor_name.values[index])
            sensor_name = special_name[position] if special_name is not None else original_name
            for case_name, case_loads in case_datasets.items():
                row = {
                    "Sensor": latex_escape(sensor_name if special_name else component + sensor_name),
                    "m": m_value,
                    "Value": case_loads.isel(sensor_name=index).sel(m=m_value).item(),
                    "Unit": latex_escape(reference_loads.sensor_unit.values[index]),
                    "Description": latex_escape(reference_loads.sensor_description.values[index]),
                }
                if special_name:
                    row["Force"] = latex_escape(detect_force(original_name))
                if case_name is not None:
                    row["Case"] = latex_escape(case_name)
                rows.append(row)

        dataframe = pd.DataFrame(rows)
        dataframe = dataframe.sort_values(
            [column for column in ["Sensor", "m", "Case", "Value"] if column in dataframe.columns]
        ).reset_index(drop=True)
        counts, multirow, previous = dataframe["Sensor"].value_counts().to_dict(), [], None
        for sensor in dataframe["Sensor"]:
            multirow.append(f"\\multirow{{{counts[sensor]}}}{{*}}{{{sensor}}}" if sensor != previous else "")
            previous = sensor
        dataframe["Sensor"] = multirow
        columns = ["Sensor", "Force"] if special_name else ["Sensor"]
        if "Case" in dataframe:
            columns.append("Case")
        dataframe = dataframe[columns + ["m", "Value", "Unit", "Description"]]
        latex = fit_table_latex_to_page(dataframe.to_latex(
            index=False, float_format=float_format, longtable=False, escape=False,
            caption=caption, label=label, column_format="l" * len(dataframe.columns),
        ))
        if output_path:
            write_text_file(output_path, latex)
        return dataframe

    def generate(self, component_name, component):
        """Generate equivalent-load tables and figures for one component."""
        component_path = self.component_path(component_name)
        base_path = self.report.output_dir / component_path
        data = self.report.case_data[self.report.reference_case]
        aggregate = data.get(self.precomputed_dataset_key)
        aggregate_cases = {name: case_data[self.precomputed_dataset_key]
                           for name, case_data in self.report.case_data.items()
                           if self.precomputed_dataset_key in case_data}
        aggregate_ids = component.sensor_ids.get(self.precomputed_dataset_key, [])
        if aggregate is not None and aggregate_ids:
            m_values = normalize_m_values(component.fatigue_m, aggregate_ids)
            values = aggregate_cases if len(aggregate_cases) > 1 else aggregate
            self.dlb_equivalent_loads_to_latex(
                values, sensor_input=aggregate_ids, m_values=m_values,
                component=f"{component_name} ", caption=f"Equivalent loads for {component_name}",
                label=f"tab:Equivalent_loads_{component_path}",
                output_path=str(base_path / "latex_tables" / "eq_load" / f"{component_path}.tex"),
            )
            plot_dlb_equivalent_bar(
                values, sensor_input=aggregate_ids, m_value=m_values, component=component_name,
                save_dir=str(base_path / "figures" / "equivalent" / "summary"), show=False,
            )

        summary_dir = base_path / "figures" / "equivalent" / "summary"
        for stale_pattern in ("*_damage_contribution_m*.png", "*_wind_speed_damage_*.png"):
            self.clear_matching_files(summary_dir, stale_pattern)
        if self.damage_dataset_key in data:
            damage_ids = component.sensor_ids.get(self.damage_dataset_key, [])
            if damage_ids:
                damage_cases = {
                    case_name: case_data[self.damage_dataset_key]
                    for case_name, case_data in self.report.case_data.items()
                }
                damage_values = damage_cases if len(damage_cases) > 1 else data[self.damage_dataset_key]
                damage_plot = (
                    plot_fatigue_summary_damage_cases
                    if len(damage_cases) > 1 else plot_fatigue_summary_damage
                )
                damage_plot(
                    damage_values, sensor_input=damage_ids,
                    m_values=normalize_m_values(component.fatigue_m, damage_ids),
                    fatigue_dlcs=self.selected_fatigue_dlcs(), component=component_name,
                    save_dir=summary_dir, show=False,
                )

        if self.short_term_dataset_key not in data:
            print(
                f"WARNING: Skipping short-term equivalent-load plots for "
                f"'{component_name}': no raw 'equivalent' dataset was loaded. "
                "Precomputed DEL tables, summary plots, and damage figures for "
                "supplied datasets will still be generated."
            )
            return

        sensor_ids = component.sensor_ids.get(self.short_term_dataset_key, [])
        m_values = normalize_m_values(component.fatigue_m, sensor_ids)
        equivalent_cases = {
            case_name: case_data[self.short_term_dataset_key]
            for case_name, case_data in self.report.case_data.items()
        }
        for local_idx, sensor_id in enumerate(sensor_ids):
            name = str(data[self.short_term_dataset_key]["sensor_name"][sensor_id].values)
            slug = name.strip().lower().replace(" ", "_").replace(":", "")
            short_term_dir = base_path / "figures" / "equivalent" / f"{component_path}_{slug}"
            values = data[self.short_term_dataset_key]
            plot_function = plot_dlb_equivalent
            if len(equivalent_cases) > 1:
                short_term_dir = base_path / "figures" / "equivalent" / "comparison" / f"{component_path}_{slug}"
                values = equivalent_cases
                plot_function = plot_dlb_equivalent_cases
            short_term_dir.mkdir(parents=True, exist_ok=True)
            self.clear_matching_files(short_term_dir, "*.png")
            plot_function(
                values, sensor_input=[sensor_id],
                m_value=[m_values[local_idx]], component=component_name,
                save_dir=str(short_term_dir), show=False,
            )
