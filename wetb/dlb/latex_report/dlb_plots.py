import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from dlb_postprocs import get_DLB_eq_loads
from dlb_utils import *


NO_SEED_AVERAGE_DLCS = {"14", "15", "23", "31", "32", "33", "41", "42"}


def make_subplot_grid(count, polar=False):
    """Create a 2 by 3 subplot page and return flattened axes."""
    subplot_kw = {"polar": True} if polar else None
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 18),
        subplot_kw=subplot_kw,
        squeeze=False,
    )
    return fig, axes.ravel()


def finish_subplot_grid(fig, axes, used_count, save_path=None, show=True):
    """Hide unused axes, save, and close a grouped figure."""
    for ax in axes[used_count:]:
        ax.axis("off")

    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show, close_all=True)


def finish_figure(fig, save_path=None, show=True, close_all=False):
    """Save, show, and close a matplotlib figure consistently."""
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if close_all:
        plt.close("all")


def plot_dlc_statistics(
    da: xr.DataArray,
    sensor_input,
    sensor_title = None,
    sort_sensors: bool = True,
    save_dir: str | None = None,
    show: bool = True,
    plots_per_file: int = 6,
):
    """
    Plot statistics vs wind speed for each DLC and sensor.

    Parameters
    ----------
    da : xr.DataArray
        Dimensions: (filename, sensor_name, statistic)

    sensor_input : list-like
        Sensors to include.

    sensor_titles : list-like
        Optinal list of sensor names to use for plot titles.

    sort_sensors : bool
        If True, sensors are sorted alphabetically.

    save_dir : str or None
        If provided, figures are saved in this directory.

    show : bool
        If True, displays figures. If False, only saves.
    """

    # -------------------------------------------------
    # 1. Select sensors
    # -------------------------------------------------
    sensors = normalize_sensor_ids(sensor_input, sort_sensors)
    da_sel = da.isel(sensor_name=sensors)

    if sensor_title is None:
        sensor_title = [str(da_sel.sensor_name.values[i]) for i in range(len(sensors))]


    # -------------------------------------------------
    # 4. Loop through sensors and DLCs
    # -------------------------------------------------
    dlcs = np.unique(da_sel.dlc.values)

    for sensor in range(len(sensors)):
        da_sensor = da_sel.isel(sensor_name=sensor)
        plot_tasks = []

        for dlc in dlcs:
            da_dlc = da_sensor.where(da_sensor.dlc == dlc, drop=True)

            if da_dlc.filename.size == 0:
                continue

            plot_tasks.append((dlc, da_dlc))

        save_dir = ensure_output_dir(save_dir)

        for page_idx, page_tasks in enumerate(chunked(plot_tasks, plots_per_file), start=1):
            fig, axes = make_subplot_grid(len(page_tasks))

            for ax, (dlc, da_dlc) in zip(axes, page_tasks):
            # sort by wind speed
                da_dlc = da_dlc.sortby("wsp")

                wsp_vals = da_dlc.wsp.values

                mean_vals = da_dlc.sel(statistic="mean").values
                min_vals = da_dlc.sel(statistic="min").values
                max_vals = da_dlc.sel(statistic="max").values

            # -----------------------------------------
            # Plot
            # -----------------------------------------
                ax.plot(wsp_vals, mean_vals, linestyle = '', marker="o", color = 'red', label="Mean")
                ax.plot(wsp_vals, min_vals, linestyle = '', marker="v",color = 'blue', label="Min")
                ax.plot(wsp_vals, max_vals, linestyle = '', marker="^",color = 'black', label="Max")

                ax.set_xlabel("Wind Speed [m/s]")
                ax.set_ylabel(da_dlc.sensor_unit.values)
                ax.set_title(f"{sensor_title[sensor]} {da_dlc.sensor_name.values} - DLC {dlc}", fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True)

            # -----------------------------------------
            # Save if requested
            # -----------------------------------------
            save_path = None
            if save_dir is not None:
                first_dlc = page_tasks[0][0]
                last_dlc = page_tasks[-1][0]
                sensor_name = da_sensor.sensor_name.values
                fname = (
                    f"{safe_filename(sensor_title[sensor])}_"
                    f"{safe_filename(sensor_name)}_{dlc_label(first_dlc)}_to_{dlc_label(last_dlc)}.png"
                )
                save_path = save_dir / fname

            finish_subplot_grid(fig, axes, len(page_tasks), save_path=save_path, show=show)


def plot_dlc_statistics_cases(
    case_data,
    sensor_input,
    sensor_title=None,
    sort_sensors: bool = True,
    save_dir: str | None = None,
    show: bool = True,
    plots_per_file: int = 6,
):
    """Overlay statistics plots for one or two postproc cases."""
    case_names = list(case_data)
    first_da = case_data[case_names[0]]
    sensors = normalize_sensor_ids(sensor_input, sort_sensors)

    if sensor_title is None:
        sensor_title = [str(first_da.sensor_name.values[i]) for i in sensors]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    case_colors = {
        case_name: colors[idx % len(colors)]
        for idx, case_name in enumerate(case_names)
    }
    statistic_markers = {
        "mean": "o",
        "min": "v",
        "max": "^",
    }

    all_dlcs = sorted(
        {
            str(dlc)
            for da in case_data.values()
            for dlc in np.unique(da.isel(sensor_name=sensors).dlc.values)
        }
    )

    for sensor_idx, sensor_id in enumerate(sensors):
        plot_tasks = []

        for dlc in all_dlcs:
            plot_tasks.append(dlc)

        save_dir = ensure_output_dir(save_dir)

        for page_idx, page_tasks in enumerate(chunked(plot_tasks, plots_per_file), start=1):
            fig, axes = make_subplot_grid(len(page_tasks))
            used_count = 0

            for ax, dlc in zip(axes, page_tasks):
                plotted = False

                for case_name, da in case_data.items():
                    da_sensor = da.isel(sensor_name=sensor_id)
                    da_dlc = da_sensor.where(da_sensor.dlc == dlc, drop=True)

                    if da_dlc.filename.size == 0:
                        continue

                    da_dlc = da_dlc.sortby("wsp")
                    wsp_vals = da_dlc.wsp.values

                    for statistic, marker in statistic_markers.items():
                        ax.plot(
                            wsp_vals,
                            da_dlc.sel(statistic=statistic).values,
                            linestyle="",
                            marker=marker,
                            color=case_colors[case_name],
                            label=f"{case_name} {statistic}",
                        )
                        plotted = True

                if not plotted:
                    ax.axis("off")
                    continue

                da_ref = first_da.isel(sensor_name=sensor_id)
                ax.set_xlabel("Wind Speed [m/s]")
                ax.set_ylabel(da_ref.sensor_unit.values)
                ax.set_title(f"{sensor_title[sensor_idx]} {da_ref.sensor_name.values} - DLC {dlc}", fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True)
                used_count += 1

            save_path = None
            if save_dir is not None:
                first_dlc = page_tasks[0]
                last_dlc = page_tasks[-1]
                sensor_name = first_da.sensor_name.values[sensor_id]
                fname = (
                    f"{safe_filename(sensor_title[sensor_idx])}_"
                    f"{safe_filename(sensor_name)}_{dlc_label(first_dlc)}_to_{dlc_label(last_dlc)}_comparison.png"
                )
                save_path = save_dir / fname

            if used_count:
                finish_subplot_grid(fig, axes, len(page_tasks), save_path=save_path, show=show)
            else:
                plt.close(fig)


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


def _equivalent_group_label(page_tasks):
    """Return a compact DLC range label for one grouped equivalent-load figure."""
    dlcs = [str(dlc_group[0]) for dlc_group in page_tasks if len(dlc_group)]

    if not dlcs:
        return "group"
    if len(dlcs) == 1:
        return dlc_label(dlcs[0])
    return f"{dlc_label(dlcs[0])}_to_{dlc_label(dlcs[-1])}"


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
            if driver_max not in available_drivers or driver_min not in available_drivers:
                continue

            plot_tasks.append((base, base_index, driver_max, "max"))
            plot_tasks.append((base, base_index, driver_min, "min"))

        save_dir = ensure_output_dir(save_dir)

        for group_idx, page_tasks in enumerate(chunked(plot_tasks, plots_per_file), start=1):
            fig, axes = make_subplot_grid(len(page_tasks))

            for ax, (base, base_index, driver, metric) in zip(axes, page_tasks):
                values_by_case = {}
                all_dlcs = []

                for case_name, da_case in case_data.items():
                    da_sensor = da_case.isel(sensor_name=sensor_id)
                    values = _dlc_seed_average_extreme(da_sensor, driver, base, metric)
                    values_by_case[case_name] = values.set_index("dlc")
                    all_dlcs.extend(values["dlc"].tolist())

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
                        ax.annotate(
                            annotation,
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, y_offset),
                            textcoords="offset points",
                            ha="center",
                            va=va,
                            rotation=90,
                            fontsize=6,
                        )

                labels = [f"DLC {dlc}" for dlc in dlcs]

                ax.set_xlabel("DLC")
                ax.set_ylabel(base +" ["+str(da_ref_sensor.sensor_unit.values[base_index] + "]"))
                metric_label = "Max of seed-averaged max" if metric == "max" else "Min of seed-averaged min"
                ax.set_title(f"{da_ref_sensor.sensor_name.values} - {base} {metric_label}", fontsize=10)
                ax.set_xticks(x, labels, rotation=45, ha="right")
                ax.grid(True, axis="y")
                if len(case_names) > 1:
                    ax.legend(fontsize=8)

            save_path = None
            if save_dir is not None:
                group_label = _extreme_group_label(page_tasks)
                fname = f"{safe_filename(sensor_title[sensor_idx])}_extreme_{group_label}.png"
                save_path = save_dir / fname

            finish_subplot_grid(fig, axes, len(page_tasks), save_path=save_path, show=show)


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
        case_loads = {"DEL": dlb_equivalent_loads}

    sensors = normalize_sensor_ids(sensor_input, sort_sensors)
    m_values = normalize_m_values(m_value, sensors)

    reference = next(iter(case_loads.values()))
    selected = reference.isel(sensor_name=sensors)
    units = [
        unit for unit in np.unique(selected.sensor_unit.values)
        if str(unit) != "nan"
    ]

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


def fatigue_damage_by_dlc(
    eq_loads,
    weight_list,
    fatigue_dlcs=None,
    neq=1e7,
    neq_individual=xr.DataArray(
        data=[600, 600, 100, 100, 600],
        dims=("dlc",),
        coords={"dlc": ["12", "24", "31", "41", "64"]},
    ),
):
    """Return unnormalized fatigue damage contribution per DLC."""
    if isinstance(weight_list, dict):
        weight_list = xr.DataArray(
            data=list(weight_list.values()),
            dims=("filename",),
            coords={"filename": list(weight_list.keys())},
        )

    fatigue_dlcs = [str(dlc) for dlc in (fatigue_dlcs or neq_individual.dlc.values)]
    available_dlcs = [
        dlc for dlc in fatigue_dlcs
        if dlc in set(eq_loads.dlc.astype(str).values)
    ]
    if not available_dlcs:
        return None

    damage_parts = []
    damage_dlcs = []
    for dlc in available_dlcs:
        eq_loads_dlc = eq_loads.where(eq_loads.dlc.astype(str) == dlc, drop=True)
        if eq_loads_dlc.filename.size == 0:
            continue

        weight_list_dlc = weight_list.sel(filename=eq_loads_dlc.filename)
        neq_dlc = neq_individual.sel(dlc=[dlc])
        dlb_eq_loads_dlc = get_DLB_eq_loads(
            eq_loads_dlc,
            weight_list_dlc,
            neq=neq,
            neq_individual=neq_dlc,
        )
        damage_parts.append((dlb_eq_loads_dlc ** dlb_eq_loads_dlc.m) * neq)
        damage_dlcs.append(dlc)

    if not damage_parts:
        return None

    damage_by_dlc = xr.concat(damage_parts, dim=xr.DataArray(
        data=damage_dlcs,
        dims=("dlc",),
        coords={"dlc": damage_dlcs},
    ))

    if "variable" in damage_by_dlc.coords:
        damage_by_dlc = damage_by_dlc.drop_vars("variable")

    return damage_by_dlc


def plot_fatigue_dlc_contribution(
    eq_loads,
    weight_list,
    sensor_input,
    m_values,
    fatigue_dlcs=None,
    component=None,
    save_dir=None,
    show=True,
):
    """Save one pie chart per selected sensor showing fatigue damage by DLC."""
    damage_by_dlc = fatigue_damage_by_dlc(
        eq_loads,
        weight_list,
        fatigue_dlcs=fatigue_dlcs,
    )
    if damage_by_dlc is None:
        return []

    sensors = list(sensor_input)
    m_values = normalize_m_values(m_values, sensors, name="m_values")
    save_dir = ensure_output_dir(save_dir)

    output_paths = []
    dlc_values = [str(dlc) for dlc in damage_by_dlc.dlc.values]
    labels = [dlc_label(dlc) for dlc in dlc_values]

    for sensor_id, m_value in zip(sensors, m_values):
        sensor_damage = damage_by_dlc.isel(sensor_name=sensor_id).sel(m=m_value)
        values = np.asarray(sensor_damage.values, dtype=float)
        total = values.sum()
        if not np.isfinite(total) or total <= 0:
            continue

        percentages = values / total * 100
        sensor_name = str(eq_loads.sensor_name.values[sensor_id])
        title_prefix = f"{component} " if component else ""

        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, _ = ax.pie(percentages, startangle=90)
        legend_labels = [
            f"{label}: {percentage:.1f}%"
            for label, percentage in zip(labels, percentages)
        ]
        ax.legend(
            wedges,
            legend_labels,
            title="Load cases",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
        )
        ax.set_title(f"{title_prefix}{sensor_name} damage contribution, m={m_value}")
        ax.axis("equal")
        fig.tight_layout()

        save_path = None
        if save_dir is not None:
            filename = (
                f"{safe_filename(title_prefix + sensor_name)}"
                f"_damage_contribution_m{safe_filename(m_value)}.png"
            )
            save_path = save_dir / filename
            output_paths.append(save_path)

        finish_figure(fig, save_path=save_path, show=show)

    return output_paths


def fatigue_damage_by_wind_speed(
    eq_loads,
    weight_list,
    dlc="12",
    neq_individual=xr.DataArray(
        data=[600, 600, 100, 100, 600],
        dims=("dlc",),
        coords={"dlc": ["12", "24", "31", "41", "64"]},
    ),
):
    """Return fatigue damage contribution for one DLC grouped by wind speed."""
    if "wsp" not in eq_loads.coords:
        raise ValueError("eq_loads must have a 'wsp' coordinate")

    if isinstance(weight_list, dict):
        weight_list = xr.DataArray(
            data=list(weight_list.values()),
            dims=("filename",),
            coords={"filename": list(weight_list.keys())},
        )

    dlc = str(dlc)
    eq_loads_dlc = eq_loads.where(eq_loads.dlc.astype(str) == dlc, drop=True)
    if eq_loads_dlc.filename.size == 0:
        return None

    weight_list_dlc = weight_list.sel(filename=eq_loads_dlc.filename)
    neq_dlc = neq_individual.sel(dlc=dlc).item()
    damage = weight_list_dlc * neq_dlc * eq_loads_dlc ** eq_loads_dlc.m
    damage_by_wsp = damage.groupby("wsp").sum("filename")

    if "variable" in damage_by_wsp.coords:
        damage_by_wsp = damage_by_wsp.drop_vars("variable")

    return damage_by_wsp


def plot_fatigue_wind_speed_damage(
    eq_loads,
    weight_list,
    sensor_input,
    m_values,
    dlc="12",
    component=None,
    save_dir=None,
    show=True,
):
    """Save one bar plot per selected sensor showing one DLC's damage by wind speed."""
    damage_by_wsp = fatigue_damage_by_wind_speed(eq_loads, weight_list, dlc=dlc)
    if damage_by_wsp is None:
        return []

    sensors = list(sensor_input)
    m_values = normalize_m_values(m_values, sensors, name="m_values")
    save_dir = ensure_output_dir(save_dir)

    output_paths = []
    wsp_values = damage_by_wsp.wsp.values

    for sensor_id, m_value in zip(sensors, m_values):
        sensor_damage = damage_by_wsp.isel(sensor_name=sensor_id).sel(m=m_value)
        values = np.asarray(sensor_damage.values, dtype=float)
        if not np.any(np.isfinite(values)):
            continue

        sensor_name = str(eq_loads.sensor_name.values[sensor_id])
        title_prefix = f"{component} " if component else ""

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(wsp_values, values, width=1.2)
        ax.set_xlabel("Wind speed [m/s]")
        ax.set_ylabel("Fatigue damage contribution")
        ax.set_title(
            f"{title_prefix}{sensor_name} {dlc_label(dlc)} damage by wind speed, "
            f"m={m_value}"
        )
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(wsp_values)
        fig.tight_layout()

        save_path = None
        if save_dir is not None:
            filename = (
                f"{safe_filename(title_prefix + sensor_name)}"
                f"_wind_speed_damage_{safe_filename(dlc_label(dlc))}"
                f"_m{safe_filename(m_value)}.png"
            )
            save_path = save_dir / filename
            output_paths.append(save_path)

        finish_figure(fig, save_path=save_path, show=show)

    return output_paths


def plot_fatigue_summary_damage(
    eq_loads,
    weight_list,
    sensor_input,
    m_values,
    fatigue_dlcs=None,
    wind_speed_dlc="12",
    component=None,
    save_dir=None,
    show=True,
):
    """Save one page-sized figure per sensor with DLC pie and wind-speed bars."""
    damage_by_dlc = fatigue_damage_by_dlc(
        eq_loads,
        weight_list,
        fatigue_dlcs=fatigue_dlcs,
    )
    damage_by_wsp = fatigue_damage_by_wind_speed(
        eq_loads,
        weight_list,
        dlc=wind_speed_dlc,
    )
    if damage_by_dlc is None or damage_by_wsp is None:
        return []

    sensors = list(sensor_input)
    m_values = normalize_m_values(m_values, sensors, name="m_values")
    save_dir = ensure_output_dir(save_dir)

    output_paths = []
    dlc_values = [str(dlc) for dlc in damage_by_dlc.dlc.values]
    pie_labels = [dlc_label(dlc) for dlc in dlc_values]
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
        sensor_name = str(eq_loads.sensor_name.values[sensor_id])
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
        ax_bar.set_title(f"{dlc_label(wind_speed_dlc)} damage by wind speed")
        ax_bar.grid(axis="y", alpha=0.3)
        ax_bar.set_xticks(wsp_values)

        fig.tight_layout()

        save_path = None
        if save_dir is not None:
            filename = (
                f"{safe_filename(title_prefix + sensor_name)}"
                f"_fatigue_damage_summary_m{safe_filename(m_value)}.png"
            )
            save_path = save_dir / filename
            output_paths.append(save_path)

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

    if component is None:
        component = [str(da_sel.sensor_name.values[i]) for i in range(len(sensors))]

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
            fig, axes = make_subplot_grid(len(page_tasks))

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

                if component == None:
                    title =  da_sensor.sensor_description.values
                else:
                    title =  component +" " + da_sensor.sensor_name.values


                ax.set_xlabel("Wind Speed [m/s]")
                ax.set_ylabel("Short term Eq Load" +" ["+str(da_sensor.sensor_unit.values + "]"))
                ax.set_title(
                    f"{title} - (DLC {dlc_group[0]})",
                    fontsize=10,
                )
                ax.grid(True)
                ax.legend(fontsize=8)

            save_path = None
            if save_dir is not None:
                group_label = _equivalent_group_label(page_tasks)
                fname = f"{safe_filename(title)}_equivalent_{group_label}.png"
                save_path = save_dir / fname

            finish_subplot_grid(fig, axes, len(page_tasks), save_path=save_path, show=show)


def plot_load_envl(
    da: xr.DataArray,
    sensor_input,
    m_value,
    component=None,
    sort_sensors: bool = True,
    save_dir: str | None = None,
    show: bool = True,
):

# step 1 

    da = da.isel(sensor_name= sensor_input)
    DLC = da.dlc.values
    angle = da.angle.values
    load_design = da.values[:, 0, :]
    plot_title = da.sensor_name.values[0]
    fig, ax = plt.subplots(1, 2, figsize=[10,6] , subplot_kw=dict(polar=True))

    idx = np.argmax(load_design, axis = 0)
    envelope = load_design#[np.arange(load_design.shape[0]), idx] 

    
    cmap = plt.get_cmap("tab20", len(DLC))

    dlc_colors = {
        dlc: cmap(i)
        for i, dlc in enumerate(DLC)
    }

    for i in range(len(DLC)):
        
        plot_angle = np.append(angle, angle[0])
        plot_ld = np.append(load_design[i,:], load_design[i,0])
        # if DLC[i] == 'DLC12':
        #     ax[0].plot(np.deg2rad(plot_angle),  plot_ld, '-o', label='DLC11')
        # else:
        col = dlc_colors.get(DLC[i], 'black')
        ax[0].plot(np.deg2rad(plot_angle),  plot_ld, '-o', color = col,label=DLC[i])
        ax[0].set_xticks(np.deg2rad(angle+150))  # Set tick positions and labels (every 30 degrees)
        
    for j in range(len(angle)):

        col = dlc_colors.get(DLC[idx[j]], 'black')
                      
        ax[1].plot(np.deg2rad(angle[j]),  envelope[idx[j],j], color = col,  linestyle='-', marker='o', label=DLC[idx[j]])
        ax[1].set_xticks(np.deg2rad(angle+150))  # Set tick positions and labels (every 30 degrees)
    
    handles, labels = ax[0].get_legend_handles_labels()
    nrow = 3

    ncol = np.ceil(len(DLC) / nrow)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.),
        ncol=ncol
    )
    fig.suptitle(plot_title)
    plt.tight_layout()

    save_path = None
    if save_dir is not None:
        save_dir = ensure_output_dir(save_dir)
        fname = f"{plot_title.replace(" ","_").replace(":","")}.png"
        save_path = save_dir / fname

    finish_figure(fig, save_path=save_path, show=show)


def plot_dir_load_envl(
    da: xr.DataArray,
    sensor_input,
    m_value,
    component=None,
    sort_sensors: bool = True,
    save_dir: str | None = None,
    show: bool = True,
):

    sensors = normalize_sensor_ids(sensor_input, sort_sensors)
    m_values = normalize_m_values(m_value, sensors)

    fig, axes = plt.subplots(
        1,
        len(sensors),
        figsize=(5 * len(sensors), 6),
        subplot_kw=dict(polar=True),
        squeeze=False,
    )

    subplot_titles = []

    for ax, sensor_id, m_val in zip(axes.ravel(), sensors, m_values):
        da_sensor = da.isel(sensor_name=[sensor_id]).sel(m=m_val)
        angle = da_sensor.angle.values
        load_design = da_sensor.values

        plot_angle = np.append(angle, angle[0])
        plot_ld = np.append(load_design[0, :], load_design[0, 0])
        plot_title = str(da_sensor.sensor_name.values[0])

        ax.plot(np.deg2rad(plot_angle), plot_ld, "-o")
        ax.set_xticks(np.deg2rad(angle + 150))
        ax.set_title(plot_title, fontsize=10)
        subplot_titles.append(plot_title)

    plt.tight_layout()

    save_path = None
    if save_dir is not None:
        save_dir = ensure_output_dir(save_dir)
        if component is None:
            safe_title = "_to_".join(safe_filename(title) for title in subplot_titles)
        else:
            safe_title = f"{safe_filename(component)}_directional_equivalent_loads"
        fname = f"{safe_title}.png"
        save_path = save_dir / fname

    finish_figure(fig, save_path=save_path, show=show)


def plot_dir_load_envl_cases(
    cases,
    sensor_input,
    m_value,
    component=None,
    sort_sensors: bool = True,
    save_dir: str | None = None,
    show: bool = True,
):
    """Overlay directional equivalent load envelopes for one sensor across cases."""
    if not cases:
        return

    sensors = normalize_sensor_ids(sensor_input, sort_sensors)
    m_values = normalize_m_values(m_value, sensors)
    save_dir = ensure_output_dir(save_dir)

    reference = next(iter(cases.values()))

    for local_idx, sensor_id in enumerate(sensors):
        m_val = m_values[local_idx]
        ref_sensor = reference.isel(sensor_name=[sensor_id]).sel(m=m_val)
        angle = ref_sensor.angle.values
        plot_angle = np.append(angle, angle[0])
        plot_title = str(ref_sensor.sensor_name.values[0])

        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))

        for case_name, da_case in cases.items():
            da_sensor = da_case.isel(sensor_name=[sensor_id]).sel(m=m_val)
            load_design = da_sensor.values
            plot_ld = np.append(load_design[0, :], load_design[0, 0])
            ax.plot(
                np.deg2rad(plot_angle),
                plot_ld,
                linestyle="-",
                marker="o",
                label=str(case_name),
            )

        ax.set_xticks(np.deg2rad(angle + 150))
        ax.legend(loc="best")

        if component is None:
            title = plot_title
        else:
            title = f"{component} {plot_title}"

        fig.suptitle(f"{title} directional equivalent comparison")
        plt.tight_layout()

        save_path = None
        if save_dir is not None:
            safe_title = title.replace(" ", "_").replace(":", "")
            fname = f"{safe_title}_directional_equivalent_comparison.png"
            save_path = save_dir / fname

        finish_figure(fig, save_path=save_path, show=show)

