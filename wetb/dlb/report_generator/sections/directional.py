import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from wetb.dlb.report_generator.dlb_utils import finish_figure
from wetb.dlb.report_generator.dlb_utils import ensure_output_dir, normalize_m_values, normalize_sensor_ids
from wetb.dlb.report_generator.sections.base import ReportSection


def component_sensor_title(component, sensor_name):
    """Prefix a sensor name only when it does not already name its component."""
    component_label = " ".join(str(component or "").split())
    sensor_label = " ".join(str(sensor_name).split())
    if not component_label or sensor_label.casefold().startswith(component_label.casefold()):
        return sensor_label
    return f"{component_label} {sensor_label}"


def plot_load_envl(
    da: xr.DataArray,
    sensor_input,
    component=None,
    sort_sensors: bool = True,
    save_dir: str | None = None,
    show: bool = True,
):
    """Plot directional extreme load envelopes for one selected sensor.

    Parameters
    ----------
    da : xarray.DataArray
        Directional extreme-load data.
    sensor_input : int
        Sensor index to plot.
    """

# step 1 

    da = da.isel(sensor_name=sensor_input)
    # ``squeeze(drop=True)`` below intentionally removes the singleton
    # ``sensor_name`` coordinate.  Keep its label for the figure title first.
    if "sensor_name" in da.coords:
        plot_title = str(np.asarray(da.coords["sensor_name"].values).reshape(-1)[0])
    else:
        plot_title = str(component or "Directional loads")
    if 'sensor_name' in da.dims:
        da = da.squeeze('sensor_name', drop=True)
    # Rows are DLCs and columns are directions. This layout lets the envelope
    # retain the DLC that governs each individual direction.
    DLC = da.dlc.values
    angle = da.angle.values
    load_design = da.values
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
        # Colour each envelope point by its governing DLC, not by a single
        # global envelope colour.
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
        ncol=ncol,
        fontsize=12,
    )
    fig.suptitle(plot_title)
    plt.tight_layout()

    save_path = None
    if save_dir is not None:
        save_dir = ensure_output_dir(save_dir)
        fname = f"{plot_title.replace(" ","_").replace(":","")}.png"
        save_path = save_dir / fname

    finish_figure(fig, save_path=save_path, show=show)


def plot_load_envl_cases(
    cases,
    sensor_input,
    component=None,
    sort_sensors: bool = True,
    save_dir: str | None = None,
    show: bool = True,
):
    """Overlay directional extreme envelopes for one sensor across cases."""
    if not cases:
        return

    sensors = normalize_sensor_ids(sensor_input, sort_sensors)
    save_dir = ensure_output_dir(save_dir)
    reference = next(iter(cases.values()))

    for local_idx, sensor_id in enumerate(sensors):
        ref_sensor = reference.isel(sensor_name=sensor_id)
        angle = ref_sensor.angle.values
        plot_angle = np.deg2rad(np.append(angle, angle[0]))
        plot_title = str(ref_sensor.sensor_name.item())

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))
        colors = plt.get_cmap("tab20")
        color_by_dlc = {}
        case_line_styles = ("-", "--", ":", "-.")
        case_markers = ("o", "s", "^", "D")

        for case_idx, (case_name, da_case) in enumerate(cases.items()):
            da_sensor = da_case.isel(sensor_name=sensor_id)
            loads = da_sensor.values
            dlcs = da_sensor.dlc.values
            linestyle = case_line_styles[case_idx % len(case_line_styles)]
            marker = case_markers[case_idx % len(case_markers)]

            for dlc_idx, dlc in enumerate(dlcs):
                # DLC colour is stable across cases; line style and marker are
                # reserved for distinguishing the compared cases.
                color_by_dlc.setdefault(str(dlc), colors(len(color_by_dlc) % 20))
                closed_load = np.append(loads[dlc_idx], loads[dlc_idx, 0])
                axes[0].plot(
                    plot_angle,
                    closed_load,
                    color=color_by_dlc[str(dlc)],
                    linestyle=linestyle,
                    marker=marker,
                    label=f"{case_name} DLC {dlc}",
                )

            envelope = loads.max(axis=0)
            axes[1].plot(
                plot_angle,
                np.append(envelope, envelope[0]),
                linestyle=linestyle,
                marker=marker,
                label=str(case_name),
            )

        for ax in axes:
            ax.set_xticks(np.deg2rad(angle + 150))
        axes[0].set_title("DLC envelopes")
        axes[1].set_title("Maximum envelope")
        title = component_sensor_title(component, plot_title)
        fig.suptitle(f"{title} directional extreme comparison")
        fig.tight_layout()
        dlc_handles = [
            Line2D([0], [0], color=color, linewidth=2, label=f"DLC {dlc}")
            for dlc, color in color_by_dlc.items()
        ]
        case_handles = [
            Line2D(
                [0], [0], color="black",
                linestyle=case_line_styles[index % len(case_line_styles)],
                marker=case_markers[index % len(case_markers)], label=str(case_name),
            )
            for index, case_name in enumerate(cases)
        ]
        axes[0].legend(handles=dlc_handles + case_handles, fontsize=12, loc="best")
        axes[1].legend(fontsize=12, loc="best")

        if save_dir is not None:
            filename = f"{str(title).replace(' ', '_').replace(':', '').replace('/', '_')}_directional_extreme_comparison.png"
            finish_figure(fig, save_path=save_dir / filename, show=show)
        else:
            finish_figure(fig, show=show)


def plot_dir_load_envl(
    da: xr.DataArray,
    sensor_input,
    m_value,
    component=None,
    sort_sensors: bool = True,
    save_dir: str | None = None,
    show: bool = True,
):
    """Plot directional equivalent-load envelopes for selected sensors.

    Parameters
    ----------
    da : xarray.DataArray
        Directional equivalent-load data.
    sensor_input : iterable of int
        Sensor indexes to plot.
    m_value : scalar or iterable
        Fatigue exponents aligned with the selected sensors.
    """

    sensors = normalize_sensor_ids(sensor_input, sort_sensors)
    m_values = normalize_m_values(m_value, sensors)

    # Keep axes two-dimensional even for one sensor so the common iteration
    # below does not require a special one-sensor branch.
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
            safe_title = "_to_".join(str(title).replace(" ", "_").replace(":", "").replace("/", "_") for title in subplot_titles)
        else:
            safe_title = f"{str(component).replace(' ', '_').replace(':', '').replace('/', '_')}_directional_equivalent_loads"
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

        case_line_styles = ("-", "--", ":", "-.")
        case_markers = ("o", "s", "^", "D")
        for case_idx, (case_name, da_case) in enumerate(cases.items()):
            da_sensor = da_case.isel(sensor_name=[sensor_id]).sel(m=m_val)
            load_design = da_sensor.values
            plot_ld = np.append(load_design[0, :], load_design[0, 0])
            ax.plot(
                np.deg2rad(plot_angle),
                plot_ld,
                linestyle=case_line_styles[case_idx % len(case_line_styles)],
                marker=case_markers[case_idx % len(case_markers)],
                label=str(case_name),
            )

        ax.set_xticks(np.deg2rad(angle + 150))
        ax.legend(loc="best")

        title = component_sensor_title(component, plot_title)

        fig.suptitle(f"{title} directional equivalent comparison")
        plt.tight_layout()

        save_path = None
        if save_dir is not None:
            safe_title = title.replace(" ", "_").replace(":", "").replace("/", "_")
            fname = f"{safe_title}_directional_equivalent_comparison.png"
            save_path = save_dir / fname

        finish_figure(fig, save_path=save_path, show=show)



class DirectionalExtremeSection(ReportSection):
    """Generate directional extreme-load figures for report components."""
    key = "directional_extreme"
    dataset_key = "dlb_directional_extreme"
    psf_dataset_key = "dlb_directional_extreme_psf"
    dataset_keys = (dataset_key, psf_dataset_key)
    latex_title = "Directional Extreme Loads"
    latex_figure_subdirectory = "figures/directional_extreme"

    def is_available(self):
        """Return whether precomputed directional extreme loads are available."""
        return self.dataset_key in self.report.case_data[self.report.reference_case]

    def generate(self, component_name, component):
        """Generate directional extreme figures for one component."""
        sensor_ids = component.sensor_ids.get(self.dataset_key, [])
        if not sensor_ids:
            return
        component_path = self.component_path(component_name)
        base_path = self.report.output_dir / component_path
        datasets = {name: data[self.dataset_key]
                    for name, data in self.report.case_data.items()}
        if len(datasets) == 2:
            # Comparison figures overlay matching sensor/DLC data only after
            # report-level validation has confirmed aligned coordinates.
            comparison_dir = base_path / "figures" / "directional_extreme" / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            self.clear_matching_files(comparison_dir, "*.png")
            for sensor_id in sensor_ids:
                plot_load_envl_cases(datasets, sensor_input=[sensor_id], component=component_name,
                                     save_dir=str(comparison_dir), show=False)
            return

        for sensor_id in sensor_ids:
            plot_load_envl(
                self.report.case_data[self.report.reference_case][self.dataset_key], sensor_input=[sensor_id],
                component=component_name,
                save_dir=str(base_path / "figures" / "directional_extreme"), show=False,
            )


class DirectionalEquivalentSection(ReportSection):
    """Generate directional equivalent-load figures for report components."""
    key = "directional_equivalent"
    dataset_key = "dlb_directional_equivalent"
    dataset_keys = (dataset_key,)
    latex_title = "Directional Equivalent Loads"
    latex_figure_subdirectory = "figures/directional_equivalent"

    def is_available(self):
        """Return whether precomputed directional equivalent loads are available."""
        return self.dataset_key in self.report.case_data[self.report.reference_case]

    def generate(self, component_name, component):
        """Generate directional equivalent-load figures for one component."""
        sensor_ids = component.sensor_ids.get(self.dataset_key, [])
        if not sensor_ids:
            return
        component_path = self.component_path(component_name)
        base_path = self.report.output_dir / component_path
        m_values = normalize_m_values(component.fatigue_m, sensor_ids)
        figure_dir = base_path / "figures" / "directional_equivalent"
        figure_dir.mkdir(parents=True, exist_ok=True)
        datasets = {name: data[self.dataset_key]
                    for name, data in self.report.case_data.items()}

        if len(datasets) == 2:
            # Equivalent loads need the per-sensor fatigue exponent supplied
            # alongside the sensor index for each comparison figure.
            comparison_dir = figure_dir / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            self.clear_matching_files(comparison_dir, "*.png")
            for local_idx, sensor_id in enumerate(sensor_ids):
                plot_dir_load_envl_cases(
                    datasets, sensor_input=[sensor_id], m_value=[m_values[local_idx]],
                    component=component_name, save_dir=str(comparison_dir), show=False,
                )
            return

        self.clear_matching_files(figure_dir, "*.png")
        plot_dir_load_envl(
            self.report.case_data[self.report.reference_case][self.dataset_key], sensor_input=sensor_ids,
            m_value=m_values, component=component_name, save_dir=str(figure_dir), show=False,
        )
