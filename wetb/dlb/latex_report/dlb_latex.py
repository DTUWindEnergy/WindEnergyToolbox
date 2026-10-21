###
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dlb_postprocs import *
from dlb_reporting import *
from dlb_utils import *
from dlb_plots import *


# LaTeX Formatting And Includes

def fit_table_latex_to_page(latex):
    """Center a LaTeX table inline and scale its tabular content to page width."""
    latex = re.sub(r"\\begin\{table\}(?:\[[^\]]*\])?", r"\\begin{center}", latex, count=1)
    latex = re.sub(r"\\end\{table\}", r"\\end{center}", latex, count=1)
    latex = re.sub(r"\\caption\{", r"\\captionof{table}{", latex, count=1)

    if "\\resizebox" not in latex:
        latex = re.sub(
            r"(\\begin\{tabular\}\{[^}]+\}.*?\\end\{tabular\})",
            r"\\resizebox{\\textwidth}{!}{%\n\1\n}",
            latex,
            count=1,
            flags=re.DOTALL,
        )

    return latex

def include_tex_files(tex_folder):
    """Recursively include generated .tex files from a folder."""
    tex_folder = Path(tex_folder)
    if not tex_folder.exists():
        return ""

    parts = []
    for tex_file in sorted(tex_folder.rglob("*.tex")):
        parts.append(f"\\input{{{latex_path(tex_file)}}}")

    return "\n".join(parts)

def include_png_file(png_file):
    """Return LaTeX for one generated PNG figure."""
    caption = figure_caption_from_path(png_file)
    return "\n".join(
        [
            "\\Needspace{0.8\\textheight}",
            "\\begin{center}",
            f"    \\includegraphics[width=\\textwidth,height=0.76\\textheight,keepaspectratio]{{{latex_path(png_file)}}}",
            f"    \\captionof{{figure}}{{{caption}}}",
            "\\end{center}",
        ]
    )

def include_png_files(plot_folder):
    """Recursively include generated .png files as one figure per grouped image."""
    plot_folder = Path(plot_folder)
    if not plot_folder.exists():
        return ""

    return "\n\n".join(include_png_file(png_file) for png_file in sorted(plot_folder.rglob("*.png")))

def include_png_files_by_folder(plot_folder):
    """Include generated .png files grouped under subsubsections by folder."""
    plot_folder = Path(plot_folder)
    if not plot_folder.exists():
        return ""

    root_pngs = sorted(plot_folder.glob("*.png"))
    child_folders = [
        folder for folder in sorted(plot_folder.rglob("*"))
        if folder.is_dir() and list(folder.glob("*.png"))
    ]

    parts = []
    if root_pngs:
        parts.append(include_png_files(plot_folder))

    for folder in child_folders:
        parts.append("\\Needspace{0.8\\textheight}")
        parts.append(f"\\subsubsection{{{figure_title_from_folder(folder)}}}")
        parts.append(include_png_files(folder))

    return "\n\n".join(parts)

def include_equivalent_png_files(plot_folder):
    """Include equivalent-load figures grouped by summary, fatigue, and scatter plots."""
    plot_folder = Path(plot_folder)
    if not plot_folder.exists():
        return ""

    summary_folder = plot_folder / "summary"
    del_summary_pngs = []
    fatigue_summary_pngs = []
    if summary_folder.exists():
        del_summary_pngs = sorted(summary_folder.glob("*DEL_summary.png"))
        fatigue_summary_pngs = sorted(summary_folder.glob("*fatigue_damage_summary*.png"))

    short_term_folders = [
        folder for folder in sorted(plot_folder.rglob("*"))
        if folder.is_dir() and list(folder.glob("*.png"))
        and folder != summary_folder
    ]

    parts = [include_png_file(png_file) for png_file in del_summary_pngs]

    if fatigue_summary_pngs:
        parts.append("\\Needspace{0.8\\textheight}")
        parts.append("\\subsubsection{Fatigue Damage Contribution}")
        parts.append(fatigue_damage_contribution_explanation_latex())
        parts.extend(include_png_file(png_file) for png_file in fatigue_summary_pngs)

    if short_term_folders:
        parts.append("\\Needspace{0.8\\textheight}")
        parts.append("\\subsubsection{Short Term Equivalent Load Plots}")
        for folder in short_term_folders:
            parts.append(include_png_files(folder))

    return "\n\n".join(parts)



# Explanation Text Blocks

def fatigue_damage_contribution_explanation_latex():
    """Return explanatory LaTeX for fatigue damage contribution figures."""
    return "\n".join(
        [
            (
                "The fatigue damage contribution figures show how the equivalent-load "
                "damage is distributed across the selected fatigue DLCs. "
            ),
            "\\[",
            "D_{j,v,\\theta} =",
            "L_{\\mathrm{eq},j,v,\\theta}^{m}",
            "\\, N_{\\mathrm{eq},j}",
            "\\, P_{j,v,\\theta}",
            "\\, \\frac{T_{\\mathrm{life}}}{N_{\\mathrm{seeds},j,v,\\theta} \\, T_{\\mathrm{sim},j}}",
            "\\]",
            (
                "Here, $D_{j,v,\\theta}$ is the damage contribution for DLC $j$, "
                "wind speed $v$, and wind direction $\\theta$; "
                "$L_{\\mathrm{eq},j,v,\\theta}$ is the short-term equivalent load "
                "for that bin; $m$ is the W\\\"ohlner component; "
                "$N_{\\mathrm{eq},j}$ is the equivalent cycle count for the DLC; "
                "$P_{j,v,\\theta}$ is the probability of the DLC, wind-speed, and "
                "wind-direction bin; $T_{\\mathrm{life}}$ is the design lifetime; "
                "$N_{\\mathrm{seeds},j,v,\\theta}$ is the number of seeds in the "
                "bin; and $T_{\\mathrm{sim},j}$ is the simulation time for the DLC."
            ),
            "\\[",
            "D_j =",
            "\\sum_v \\sum_\\theta D_{j,v,\\theta}",
            "\\]",
            (
                "The total fatigue damage contribution from DLC $j$ is found by "
                "summing over all wind speeds and wind directions."
            ),
            "\\[",
            "D_{\\mathrm{total}} =",
            "\\sum_{j \\in J} D_j",
            "\\]",
            (
                "The total fatigue damage is the sum of the DLC damage contributions "
                "over the selected fatigue DLC set $J$."
            ),
            "\\[",
            "p_j =",
            "100 \\frac{D_j}{D_{\\mathrm{total}}}",
            "\\]",
            (
                "The percentage contribution from DLC $j$ is its damage contribution "
                "divided by the total fatigue damage. The pie chart shows $p_j$ for "
                "each selected fatigue DLC. The wind-speed bar chart uses the same "
                "damage calculation, but groups the summed damage by wind speed for "
                "the selected DLC."
            ),
        ]
    )

def extreme_load_bar_explanation_latex():
    """Return explanatory LaTeX for the extreme-load bar plot calculation."""
    return "\n".join(
        [
            "\\Needspace{0.8\\textheight}",
            "\\subsubsection{Extreme Load Bar Plots}",
            (
                "The extreme-load bar plots show the governing design load per DLC. "
                "For seeded DLCs, matching simulations are first grouped by operating "
                "condition, such as wind speed and direction, and then averaged across "
                "seeds. The plotted maximum is the maximum of these seed-averaged "
                "maximum values, and the plotted minimum is the minimum of these "
                "seed-averaged minimum values."
            ),
            "\\[",
            "L_{\\max,\\mathrm{DLC}} =",
            "\\max_{g \\in G_{\\mathrm{DLC}}}",
            "\\left(",
            "\\frac{1}{N_s}\\sum_{s=1}^{N_s} L_{\\max,g,s}",
            "\\right)",
            "\\]",
            "\\[",
            "L_{\\min,\\mathrm{DLC}} =",
            "\\min_{g \\in G_{\\mathrm{DLC}}}",
            "\\left(",
            "\\frac{1}{N_s}\\sum_{s=1}^{N_s} L_{\\min,g,s}",
            "\\right)",
            "\\]",
            (
                "Here, $G_{\\mathrm{DLC}}$ is the set of operating-condition groups "
                "for the DLC, $s$ is the seed index, $N_s$ is the number of seeds in "
                "group $g$, and $L_{\\max,g,s}$ and $L_{\\min,g,s}$ are the maximum "
                "and minimum load values from seed $s$ in group $g$."
            ),
            (
                "DLC 1.4, 1.5, 2.3, 3.1, 3.2, 3.3, 4.1, and 4.2 bypass seed "
                "averaging and use direct aggregation over the simulation files:"
            ),
            "\\[",
            "L_{\\max,\\mathrm{DLC}} =",
            "\\max_{f \\in F_{\\mathrm{DLC}}} L_{\\max,f}",
            "\\]",
            "\\[",
            "L_{\\min,\\mathrm{DLC}} =",
            "\\min_{f \\in F_{\\mathrm{DLC}}} L_{\\min,f}",
            "\\]",
            (
                "where $F_{\\mathrm{DLC}}$ is the set of simulation files for the DLC."
            ),
        ]
    )



# Report Assembly

def build_report_subsection(title, table_folder, figure_folder):
    """Build one subsection from the generated tables and figures on disk."""
    tables = include_tex_files(table_folder) if table_folder is not None else ""
    if figure_folder is not None and title == "Statistics":
        figures = include_png_files_by_folder(figure_folder)
    elif figure_folder is not None and title == "Equivalent Loads":
        figures = include_equivalent_png_files(figure_folder)
    else:
        figures = include_png_files(figure_folder) if figure_folder is not None else ""

    if not tables and not figures:
        return ""

    parts = ["\\Needspace{0.8\\textheight}", f"\\subsection{{{title}}}"]
    if tables:
        parts.append(tables)
    if figures and title == "Extreme Loads":
        parts.append(extreme_load_bar_explanation_latex())
    if figures:
        parts.append(figures)

    return "\n\n".join(parts)

def build_component_section(component_name, component_label, component_dir, enabled_sections):
    """Build one component section with report subsections."""
    component_dir = Path(component_dir)
    subsection_map = {
        "statistics": (
            "Statistics",
            component_dir / "latex_tables" / "stats",
            component_dir / "figures" / "stats",
        ),
        "extreme": (
            "Extreme Loads",
            component_dir / "latex_tables" / "extreme",
            component_dir / "figures" / "extreme",
        ),
        "equivalent": (
            "Equivalent Loads",
            component_dir / "latex_tables" / "eq_load",
            component_dir / "figures" / "equivalent",
        ),
        "directional_extreme": (
            "Directional Extreme Loads",
            None,
            component_dir / "figures" / "directional_extreme",
        ),
        "directional_equivalent": (
            "Directional Equivalent Loads",
            None,
            component_dir / "figures" / "directional_equivalent",
        ),
    }

    subsections = []
    for section_key in enabled_sections:
        title, table_folder, figure_folder = subsection_map[section_key]
        subsection = build_report_subsection(title, table_folder, figure_folder)
        if subsection:
            subsections.append(subsection)

    if not subsections:
        return ""

    section_title = component_label or component_name
    return "\n\n".join([f"\\section{{{section_title}}}"] + subsections)

def write_component_latex_report(
    output_dir,
    component_name,
    component,
    enabled_sections=None,
    filename=None,
):
    """Write the LaTeX body for one component and return its path."""
    output_dir = Path(output_dir)
    component_path = component_path_name(component_name)
    component_dir = output_dir / component_path
    component_label = component.get("label", component_name)
    sections = component.get("enabled_sections", enabled_sections or [])
    component_tex = build_component_section(
        component_name,
        component_label,
        component_dir,
        sections,
    )

    if not component_tex:
        return None

    output_path = component_dir / (filename or f"{component_path}.tex")
    write_text_file(output_path, component_tex)
    return output_path

def write_main_latex_report(
    output_dir,
    components,
    enabled_sections=None,
    filename="main.tex",
    title="Component Load Report",
):
    """Write a compilable main LaTeX document for the generated report content."""
    output_dir = Path(output_dir)

    component_inputs = []
    for component_name, component in components.items():
        component_tex_path = write_component_latex_report(
            output_dir,
            component_name,
            component,
            enabled_sections=enabled_sections,
        )
        if component_tex_path is not None:
            component_inputs.append(f"\\input{{{latex_path(component_tex_path)}}}")

    document = "\n".join(
        [
            "\\documentclass[11pt]{article}",
            "\\usepackage[margin=2.5cm]{geometry}",
            "\\usepackage{graphicx}",
            "\\usepackage{float}",
            "\\usepackage{longtable}",
            "\\usepackage{multirow}",
            "\\usepackage{booktabs}",
            "\\usepackage{amsmath}",
            "\\usepackage{caption}",
            "\\usepackage[table]{xcolor}",
            "\\usepackage{placeins}",
            "\\usepackage{needspace}",
            "\\title{" + title + "}",
            "\\date{}",
            "\\begin{document}",
            "\\maketitle",
            "\\tableofcontents",
            "\\clearpage",
            "\\section{Overview}",
            "\\input{" + latex_path(output_dir / "component_table.tex") + "}",
            "\\input{" + latex_path(output_dir / "psf_table.tex") + "}",
            "\\input{" + latex_path(output_dir / "sensor_table.tex") + "}"
            if (output_dir / "sensor_table.tex").exists()
            else "",
            "\\clearpage",
            "\n\n".join(component_inputs),
            "\\end{document}",
        ]
    )

    output_path = output_dir / filename
    write_text_file(output_path, document)
    return output_path



# Table Helpers

def detect_force(name):
    name_low = name.lower()
    for p in ["fx","fy","fz","mx","my","mz","dll","cl","cd","alfa","bea","torque","power","thrust"]:
        if p in name_low:
            return p.capitalize()
    return "-"

def highlight_extreme_diagonal(df, float_format="%.3f"):
    """Highlight load cells matching the row driver base, e.g. Fx_max/Fx."""
    display_df = df.copy().astype(object)
    base_loads = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    for row_label in display_df.index:
        row_text = str(row_label).replace("\\_", "_")
        row_base = row_text.split("_", 1)[0]

        for column in display_df.columns:
            column_text = str(column).replace("\\_", "_")
            value = display_df.loc[row_label, column]
            formatted = latex_format_value(value, float_format)

            if row_base in base_loads and column_text == row_base:
                display_df.loc[row_label, column] = f"\\cellcolor{{gray!20}}{formatted}"
            else:
                display_df.loc[row_label, column] = formatted

    return display_df



# Overview Tables

def make_sensor_table(
        da,
        selected_sensors,
        sensor_components=None,
        special_name = None,
        output_dir=None,
        unit_coord=None,
        caption="Sensor overview",
        label="tab:sensor_overview",
):
    rows = []
    da_sel = da.isel(sensor_name = selected_sensors)
    if sensor_components is None:
        sensor_components = [""] * len(selected_sensors)

    if len(sensor_components) != len(selected_sensors):
        raise ValueError("sensor_components must match selected_sensors length")

    # Build raw rows
    for i in range(len(selected_sensors)):
        component = str(sensor_components[i])
        sensor_name = str(da_sel.sensor_name[i].values)
        name = f"{component} {sensor_name}".strip()
        # Detect load component
        force = detect_force(sensor_name)
        description = (
            da_sel.sensor_description.values[i]
            if "sensor_description" in da_sel.coords
            else ""
        )

        # Get unit if available
        if unit_coord and unit_coord in da.coords:
            try:
                unit = da_sel.sensor_unit.values[i]
            except Exception:
                unit = ""
        else:
            unit = ""



        if special_name:
            rows.append({
                "Sensor ID": selected_sensors[i],
                "Sensor": latex_escape(special_name[i]),
                "Description": latex_escape(description),
                "Force": latex_escape(force),
                "Unit": latex_escape(unit),
            })
        else:
             rows.append({
                "Sensor ID": selected_sensors[i],
                "Sensor": latex_escape(name),
                "Description": latex_escape(description),
                "Force": latex_escape(force),
                "Unit": latex_escape(unit),
            })           

    df = pd.DataFrame(rows)

    if df.empty:
        print("No matching sensors found.")
        return df

    # Sort
    df = df.sort_values(["Sensor", "Force", "Description"])

    # Create multirow column
    multi = []
    last = None
    counts = df["Sensor"].value_counts().to_dict()

    for sensor in df["Sensor"]:
        if sensor != last:
            multi.append(f"\\multirow{{{counts[sensor]}}}{{*}}{{{sensor}}}")
            last = sensor
        else:
            multi.append("")

    df["Sensor"] = multi

    df = df[["Sensor ID", "Sensor", "Description", "Force", "Unit"]]

    # Convert to LaTeX
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="lllll",
        caption=caption,
        label=label,
    )
    latex = fit_table_latex_to_page(latex)

    # Save if output_dir provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "sensor_table.tex")
        write_text_file(file_path, latex)
        print(f"LaTeX table saved to: {file_path}")

    return df

def make_component_table(
    standard_components,
    output_dir=None,
    filename="component_table.tex",
    caption="Component overview",
    label="tab:component_overview",
):
    rows = []

    for component, rules in standard_components.items():
        enabled_sections = rules.get("enabled_sections", [])
        if isinstance(enabled_sections, str):
            enabled_sections = [enabled_sections]

        rows.append({
            "Component": latex_escape(component),
            "Label": latex_escape(rules.get("label", component)),
            "Enabled sections": latex_escape(", ".join(map(str, enabled_sections))),
            "Wöhler component (m)": latex_escape(rules.get("fatigue_m", "")),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No components found.")
        return df

    df = df.sort_values("Component")

    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="llll",
        caption=caption,
        label=label,
    )
    latex = fit_table_latex_to_page(latex)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        write_text_file(file_path, latex)
        print(f"LaTeX table saved to: {file_path}")

    return df

def make_dlc_overview_table(
    dlcs,
    output_dir=None,
    filename="dlc_table.tex",
    caption="DLC overview",
    label="tab:dlc_overview",
):
    unique_dlcs = list(dict.fromkeys(dlcs))
    if not unique_dlcs:
        df = pd.DataFrame()
        print("No DLCs found.")
        return df

    columns = [latex_escape(dlc_label(dlc)) for dlc in unique_dlcs]
    df = pd.DataFrame([columns], columns=columns, index=["DLC"])

    latex = df.to_latex(
        index=True,
        escape=False,
        column_format="l" + "c" * len(columns),
        caption=caption,
        label=label,
    )
    latex = latex.replace("\\begin{table}", "\\begin{table}\n\\centering", 1)
    latex = re.sub(
        r"(\\begin\{tabular\}\{[^}]+\}.*?\\end\{tabular\})",
        r"\\resizebox{\\textwidth}{!}{%\n\1\n}",
        latex,
        count=1,
        flags=re.DOTALL,
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        write_text_file(file_path, latex)
        print(f"LaTeX table saved to: {file_path}")

    return df

def make_psf_table(
    dlcs,
    safety_factors,
    output_dir=None,
    filename="psf_table.tex",
    caption="Partial safety factors",
    label="tab:partial_safety_factors",
):
    rows = []

    unique_dlcs = list(dict.fromkeys(dlcs))
    if not unique_dlcs:
        df = pd.DataFrame()
        print("No partial safety factors found.")
        return df

    for dlc in unique_dlcs:
        dlc_key = dlc_label(dlc)
        rows.append((latex_escape(dlc_key), safety_factors.get(dlc_key, "")))

    if len(rows) > 10:
        split_idx = int(np.ceil(len(rows) / 2))
        row_groups = [rows[:split_idx], rows[split_idx:]]
    else:
        row_groups = [rows]

    max_group_len = max(len(group) for group in row_groups)
    table_rows = []
    for group in row_groups:
        padded_group = group + [("", "")] * (max_group_len - len(group))
        table_rows.append(["DLC"] + [dlc for dlc, _ in padded_group])
        table_rows.append(["Partial safety factor"] + [factor for _, factor in padded_group])

    df = pd.DataFrame(table_rows)
    column_format = "l" + "c" * max_group_len
    table_body = "\n".join(
        " & ".join(latex_format_value(value) for value in row) + r" \\"
        for row in table_rows
    )
    latex = "\n".join(
        [
            "\\begin{table}",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\resizebox{\\textwidth}{!}{%",
            f"\\begin{{tabular}}{{{column_format}}}",
            "\\toprule",
            table_body,
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}",
        ]
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        write_text_file(file_path, latex)
        print(f"LaTeX table saved to: {file_path}")

    return df



# Statistics Tables

def dlc_statistics_to_latex(
    da: xr.DataArray,
    sensor_input,
    caption: str = "DLC statistics summary",
    label: str = "tab:dlc_stats",
    float_format: str = "%.3f",
    output_path: str | None = None,
) -> str:
    """
    Compute aggregated statistics per DLC for selected sensors
    and return a LaTeX table. Optionally saves to file.

    Parameters
    ----------
    da : xr.DataArray
        Dimensions: (filename, sensor_name, statistic)

    sensor_input : list-like
        List of sensor names to include.

    caption : str
        LaTeX table caption.

    label : str
        LaTeX label for referencing.

    float_format : str
        Format string for floats (e.g. "%.3f").

    output_path : str or None
        If provided, saves LaTeX table to this file.

    Returns
    -------
    str
        LaTeX table as string.
    """

    # -------------------------------------------------
    # 1. Select requested sensors
    # -------------------------------------------------
    da_sel = da.isel(sensor_name=sensor_input)

    # -------------------------------------------------
    # 2. Extract DLC from filename
    # -------------------------------------------------
    # dlc_coord = da_sel.filename.str.split("/").str[0]
    # da_sel = da_sel.assign_coords(dlc=("filename", dlc_coord))

    values = da_sel.to_dataframe("value").reset_index()
    group_columns = ["dlc", "sensor_name"]
    rows = []

    for (dlc, sensor_name), group in values.groupby(group_columns, sort=True):
        max_rows = group[group["statistic"] == "max"]
        min_rows = group[group["statistic"] == "min"]
        if max_rows.empty or min_rows.empty:
            continue

        max_idx = max_rows["value"].idxmax()
        min_idx = min_rows["value"].idxmin()
        rows.append({
            "DLC": dlc,
            "Sensor": latex_escape(sensor_name),
            "max of max": max_rows.loc[max_idx, "value"],
            "min of min": min_rows.loc[min_idx, "value"],
            "file of max": latex_escape(max_rows.loc[max_idx, "filename"]),
            "file of min": latex_escape(min_rows.loc[min_idx, "filename"]),
        })

    df = pd.DataFrame(rows)
        # -------------------------------------------------
        # 7. Generate LaTeX table
        # -------------------------------------------------
    latex_table = df.to_latex(
            index=False,
            float_format=float_format,
            caption=caption,
            label=label,
            escape=True,
            longtable=False,
        )
    latex_table = fit_table_latex_to_page(latex_table)

    # -------------------------------------------------
    # 8. Optionally save to file
    # -------------------------------------------------
    if output_path is not None:
        write_text_file(output_path, latex_table)

    return latex_table

def dlc_statistics_tables_to_latex_file(
    da: xr.DataArray,
    sensor_input,
    caption_prefix: str = "DLC statistics summary",
    label_prefix: str = "tab:dlc_stats",
    float_format: str = "%.3f",
    output_path: str | None = None,
) -> str:
    """Write one .tex file containing one separate statistics table per sensor."""
    tables = []

    for sensor_id in sensor_input:
        sensor_name = str(da.sensor_name.values[sensor_id])
        sensor_slug = safe_filename(sensor_name)
        table = dlc_statistics_to_latex(
            da,
            sensor_input=[sensor_id],
            caption=f"{caption_prefix} {sensor_name}",
            label=f"{label_prefix}_{sensor_slug}",
            float_format=float_format,
            output_path=None,
        )
        tables.append(table)

    latex = "\n\n".join(tables)

    if output_path is not None:
        write_text_file(output_path, latex)

    return latex



# Extreme Load Tables

def dlc_extreme_loads_to_latex(
    dlc_extreme_loads: xr.DataArray,
    sensor_input,
	caption: str = "DLC extreme loads summary",
    label: str = "tab:dlc_extreme",
    float_format: str = "%.3f",
	output_path: str | None = None,
):
    """
    Generate LaTeX tables for selected sensors.

    Parameters
    ----------
    dlc_extreme_loads : xarray.DataArray
    output_dir : str
        Directory where .tex files will be saved
    sensors : list or None
        Explicit list of sensor names to include (overrides start/stop)
    sort_sensors : bool
        If True, sensors will be sorted alphabetically
    start : int or None
        Start index for slicing sensor list
    stop : int or None
        Stop index for slicing sensor list
    float_format : str
        Float formatting for LaTeX table
    """

    

    # Get all sensor names
    if sensor_input is not None:
        da_sel = dlc_extreme_loads[sensor_input]#.isel(sensor_name=sensor_input)
    else:
        da_sel = dlc_extreme_loads

    df = da_sel[0].to_pandas()
    if df.index.dtype == object:
        df.index = [latex_escape(idx) for idx in df.index]
    if list(df.columns):
        df.columns = [latex_escape(col) for col in df.columns]
    df = highlight_extreme_diagonal(df, float_format=float_format)

    latex_table = df.to_latex(
        bold_rows=True,
        caption=caption,
        label=label,
        escape=False,
    )
    latex_table = fit_table_latex_to_page(latex_table)

    if output_path is not None:
        write_text_file(output_path, latex_table)

    print(f"Written: {output_path}.tex")

def diagonal_extreme_loads_to_latex(
    extreme_data: xr.DataArray,
    sensor_input,
    caption: str = "Diagonal extreme loads source files",
    label: str = "tab:diagonal_extreme_sources",
    float_format: str = "%.3f",
    output_path: str | None = None,
):
    """Write a table of diagonal extreme values and their source filenames."""
    da_sensor = extreme_data.isel(sensor_name=sensor_input)
    load_values = set(map(str, da_sensor.coords["load"].values))
    driver_values = set(map(str, da_sensor.coords["driver"].values))
    rows = []

    for base in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
        for suffix, reducer in [("max", "idxmax"), ("min", "idxmin")]:
            driver = f"{base}_{suffix}"
            if base not in load_values or driver not in driver_values:
                continue

            values = da_sensor.sel(load=base, driver=driver)
            if values.filename.size == 0:
                continue

            if reducer == "idxmax":
                idx = int(values.argmax(dim="filename").item())
            else:
                idx = int(values.argmin(dim="filename").item())

            filename = values.filename.values[idx]
            dlc = values.dlc.values[idx] if "dlc" in values.coords else ""
            rows.append(
                {
                    "Load": latex_escape(base),
                    "Driver": latex_escape(driver),
                    "Value": values.isel(filename=idx).item(),
                    "DLC": latex_escape(dlc),
                    "File": latex_escape(filename),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    latex_table = df.to_latex(
        index=False,
        float_format=float_format,
        caption=caption,
        label=label,
        escape=False,
        longtable=False,
    )
    latex_table = fit_table_latex_to_page(latex_table)

    if output_path is not None:
        write_text_file(output_path, latex_table)

    return df



# Equivalent Load Tables

def DLB_equivalent_loads_to_latex(
    DLB_equivalent_loads,
    sensor_input,
    m_values,
    component = "",
    special_name=None,
    output_path=None,
    float_format="%.3f",
    caption="DLB Custom Sensor Ranking",
    label="tab:dlb_custom",
):
    """
    Generate one LaTeX table using selected sensors and
    corresponding m values, with optional special names
    and multirow grouping.
    """
    if isinstance(DLB_equivalent_loads, dict):
        case_loads = DLB_equivalent_loads
        reference_loads = next(iter(case_loads.values()))
    else:
        case_loads = {None: DLB_equivalent_loads}
        reference_loads = DLB_equivalent_loads

    if len(sensor_input) != len(m_values):
        raise ValueError("selected_sensors and m_values must have same length")

    if special_name is not None and len(special_name) != len(sensor_input):
        raise ValueError("special_name must match length of selected_sensors")

    rows = []

    for i, (idx, m_val) in enumerate(zip(sensor_input, m_values)):

        original_name = str(reference_loads.sensor_name.values[idx])

        sensor_name = special_name[i] if special_name is not None else original_name

        # Detect load component
        force = detect_force(original_name)

        description = reference_loads.sensor_description.values[idx]
        unit = reference_loads.sensor_unit.values[idx]

        for case_name, case_loads_da in case_loads.items():
            value = case_loads_da.isel(sensor_name=idx).sel(m=m_val).item()

            row = {
                "Sensor": latex_escape(sensor_name if special_name else component + sensor_name),
                "m": m_val,
                "Value": value,
                "Unit": latex_escape(unit),
                "Description": latex_escape(description),
            }
            if special_name:
                row["Force"] = latex_escape(force)
            if case_name is not None:
                row["Case"] = latex_escape(case_name)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Sort so identical sensor names are grouped
    sort_columns = [column for column in ["Sensor", "m", "Case", "Value"] if column in df.columns]
    df = df.sort_values(sort_columns).reset_index(drop=True)

    # ---- Multirow logic ----
    multi = []
    last = None
    counts = df["Sensor"].value_counts().to_dict()

    for sensor in df["Sensor"]:
        if sensor != last:
            multi.append(f"\\multirow{{{counts[sensor]}}}{{*}}{{{sensor}}}")
            last = sensor
        else:
            multi.append("")

    df["Sensor"] = multi
    # ------------------------

    if special_name:
        columns = ["Sensor", "Force"]
    else:
        columns = ["Sensor"]
    if "Case" in df.columns:
        columns.append("Case")
    columns += ["m", "Value", "Unit", "Description"]
    df = df[columns]

    latex = df.to_latex(
        index=False,
        float_format=float_format,
        longtable=False,
        escape=False,
        caption=caption,
        label=label,
        column_format="l" * len(df.columns)
    )
    latex = fit_table_latex_to_page(latex)

    if output_path:
        write_text_file(output_path, latex)
        print(f"LaTeX table saved to: {output_path}")

    return df



# Compatibility Config

metric_list = build_metric_list(mean_upperhalf)
