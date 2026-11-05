###
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

from wetb.dlb.report_generator.dlb_utils import (
    latex_escape,
    latex_path,
    write_text_file,
)

# LaTeX Formatting And Includes

def latex_label(value):
    """Return a LaTeX-safe identifier suitable for ``\\label``."""
    return re.sub(r"[^A-Za-z0-9:.-]+", "-", str(value)).strip("-")


def fit_table_latex_to_page(latex):
    """Center a LaTeX table inline and scale it to page width safely."""
    def safe_label(match):
        """Replace one LaTeX label with its identifier-safe equivalent."""
        return rf"\label{{{latex_label(match.group(1))}}}"

    # Labels are identifiers, not visible text. In particular, underscores are
    # invalid in text mode and make LaTeX report ``Missing $ inserted``.
    latex = re.sub(r"\\label\{([^}]*)\}", safe_label, latex)
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

def include_tex_files(tex_folder, relative_to=None):
    """Include generated .tex files from a folder."""
    tex_folder = Path(tex_folder)
    if not tex_folder.exists():
        return ""

    parts = []
    for tex_file in sorted(tex_folder.rglob("*.tex")):
        parts.append(f"\\input{{{latex_path(tex_file, relative_to)}}}")

    return "\n".join(parts)

def include_png_file(png_file, relative_to=None):
    """Return LaTeX for one generated PNG figure."""
    caption = latex_escape(
        Path(png_file).stem.replace("_to_", " to ").replace("-", " ").replace("_", " ")
    )
    return "\n".join(
        [
            "\\Needspace{0.8\\textheight}",
            "\\begin{center}",
            f"    \\includegraphics[width=\\textwidth,height=0.76\\textheight,keepaspectratio]{{{latex_path(png_file, relative_to)}}}",
            f"    \\captionof{{figure}}{{{caption}}}",
            "\\end{center}",
        ]
    )

def include_png_files(plot_folder, relative_to=None):
    """Include generated .png files as one figure per grouped image."""
    plot_folder = Path(plot_folder)
    if not plot_folder.exists():
        return ""

    return "\n\n".join(
        include_png_file(png_file, relative_to)
        for png_file in sorted(plot_folder.rglob("*.png"))
    )

def include_png_files_by_folder(plot_folder, relative_to=None):
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
        parts.append("\n\n".join(include_png_file(png_file, relative_to) for png_file in root_pngs))

    for folder in child_folders:
        parts.append("\\Needspace{0.8\\textheight}")
        parts.append(
            f"\\subsubsection{{{latex_escape(folder.name.replace('-', ' ').replace('_', ' '))}}}"
        )
        parts.append(include_png_files(folder, relative_to))

    return "\n\n".join(parts)

def include_equivalent_png_files(plot_folder, relative_to=None):
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

    parts = [include_png_file(png_file, relative_to) for png_file in del_summary_pngs]

    if fatigue_summary_pngs:
        parts.append("\\Needspace{0.8\\textheight}")
        parts.append("\\subsubsection{Fatigue Damage Contribution}")
        parts.append(fatigue_damage_contribution_explanation_latex())
        parts.extend(include_png_file(png_file, relative_to) for png_file in fatigue_summary_pngs)

    if short_term_folders:
        parts.append("\\Needspace{0.8\\textheight}")
        parts.append("\\subsubsection{Short Term Equivalent Load Plots}")
        for folder in short_term_folders:
            parts.append(include_png_files(folder, relative_to))

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

def build_report_subsection(
    title,
    table_folder,
    figure_folder,
    relative_to=None,
    figure_renderer=include_png_files,
    figure_intro="",
):
    """Build one subsection from the generated tables and figures on disk."""
    tables = include_tex_files(table_folder, relative_to) if table_folder is not None else ""
    figures = figure_renderer(figure_folder, relative_to) if figure_folder is not None else ""

    if not tables and not figures:
        return ""

    parts = ["\\Needspace{0.8\\textheight}", f"\\subsection{{{title}}}"]
    if tables:
        parts.append(tables)
    if figures and figure_intro:
        parts.append(figure_intro)
    if figures:
        parts.append(figures)

    return "\n\n".join(parts)

def build_component_section(
    component_name, component_label, component_dir, enabled_section_keys, sections,
    relative_to=None,
):
    """Build one component section with report subsections."""
    subsections = []
    for section_key in enabled_section_keys:
        subsection = sections[section_key].build_latex_subsection(
            component_dir, relative_to=relative_to
        )
        if subsection:
            subsections.append(subsection)

    if not subsections:
        return ""

    section_title = latex_escape(component_label or component_name)
    return "\n\n".join([f"\\section{{{section_title}}}"] + subsections)

def write_component_latex_report(
    output_dir,
    component_name,
    component,
    sections,
    filename=None,
):
    """Write the LaTeX body for one component and return its path."""
    output_dir = Path(output_dir)
    component_path = re.sub(r"\s+", "_", str(component_name).strip())
    component_dir = output_dir / component_path
    component_label = component.label
    enabled_section_keys = component.enabled_sections
    component_tex = build_component_section(
        component_name,
        component_label,
        component_dir,
        enabled_section_keys,
        sections,
        relative_to=output_dir,
    )

    if not component_tex:
        return None

    output_path = component_dir / (filename or f"{component_path}.tex")
    write_text_file(output_path, component_tex)
    return output_path

def write_main_latex_report(
    output_dir,
    components,
    sections,
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
            sections,
        )
        if component_tex_path is not None:
            component_inputs.append(
                f"\\input{{{latex_path(component_tex_path, output_dir)}}}"
            )

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
            "\\title{" + latex_escape(title) + "}",
            "\\date{}",
            "\\begin{document}",
            "\\maketitle",
            "\\tableofcontents",
            "\\clearpage",
            "\\section{Overview}",
            "\\input{component_table.tex}"
            if (output_dir / "component_table.tex").exists()
            else "",
            "\\input{psf_table.tex}"
            if (output_dir / "psf_table.tex").exists()
            else "",
            "\\input{sensor_table.tex}"
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
    """Return the load type identified by a sensor name, when recognized."""
    name_low = name.lower()
    for p in ["fx","fy","fz","mx","my","mz","dll","cl","cd","alfa","bea","torque","power","thrust"]:
        if p in name_low:
            return p.capitalize()
    return "-"

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
    """Create a LaTeX table describing selected report sensors.

    Parameters
    ----------
    da : xarray.DataArray
        Dataset containing sensor metadata.
    selected_sensors : iterable of int
        Sensor indexes to include in the table.

    Returns
    -------
    pandas.DataFrame
        Values represented in the generated table.
    """
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
        print(f"Table saved to: {file_path}")

    return df

def make_component_table(
    standard_components,
    output_dir=None,
    filename="component_table.tex",
    caption="Component overview",
    label="tab:component_overview",
):
    """Create a LaTeX table summarizing configured report components.

    Parameters
    ----------
    standard_components : dict
        Components keyed by their report names.
    output_dir : path-like, optional
        Directory to which the table is written.
    filename : str, optional
        Name of the generated LaTeX file.

    Returns
    -------
    pandas.DataFrame
        Values represented in the generated table.
    """
    rows = []

    for component, rules in standard_components.items():
        if not isinstance(rules, dict):
            rules = vars(rules)
        enabled_sections = rules.get("enabled_sections", [])
        if isinstance(enabled_sections, str):
            enabled_sections = [enabled_sections]

        rows.append({
            "Component": latex_escape(component),
            "Label": latex_escape(rules.get("label", component)),
            "Enabled sections": latex_escape(", ".join(map(str, enabled_sections))),
            "W\\\"ohler component (m)": latex_escape(rules.get("fatigue_m", "")),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No components found.")
        if output_dir is not None:
            write_text_file(
                Path(output_dir) / filename,
                "\\begin{center}No components found.\\end{center}\n",
            )
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
        print(f"Table saved to: {file_path}")

    return df

def make_psf_table(
    dlcs,
    safety_factors,
    output_dir=None,
    filename="psf_table.tex",
    caption="Partial safety factors",
    label="tab:partial_safety_factors",
    dlcs_per_block=5,
):
    """Create a compact, wrapped partial-safety-factor table.

    Each block consists of a ``DLC`` row and a ``Partial safety factor`` row.
    Blocks are separated by a rule so a large DLC set remains readable without
    widening the table beyond its natural width.
    """
    if dlcs_per_block < 1:
        raise ValueError("dlcs_per_block must be at least one")

    rows = []

    unique_dlcs = list(dict.fromkeys(dlcs))
    if not unique_dlcs:
        df = pd.DataFrame()
        print("No partial safety factors found.")
        if output_dir is not None:
            write_text_file(
                Path(output_dir) / filename,
                "\\begin{center}No partial safety factors found.\\end{center}\n",
            )
        return df

    for dlc in unique_dlcs:
        dlc_key = str(dlc) if str(dlc).upper().startswith("DLC") else f"DLC{dlc}"
        rows.append((latex_escape(dlc_key), safety_factors.get(dlc_key, "")))

    row_groups = [
        rows[start:start + dlcs_per_block]
        for start in range(0, len(rows), dlcs_per_block)
    ]
    max_group_len = min(len(rows), dlcs_per_block)
    table_rows = []
    for group in row_groups:
        padded_group = group + [("", "")] * (max_group_len - len(group))
        table_rows.append(["DLC"] + [dlc for dlc, _ in padded_group])
        table_rows.append(["Partial safety factor"] + [factor for _, factor in padded_group])

    df = pd.DataFrame(table_rows)
    column_format = "l" + "c" * max_group_len
    table_body_lines = []
    for block_number in range(len(row_groups)):
        if block_number:
            table_body_lines.extend(["\\midrule", "\\addlinespace[0.35em]"])
        for row in table_rows[2 * block_number:2 * block_number + 2]:
            formatted_values = []
            for value in row:
                if pd.isna(value):
                    formatted_value = ""
                elif isinstance(value, (float, int, np.floating, np.integer)):
                    formatted_value = "%.3f" % value
                else:
                    formatted_value = latex_escape(value)
                formatted_values.append(formatted_value)
            table_body_lines.append(" & ".join(formatted_values) + r" \\")
    table_body = "\n".join(table_body_lines)
    label = latex_label(label)
    latex = "\n".join(
        [
            "\\begin{table}",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{column_format}}}",
            "\\toprule",
            table_body,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        write_text_file(file_path, latex)
        print(f"Table saved to: {file_path}")

    return df
