"""Markdown assembly for generated DLB report assets."""

import re
from pathlib import Path

from wetb.dlb.report_generator.dlb_utils import latex_path


SECTION_TITLES = {
    "statistics": "Statistics",
    "extreme": "Extreme Loads",
    "equivalent": "Equivalent Loads",
    "directional_extreme": "Directional Extreme Loads",
    "directional_equivalent": "Directional Equivalent Loads",
}


def markdown_title_from_stem(stem):
    """Convert a generated asset filename into a readable Markdown heading."""
    return re.sub(r"\s+", " ", str(stem).replace("_", " ")).strip()


def latex_cell_to_markdown(cell):
    """Convert the formatting used in generated table cells to plain Markdown."""
    is_highlighted = bool(re.search(r"\\cellcolor\{[^}]*\}", cell))
    cell = re.sub(r"\\cellcolor\{[^}]*\}", "", cell)
    cell = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{(.*)\}", r"\1", cell)
    # LaTeX tables use bold driver labels and literal typewriter filenames.
    # Markdown needs their contents, not the LaTeX formatting commands.
    cell = re.sub(r"\\textbf\{(.*)\}", r"**\1**", cell)
    cell = re.sub(r"\\texttt\{\\detokenize\{(.*)\}\}", r"\1", cell)
    replacements = {
        r"\_": "_",
        r"\&": "&",
        r"\%": "%",
        r"\$": "$",
        r"\#": "#",
        r"\{": "{",
        r"\}": "}",
    }
    for latex, plain in replacements.items():
        cell = cell.replace(latex, plain)
    cell = cell.strip().replace("|", "\\|")
    if is_highlighted and not (cell.startswith("**") and cell.endswith("**")):
        return f"**{cell}**"
    return cell


def latex_caption_to_markdown(source, fallback):
    """Extract a generated table caption, falling back to its filename."""
    match = re.search(
        r"\\(?:caption|captionof\{table\})(?:\[[^]]*\])?\{(.*?)\}",
        source,
        re.DOTALL,
    )
    return latex_cell_to_markdown(match.group(1)) if match else markdown_title_from_stem(fallback)


def fatigue_damage_contribution_explanation_markdown():
    """Return the fatigue-damage calculation explanation in Markdown form."""
    return "\n\n".join(
        [
            "#### Fatigue damage contribution calculation",
            (
                "The fatigue damage contribution figures show how equivalent-load damage is "
                "distributed across the selected fatigue DLCs."
            ),
            "$$D_{j,v,\\theta} = L_{\\mathrm{eq},j,v,\\theta}^{m} N_{\\mathrm{eq},j} "
            "P_{j,v,\\theta} \\frac{T_{\\mathrm{life}}}{N_{\\mathrm{seeds},j,v,\\theta} "
            "T_{\\mathrm{sim},j}}$$",
            (
                "Here, $D_{j,v,\\theta}$ is the damage contribution for DLC $j$, wind speed $v$, "
                "and wind direction $\\theta$; $L_{\\mathrm{eq},j,v,\\theta}$ is the short-term "
                "equivalent load; $m$ is the Wöhler exponent; $N_{\\mathrm{eq},j}$ is the "
                "equivalent cycle count; $P_{j,v,\\theta}$ is the bin probability; "
                "$T_{\\mathrm{life}}$ is the design lifetime; $N_{\\mathrm{seeds},j,v,\\theta}$ "
                "is the number of seeds; and $T_{\\mathrm{sim},j}$ is the simulation time."
            ),
            "$$D_j = \\sum_v \\sum_\\theta D_{j,v,\\theta}, \\qquad "
            "D_{\\mathrm{total}} = \\sum_{j \\in J} D_j, \\qquad "
            "p_j = 100 \\frac{D_j}{D_{\\mathrm{total}}}$$",
            (
                "The pie chart shows $p_j$, the percentage contribution from each selected DLC. "
                "The wind-speed bar chart uses the same damage calculation, summed by wind speed."
            ),
        ]
    )


def extreme_load_bar_explanation_markdown():
    """Return the extreme-load bar-plot calculation explanation in Markdown form."""
    return "\n\n".join(
        [
            "#### Extreme-load bar plot calculation",
            (
                "The extreme-load bar plots show the governing design load per DLC. For seeded DLCs, "
                "simulations are grouped by operating condition, such as wind speed and direction, "
                "then averaged across seeds."
            ),
            "$$L_{\\max,\\mathrm{DLC}} = \\max_{g \\in G_{\\mathrm{DLC}}} "
            "\\left(\\frac{1}{N_s}\\sum_{s=1}^{N_s}L_{\\max,g,s}\\right), \\qquad "
            "L_{\\min,\\mathrm{DLC}} = \\min_{g \\in G_{\\mathrm{DLC}}} "
            "\\left(\\frac{1}{N_s}\\sum_{s=1}^{N_s}L_{\\min,g,s}\\right)$$",
            (
                "Here, $G_{\\mathrm{DLC}}$ is the set of operating-condition groups, $s$ is the "
                "seed index, and $N_s$ is the number of seeds in group $g$. DLC 1.4, 1.5, 2.3, "
                "3.1, 3.2, 3.3, 4.1, and 4.2 bypass seed averaging and aggregate directly over files:"
            ),
            "$$L_{\\max,\\mathrm{DLC}} = \\max_{f \\in F_{\\mathrm{DLC}}}L_{\\max,f}, \\qquad "
            "L_{\\min,\\mathrm{DLC}} = \\min_{f \\in F_{\\mathrm{DLC}}}L_{\\min,f}$$",
        ]
    )


def latex_tabular_to_markdown(source):
    """Convert LaTeX source containing simple tabular blocks into Markdown tables."""
    tabular_bodies = re.findall(
        r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", source, re.DOTALL
    )
    markdown_tables = []
    for body in tabular_bodies:
        body = re.sub(r"\\(?:toprule|midrule|bottomrule)\s*", "", body)
        body = re.sub(r"\\addlinespace(?:\[[^\]]*\])?\s*", "", body)
        rows = []
        for raw_row in body.split(r"\\"):
            row = raw_row.strip()
            if not row:
                continue
            cells = [latex_cell_to_markdown(cell) for cell in row.split("&")]
            rows.append(cells)
        if not rows:
            continue

        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        header, *body_rows = rows
        body_rows = [
            [f"**{cell}**" if cell else "" for cell in row]
            if row[0] == "DLC" else row
            for row in body_rows
        ]
        markdown_tables.append("\n".join(
            [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * width) + " |",
                *["| " + " | ".join(row) + " |" for row in body_rows],
            ]
        ))
    return "\n\n".join(markdown_tables)


def markdown_table_block(table_source, table_path, output_dir, heading_level="####"):
    """Render a generated LaTeX table and retain a link to its source."""
    table_path = Path(table_path)
    title = latex_caption_to_markdown(table_source, table_path.stem)
    parts = [f"{heading_level} {title}"]
    converted_table = latex_tabular_to_markdown(table_source)
    if converted_table:
        parts.append(converted_table)
    parts.append(
        f"- [{table_path.stem} table source](<"
        f"{latex_path(table_path, output_dir).replace(' ', '%20')}>)"
    )
    return "\n\n".join(parts)


def build_markdown_section(component_dir, section_key, output_dir):
    """Build Markdown for one report section from its generated assets."""
    section_title = SECTION_TITLES[section_key]
    table_folders = {
        "statistics": component_dir / "latex_tables" / "stats",
        "extreme": component_dir / "latex_tables" / "extreme",
        "equivalent": component_dir / "latex_tables" / "eq_load",
    }
    figure_folders = {
        "statistics": component_dir / "figures" / "stats",
        "extreme": component_dir / "figures" / "extreme",
        "equivalent": component_dir / "figures" / "equivalent",
        "directional_extreme": component_dir / "figures" / "directional_extreme",
        "directional_equivalent": component_dir / "figures" / "directional_equivalent",
    }

    table_folder = table_folders.get(section_key)
    figure_folder = figure_folders[section_key]
    table_files = sorted(table_folder.rglob("*.tex")) if table_folder and table_folder.exists() else []
    figure_files = sorted(figure_folder.rglob("*.png")) if figure_folder.exists() else []
    if not table_files and not figure_files:
        return ""

    parts = [f"## {section_title}"]
    if table_files:
        tables = []
        for table_path in table_files:
            table_source = table_path.read_text(encoding="utf-8")
            tables.append(markdown_table_block(table_source, table_path, output_dir))
        parts.append("### Tables\n\n" + "\n\n".join(tables))
    if figure_files:
        figures = []
        if section_key == "extreme":
            figures.append(extreme_load_bar_explanation_markdown())
        if section_key == "equivalent" and any(
            "fatigue_damage_summary" in figure_path.name for figure_path in figure_files
        ):
            figures.append(fatigue_damage_contribution_explanation_markdown())
        for figure_path in figure_files:
            caption = markdown_title_from_stem(figure_path.stem)
            figures.extend([
                f"#### {caption}",
                f"![{caption}](<{latex_path(figure_path, output_dir).replace(' ', '%20')}>)",
            ])
        parts.append("### Figures\n\n" + "\n\n".join(figures))
    return "\n\n".join(parts)


def write_markdown_report(output_dir, components, filename="report.md", title="Component Load Report"):
    """Write a Markdown report embedding generated tables and figures."""
    output_dir = Path(output_dir)
    lines = [f"# {str(title).replace('\n', ' ').replace('#', '\\#')}", "", "## Overview", ""]

    overview_tables = [
        output_dir / "component_table.tex",
        output_dir / "psf_table.tex",
        output_dir / "sensor_table.tex",
    ]
    existing_overview_tables = [path for path in overview_tables if path.exists()]
    if existing_overview_tables:
        # lines.extend(["", "## Overview"])
        for path in existing_overview_tables:
            lines.extend([
                "",
                markdown_table_block(
                    path.read_text(encoding="utf-8"), path, output_dir, "###"
                ),
            ])

    for component_name, component in components.items():
        component_dir = output_dir / re.sub(r"\s+", "_", str(component_name).strip())
        sections = [
            build_markdown_section(component_dir, section_key, output_dir)
            for section_key in component.enabled_sections
        ]
        sections = [section for section in sections if section]
        if not sections:
            continue
        component_label = str(component.label).replace("\n", " ").replace("#", "\\#")
        lines.extend(["", f"# {component_label}"])
        lines.extend(["", "\n\n".join(sections)])

    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
