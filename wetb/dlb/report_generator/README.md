# DLB LaTeX Report

This folder turns DLB post-processing NetCDF files into a component-load report.
It can produce a compilable LaTeX report, a Markdown report, or both.

Create the report configuration in your own Python script or notebook, then
call `DLBReport.generate()` in an environment where the `wetb` package is
available.

```text
postproc NetCDF files + configuration
                |
                v
          DLBReport.generate()
                |
     +----------+-----------+
     |                      |
     v                      v
figures and LaTeX tables   report assembly
     |                      |
     +----------+-----------+
                |
                v
       main.tex and/or report.md
```

## User interface

Import the configuration and report classes, define the components and input
case, then generate the requested document formats:

```python
from pathlib import Path

from wetb.dlb.report_generator.dlb_components import Component
from wetb.dlb.report_generator.dlb_report import DLBReport

components = (
    Component(
        "Tower Base",
        sections="all",
        fatigue_m=4,
        match="Tower Base",
    ),
)

DLBReport(
    output_dir=Path("report"),
    report_title="Component Load Report",
    components=components,
    cases={"baseline": Path("postproc")},
    output_formats=("latex", "markdown"),
).generate()
```

### Accepted input types

Pass report settings directly as keyword arguments to `DLBReport`.
`components` is an iterable of `Component` objects.

All paths may be written as `str` values or `pathlib.Path` values. The report
converts them to `Path` objects before using them.

#### Report settings

| Key | Accepted type | Required | Notes |
| --- | --- | --- | --- |
| `output_dir` | `str` or `Path` | Yes | Destination directory for all generated files. |
| `cases` | `dict[str, str | Path | dict[str, str | Path]]` | Yes | One or two named cases. A case is either a folder using the default filenames or a mapping from dataset keys to arbitrary file paths. |
| `output_formats` | `str` or iterable of `str` | No | Allowed values are `"latex"` and `"markdown"`; defaults to `["latex"]`. Case is ignored. |
| `main_tex_filename` | `str` | No | LaTeX document filename; defaults to `"main.tex"`. |
| `markdown_filename` | `str` | No | Markdown document filename; defaults to `"report.md"`. |
| `report_title` | `str` | Yes | Document title. |
| `dlcs` | `None`, `"all"`, `str`, `int`, `float`, or iterable of those values | No | Filters datasets with a DLC coordinate; cannot change already aggregated DLB results. |
| `fatigue_dlc_list` | iterable of DLC values | No | DLC values used for fatigue contribution plots. |

#### Component settings

Create each report component with `Component`:

| Key | Accepted type | Required | Notes |
| --- | --- | --- | --- |
| `label` | `str` | No | Display name; defaults to the component name. |
| `sections` | `"all"`, a section-name `str`, or an iterable of section-name strings | Yes | Duplicate names are removed while retaining the requested order. |
| `fatigue_m` | numeric scalar or array-like of numeric values | Yes | Fatigue/Wohler exponent. A scalar applies to every matched sensor; an array-like value needs one entry per selected sensor. |
| `match` | `str` or `dict[str, object]` | Yes | Sensor selection rule using names and/or descriptions. |

`match` accepts either a single string or a dictionary containing `name` and/or
`description` keys. Their values may be a string or an iterable of strings.
Each supplied value is matched as a case-insensitive substring after spaces are
removed. For example:

```python
match="Tower Base"
match={"name": ["Tower", "TB"], "description": "Base"}
```

### `DLBReport` settings

Controls input cases and report files.

| Setting | Meaning |
| --- | --- |
| `output_dir` | Folder that receives figures, tables, and reports. |
| `cases` | One or two named mappings of dataset keys to file paths. Two cases enable comparison plots. |
| `output_formats` | A list containing `"latex"`, `"markdown"`, or both. |
| `main_tex_filename` | Name of the generated LaTeX entry document. Defaults to `main.tex`. |
| `markdown_filename` | Name of the generated Markdown document. Defaults to `report.md`. |
| `report_title` | Title used in the generated documents. |
| `dlcs` | DLC filter for datasets retaining a DLC coordinate. |
| `fatigue_dlc_list` | DLCs shown in fatigue contribution plots. |

At most two cases are supported. When two are used, datasets needed by the
requested sections must have compatible sensor metadata and component matches.

### `COMPONENTS`

Each `Component` object creates one report component. Its `name` is the
component identifier and is converted to underscores for generated folder
names.

```python
Component(
    "Tower Base",
    label="Tower base",            # Optional display label
    sections=("extreme",),         # Or "all"
    fatigue_m=4,
    match="Tower Base",
)
```

Supported section names are:

- `statistics`
- `extreme`
- `equivalent`
- `directional_extreme`
- `directional_equivalent`

`match` may instead be a dictionary when name and description should be matched
independently:

```python
match={"name": "Tower", "description": "Base"}
```

Sensor matching is case-insensitive, ignores spaces, and uses substring
matching.

## Input data

Specify each available file separately. Omitted files are optional; the report
produces the outputs supported by the supplied data. For example:

```python
"cases": {"run": {
    "statistics": Path("inputs/statistics.nc"),
    "dlb_extreme": Path("inputs/DLB_extreme_loads.nc"),
    "dlb_equivalent": Path("inputs/DLB_equivalent_loads.nc"),
    "damage": Path("inputs/DLB_damage_by_dlc_wsp.nc"),
}}
```

The supported dataset keys and their uses are:

| Section | Required file(s) |
| --- | --- |
| `statistics` | Per-simulation statistics plots and tables. |
| `extreme` | Per-simulation extreme plots and source tables. |
| `dlb_extreme` | Precomputed DLB extreme tables. |
| `equivalent` | Per-simulation equivalent-load plots. |
| `dlb_equivalent` | Precomputed DLB equivalent-load tables and summary bars. |
| `damage` | Precomputed fatigue contributions by DLC and wind speed. |
| `dlb_directional_extreme` | Precomputed directional extreme plots, retaining one curve per DLC. |
| `dlb_directional_equivalent` | Precomputed lifetime directional equivalent-load plots. |

For backwards compatibility, a case may still specify a folder. Precomputed
DLB aggregates cannot be recalculated by a report-side DLC filter; make sure
they were generated for the intended selection. Precomputed DLB extreme files store the factors as a DLC-indexed
`psf` coordinate, so
the report generates the safety-factor table without a report-side factor
configuration.

## Generation flow

`DLBReport.generate()` runs the following stages in order:

1. **Resolve sections** — collects all sections requested by all components.
2. **Load cases** — loads the NetCDF data for each configured case and applies
   the optional DLC filter.
3. **Resolve sensors** — matches each component against the reference case and
   records matching sensor indexes for every dataset type.
4. **Create front matter** — writes component, partial-safety-factor, and sensor
   overview tables.
5. **Prepare DLB data** — loads precomputed DLB results for all report sections.
6. **Generate component sections** — writes tables and figures for each enabled
   section. With two cases, comparison plots are generated where supported.
7. **Write report documents** — assembles LaTeX and/or Markdown according to
   `output_formats`.

The intermediate generated assets are intentional: the LaTeX and Markdown
documents both reference them rather than recomputing the analysis.

## Module map

| Module | Responsibility |
| --- | --- |
| `dlb_components.py` | Component configuration container and validation. |
| `dlb_report.py` | Main orchestration: loading data, resolving sensors, calculations, section generation, and document output. |
| `dlb_utils.py` | Shared utilities, including LaTeX formatting, data normalization, and Matplotlib figure helpers. |
| `dlb_latex.py` | LaTeX table creation and assembly of the main LaTeX document and component files. |
| `dlb_markdown.py` | Markdown report assembly, including conversion of simple generated LaTeX tables. |
| `test_dlb_*.py` | Focused regression tests for configuration, Markdown conversion, and shared utilities. |

## Output layout

For a component named `Tower Base`, output typically has this shape:

```text
output_dir/
├── main.tex
├── report.md
├── component_table.tex
├── psf_table.tex
├── sensor_table.tex
└── Tower_Base/
    ├── Tower_Base.tex
    ├── latex_tables/
    │   ├── stats/
    │   ├── extreme/
    │   └── eq_load/
    └── figures/
        ├── stats/
        ├── extreme/
        ├── equivalent/
        ├── directional_extreme/
        └── directional_equivalent/
```

`main.tex` includes the overview tables and each generated component `.tex`
file. Compile it with a LaTeX installation to create a PDF. `report.md` embeds
the overview and section tables as Markdown, retains links to their generated
`.tex` sources, and embeds the PNG figures.

## Practical workflow

1. Specify the paths of the available post-processing `.nc` files in `cases`.
2. Set `output_dir` to a new or disposable report folder.
3. In your own script or notebook, configure the desired components, sections,
   and fatigue exponents, then call `DLBReport(...).generate()`.
4. Run that script or notebook.
5. Review warnings for missing datasets or unmatched sensors.
6. Compile `main.tex` if LaTeX output was requested, or open `report.md` in a
   Markdown viewer.

Generated files from an earlier run may be deleted selectively before new plots
are written, so do not keep manual work inside the generated figures or tables
folders.
