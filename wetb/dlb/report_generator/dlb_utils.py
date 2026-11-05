import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def write_text_file(output_path, text, encoding="utf-8"):
    """Write ``text`` to ``output_path``, creating missing parent directories.

    Returns the normalized ``Path`` used for the write. This helper has filesystem
    side effects and overwrites an existing file at the requested path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding=encoding)
    return output_path


def latex_escape(value):
    """Return ``value`` as text with LaTeX-special characters escaped.

    ``None`` becomes an empty string. Use this for plain text placed in LaTeX;
    it intentionally does not preserve existing LaTeX markup.
    """
    if value is None:
        return ""

    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }

    return re.sub(
        r"[\\&%$#_{}~^]",
        lambda match: replacements[match.group(0)],
        text,
    )


def latex_filename(value):
    """Return a filename for display as literal LaTeX text.

    ``\\detokenize`` keeps path separators and underscores visible, rather
    than rendering a Windows path separator as ``\\textbackslash{}``.  The
    typewriter font also makes source-file names distinguishable in tables.
    """
    if value is None:
        return ""
    return rf"\texttt{{\detokenize{{{value}}}}}"


def get_component_ids(component_match, nc_file):
    """Resolve the sensor IDs selected by one component in one dataset.

    Name and description criteria are matched independently: a sensor selected
    by either criterion is included. Matching ignores case and spaces and uses
    substring matching. Description matches are returned before name matches,
    preserving the previous function's selection order.
    """
    if "match" in component_match or "tags" in component_match:
        match = component_match.get("match") or component_match.get("tags", {})
    else:
        match = component_match

    def normalized_terms(value):
        """Return normalized text values, or ``None`` for an omitted criterion."""
        if value is None:
            return None
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            values = value
        else:
            values = [value]
        return {
            str(item).strip().lower().replace(" ", "")
            for item in values
        }

    selected_ids = []
    # check match tag exists in either or both desc and name

    description_terms = normalized_terms(match.get("description"))
    if description_terms is not None and "sensor_description" in nc_file.coords:
        for index, description in enumerate(nc_file["sensor_description"].values):
            normalized_description = str(description).strip().lower().replace(" ", "")
            if any(term in normalized_description for term in description_terms):
                selected_ids.append(index)


    name_terms = normalized_terms(match.get("name"))
    if name_terms is not None:
        for index, name in enumerate(nc_file["sensor_name"].values):
            normalized_name = str(name).strip().lower().replace(" ", "")
            if any(term in normalized_name for term in name_terms):
                selected_ids.append(index)

    return list(dict.fromkeys(selected_ids))


def latex_path(path, relative_to=None):
    """Return a POSIX path suitable for LaTeX ``\\input`` and graphics commands.

    When ``relative_to`` is supplied, ``path`` must be beneath it and is made
    relative to that directory before conversion.
    """
    path = Path(path)
    if relative_to is not None:
        path = path.relative_to(Path(relative_to))
    return path.as_posix()


def chunked(items, chunk_size):
    """Gives consecutive groups containing at most ``chunk_size`` items.

    The report uses this to split many plot tasks into figure pages with a
    limited number of subplots. For example, five tasks with ``chunk_size=2``
    yield pages containing two, two, and one task. The final group may therefore
    be shorter. ``items`` must support ``len`` and slicing.
    """
    for idx in range(0, len(items), chunk_size):
        yield items[idx:idx + chunk_size]


def ensure_output_dir(save_dir):
    """Create ``save_dir`` and return it as a ``Path``, or return ``None`` unchanged.
    """
    if save_dir is None:
        return None
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def normalize_sensor_ids(sensor_input, sort_sensors=True):
    """sorts and converts given ids into ordered array for plotting.
    """
    sensors = list(sensor_input)
    return sorted(sensors) if sort_sensors else sensors


def normalize_dlc_filter(dlcs):
    """Return selected DLC labels as strings, or ``None`` for all DLCs."""
    if dlcs in (None, "all"):
        return None
    if isinstance(dlcs, (str, int, float)):
        return [str(dlcs)]
    return [str(dlc) for dlc in dlcs]


def normalize_m_values(m_value, sensors, name="m_value"):
    """Return one fatigue exponent for each selected sensor.

    A scalar ``m_value`` is broadcast to every sensor; an array-like value must
    have exactly one entry per sensor. Raises ``ValueError`` with ``name`` in the
    message when lengths differ.
    """
    values = np.asarray(m_value)
    if values.ndim == 0:
        values = np.full(len(sensors), values.item())
    if len(sensors) != len(values):
        raise ValueError(f"sensor_input and {name} must have same length")
    return values


def make_subplot_grid(polar=False, n_panels=6):
    """Create a compact report subplot grid and return its flat axes array.

    The layout grows from one panel through a maximum of six panels. This keeps
    short final pages (for example, two remaining DLCs) from reserving the
    empty height of the standard 3-by-2 grid. Set ``polar`` for polar axes.
    """
    if not 1 <= n_panels <= 6:
        raise ValueError("n_panels must be between one and six")

    if n_panels == 1:
        nrows, ncols = 1, 1
    elif n_panels == 2:
        nrows, ncols = 1, 2
    elif n_panels <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 3, 2

    subplot_kw = {"polar": True} if polar else None
    fig, axes = _get_plt().subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 5 * nrows),
        subplot_kw=subplot_kw,
        squeeze=False,
    )
    return fig, axes.ravel()


def finish_subplot_grid(fig, axes, used_count, save_path=None, show=True):
    """Finalize a subplot page after ``used_count`` panels were populated."""
    for ax in axes[used_count:]:
        ax.axis("off")

    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show, close_all=True)


def finish_figure(fig, save_path=None, show=True, close_all=False):
    """Save, display or close ``fig`` according to the requested output mode."""
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt = _get_plt()
    if show:
        plt.show()
    else:
        plt.close(fig)

    if close_all:
        plt.close("all")


def _get_plt():
    """Import Matplotlib only for helpers that create or finalize figures."""
    import matplotlib.pyplot as plt
    return plt


