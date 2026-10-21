import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_parent_dir(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_text_file(output_path, text, encoding="utf-8"):
    output_path = ensure_parent_dir(output_path)
    output_path.write_text(text, encoding=encoding)
    return output_path


def latex_escape(value):
    """Escape plain text for safe inclusion in LaTeX tables."""
    if value is None:
        return ""

    text = str(value)
    replacements = {
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

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def latex_format_value(value, float_format="%.3f"):
    """Format a scalar for insertion into LaTeX table cells."""
    if pd.isna(value):
        return ""
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float_format % value
    return latex_escape(value)


def component_path_name(component_name):
    """Return a filesystem-safe component name for generated paths."""
    return re.sub(r"\s+", "_", str(component_name).strip())


def get_sensor_id_from_nc(
    nc_file,
    name=None,
    desc=None,
):
    def concate_str(v):
        return str(v).strip().lower().replace(" ", "")

    def normalize(v):
        if v is None:
            return {"Not provided"}
        if isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
            return {concate_str(x) for x in v}
        return {concate_str(v)}

    name_inp = normalize(name)
    desc_inp = normalize(desc)
    matched = []

    for i in range(len(nc_file["sensor_name"])):
        sensor_name = concate_str(nc_file["sensor_name"].values[i])
        sensor_desc = (
            concate_str(nc_file["sensor_description"].values[i])
            if "sensor_description" in nc_file.coords else []
        )

        if name_inp != {"Not provided"} and not any(item in sensor_name for item in name_inp):
            continue
        if desc_inp != {"Not provided"} and not any(item in sensor_desc for item in desc_inp):
            continue

        matched.append(i)

    return matched


def get_component_ids(component_match, nc_file):
    """
    Resolve sensor IDs for one component against one dataset.

    Accepts either a plain match dict or a component dict containing
    "match" or legacy "tags".
    """
    if "match" in component_match or "tags" in component_match:
        match = component_match.get("match") or component_match.get("tags", {})
    else:
        match = component_match

    selected_ids = []
    desc_tags = match.get("description")
    name_tags = match.get("name")

    if desc_tags is not None:
        selected_ids += get_sensor_id_from_nc(nc_file, desc=desc_tags)
    if name_tags is not None:
        selected_ids += get_sensor_id_from_nc(nc_file, name=name_tags)

    return list(dict.fromkeys(selected_ids))


def latex_path(path):
    """Convert a filesystem path to a LaTeX-friendly relative path."""
    return Path(path).as_posix()


def chunked(items, chunk_size):
    """Yield list chunks of a fixed maximum size."""
    for idx in range(0, len(items), chunk_size):
        yield items[idx:idx + chunk_size]


def safe_filename(value):
    """Return a compact filename-safe string."""
    return str(value).replace(" ", "_").replace(":", "").replace("/", "_")


def ensure_output_dir(save_dir):
    """Create and return a save directory Path, or None when disabled."""
    if save_dir is None:
        return None
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def normalize_sensor_ids(sensor_input, sort_sensors=True):
    """Return sensor IDs as a list with optional sorting."""
    sensors = list(sensor_input)
    return sorted(sensors) if sort_sensors else sensors


def normalize_m_values(m_value, sensors, name="m_value"):
    """Return one fatigue exponent per selected sensor."""
    values = np.asarray(m_value)
    if values.ndim == 0:
        values = np.full(len(sensors), values.item())
    if len(sensors) != len(values):
        raise ValueError(f"sensor_input and {name} must have same length")
    return values


def figure_caption_from_path(path):
    """Build a readable LaTeX figure caption from a generated image path."""
    caption = Path(path).stem.replace("_to_", " to ")
    caption = caption.replace("-", " ").replace("_", " ")
    return latex_escape(caption)


def figure_title_from_folder(path):
    """Build a readable LaTeX heading from a generated figure folder."""
    return latex_escape(Path(path).name.replace("-", " ").replace("_", " "))


def dlc_label(value):
    """Return a DLC label without duplicating the DLC prefix."""
    text = str(value)
    return text if text.upper().startswith("DLC") else f"DLC{text}"


DLC_DEFAULTS = [
    "DLC12", "DLC13", "DLC14", "DLC15", "DLC21",
    "DLC22b", "DLC22p", "DLC22y", "DLC23", "DLC24",
    "DLC31", "DLC32", "DLC33", "DLC41", "DLC42",
    "DLC51", "DLC61", "DLC62", "DLC63", "DLC64",
]
MEAN_METRIC_DLCS = {"DLC12", "DLC13", "DLC24", "DLC64"}
MAX_METRIC_DLCS = {
    "DLC14", "DLC15", "DLC23", "DLC31",
    "DLC32", "DLC33", "DLC41", "DLC42",
}
LOW_SAFETY_FACTOR_DLCS = {"DLC22b", "DLC22p", "DLC22y", "DLC23", "DLC62", "DLC71"}

regex_list = {dlc: r"DLC(\w+?)_wsp(\d{2})" for dlc in DLC_DEFAULTS}
safety_factor_list = {
    dlc: 1.1 if dlc in LOW_SAFETY_FACTOR_DLCS else 1.35
    for dlc in [*DLC_DEFAULTS, "DLC71", "DLC81"]
}


def build_metric_list(mean_upperhalf):
    """Return default DLC metric functions using the supplied upper-half reducer."""
    return {
        dlc: np.mean if dlc in MEAN_METRIC_DLCS
        else np.max if dlc in MAX_METRIC_DLCS
        else mean_upperhalf
        for dlc in DLC_DEFAULTS
    }
