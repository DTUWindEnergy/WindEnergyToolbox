from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from dlb_report import generate_report

# Main user input block.
REPORT_CONFIG = {
    "output_dir": Path("tower_base_report_v11"),
    # Use one case for a normal report, or two cases for comparison plots.
    # Sensor IDs are assumed to be identical between the two cases.
    "cases": {
        "v1": Path("postproc"),
    },
    "main_tex_filename": "main.tex",
    "report_title": "Component Load Report",
}

# no seeds "14","15","23","31","32","33",

ANALYSIS_CONFIG = {
    # Use "all" for all DLCs, or set a list like ["12", "22p", "32"].
    "dlcs": "all",
    # Optional partial safety factor override. Use None for default values.
    "safety_factor_list": None,
    "fatigue_dlc_list": ["12", "24", "31", "41", "64"],
    "weight_params": {
        "n_years": 25,
        "Vin": 3,
        "Vout": 25,
        "Vr": 10,
        "Vref": 50,
        "Vstep": 2,
    },
}


# Component-specific input block.
# Add one entry per component you want in the report.
# Use sections="all" or choose any combination of:
# "statistics", "extreme", "equivalent", "directional_extreme", "directional_equivalent".
COMPONENTS = {
    "Tower Base": {
        "sections": "all",
        "m": 4,
        "match": "Tower Base",
    },
    "Blade 1 root": {
        "sections": ["extreme"],
        "m": 10,
        "match": "Blade 1 root",
    }
}


def main():
    generate_report(REPORT_CONFIG, ANALYSIS_CONFIG, COMPONENTS)


if __name__ == "__main__":
    main()
