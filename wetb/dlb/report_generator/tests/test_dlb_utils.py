"""Focused regression tests for LaTeX report utility functions."""

import unittest
from pathlib import Path

import numpy as np
import wetb
import xarray as xr

wetb.__path__ = [str(Path(__file__).resolve().parents[3])]

from wetb.dlb.report_generator import dlb_utils as utils


class LatexUtilsTests(unittest.TestCase):
    def test_latex_escape_handles_backslash_without_reescaping_replacement(self):
        self.assertEqual(
            utils.latex_escape(r"A\B_1 & 2"),
            r"A\textbackslash{}B\_1 \& 2",
        )

    def test_latex_filename_preserves_path_separators_and_underscores(self):
        self.assertEqual(
            utils.latex_filename(r"res\DLC12\DLC12_wsp23_wdir000_s2002.hdf5"),
            r"\texttt{\detokenize{res\DLC12\DLC12_wsp23_wdir000_s2002.hdf5}}",
        )

    def test_latex_path_can_be_relative_to_document_directory(self):
        self.assertEqual(
            utils.latex_path(
                Path("report") / "Component" / "figure.png",
                relative_to=Path("report"),
            ),
            "Component/figure.png",
        )

    def test_component_ids_keep_name_or_description_matches_and_order(self):
        dataset = xr.DataArray(
            np.zeros(3),
            dims="sensor_name",
            coords={
                "sensor_name": ["Tower Base Fx", "Blade Root Fy", "Tower Base Mx"],
                "sensor_description": (
                    "sensor_name",
                    ["tower force", "blade bending", "tower moment"],
                ),
            },
        )

        sensor_ids = utils.get_component_ids(
            {"name": " Tower Base ", "description": "blade bending"},
            dataset,
        )

        # Description matches precede name matches, and a name or description
        # match is sufficient to include a sensor.
        self.assertEqual(sensor_ids, [1, 0, 2])

if __name__ == "__main__":
    unittest.main()
