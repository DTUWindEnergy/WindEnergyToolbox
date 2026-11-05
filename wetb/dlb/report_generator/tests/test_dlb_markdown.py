"""Focused regression tests for Markdown report assembly."""

import unittest
from pathlib import Path

import wetb

wetb.__path__ = [str(Path(__file__).resolve().parents[3])]

from wetb.dlb.report_generator import dlb_markdown as markdown


class MarkdownReportTests(unittest.TestCase):
    def test_converts_generated_latex_cells_to_markdown(self):
        cell = markdown.latex_cell_to_markdown(r"\cellcolor{gray!20}\multirow{2}{*}{Fx\_max}")
        self.assertEqual(cell, "**Fx_max**")

    def test_converts_latex_driver_and_filename_formatting_to_plain_text(self):
        self.assertEqual(markdown.latex_cell_to_markdown(r"\textbf{Fx\_max}"), "**Fx_max**")
        self.assertEqual(
            markdown.latex_cell_to_markdown(
                r"\texttt{\detokenize{res\DLC12\DLC12_wsp23_wdir000_s2002.hdf5}}"),
            r"res\DLC12\DLC12_wsp23_wdir000_s2002.hdf5",
        )

    def test_converts_tabular_block_to_markdown_table(self):
        converted = markdown.latex_tabular_to_markdown(
            "\\begin{tabular}{ll}\n\\toprule\nLoad & Value \\\\\n"
            "Fx\\_max & 1.0 \\\\\n\\bottomrule\n\\end{tabular}"
        )
        self.assertIn("| Load | Value |", converted)
        self.assertIn("| Fx_max | 1.0 |", converted)

    def test_uses_latex_caption_as_table_heading(self):
        source = r"\captionof{table}{Extreme loads for Tower Base}\begin{tabular}{l}Value\end{tabular}"
        self.assertEqual(
            markdown.latex_caption_to_markdown(source, "Tower_Base"),
            "Extreme loads for Tower Base",
        )

    def test_converts_diagonal_cell_highlight_to_markdown_bold(self):
        self.assertEqual(
            markdown.latex_cell_to_markdown(r"\cellcolor{gray!20}2586.645"),
            "**2586.645**",
        )

    def test_extreme_matrix_keeps_all_values_and_bolds_only_diagonal_cells(self):
        converted = markdown.latex_tabular_to_markdown(
            "\\begin{tabular}{lll}\n"
            " & Fx & Fy \\\\\n"
            "\\textbf{Fx\\_max} & \\cellcolor{gray!20}1.0 & 2.0 \\\\\n"
            "\\textbf{Fy\\_max} & 3.0 & \\cellcolor{gray!20}4.0 \\\\\n"
            "\\end{tabular}"
        )
        self.assertIn("| **Fx_max** | **1.0** | 2.0 |", converted)
        self.assertIn("| **Fy_max** | 3.0 | **4.0** |", converted)

    def test_overview_table_block_embeds_table_and_links_source(self):
        source = (
            r"\captionof{table}{Component overview}"
            "\\begin{tabular}{ll}\\toprule\nComponent & Sections \\\\\n"
            "Tower & extreme \\\\\n\\bottomrule\\end{tabular}"
        )
        rendered = markdown.markdown_table_block(
            source,
            Path("report/component_table.tex"),
            Path("report"),
            heading_level="###",
        )
        self.assertIn("### Component overview", rendered)
        self.assertIn("| Component | Sections |", rendered)
        self.assertIn("[component_table table source](<component_table.tex>)", rendered)

    def test_includes_markdown_calculation_explanations(self):
        self.assertIn("D_{j,v,\\theta}", markdown.fatigue_damage_contribution_explanation_markdown())
        self.assertIn("L_{\\max,\\mathrm{DLC}}", markdown.extreme_load_bar_explanation_markdown())


if __name__ == "__main__":
    unittest.main()
