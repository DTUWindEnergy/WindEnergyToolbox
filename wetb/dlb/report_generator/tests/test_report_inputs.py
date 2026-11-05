"""Tests for explicit report inputs and precomputed DLB results."""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import wetb
wetb.__path__ = [str(Path(__file__).resolve().parents[3])]

from wetb.dlb.report_generator import dlb_report
from wetb.dlb.report_generator import dlb_utils
from wetb.dlb.report_generator import dlb_latex
from wetb.dlb.report_generator.dlb_components import Component
from wetb.dlb.report_generator.sections import (
    directional as directional_section,
    equivalent as equivalent_section,
    extreme,
    statistics as statistics_section,
)


TEST_FILES = Path(__file__).parent / 'test_files'


class ReportInputTests(unittest.TestCase):
    @staticmethod
    def components(name, sections, fatigue_m=4, match='Tower base'):
        return (Component(name, sections, fatigue_m, match),)

    def make_report(self, files, sections, match='Tower base'):
        return dlb_report.DLBReport(
            output_dir=Path(__file__).parent, report_title='Test',
            cases={'run': {key: TEST_FILES / path for key, path in files.items()}},
            components=self.components('component', sections, match=match))

    def test_folder_input_uses_report_default_filenames(self):
        report = dlb_report.DLBReport(
            output_dir=Path(__file__).parent,
            report_title='Test',
            cases={'run': TEST_FILES},
            components=self.components('component', ['statistics']))
        self.assertEqual(
            report.case_files['run']['statistics'], TEST_FILES / 'statistics.nc')
        self.assertEqual(
            report.case_files['run']['dlb_extreme'],
            TEST_FILES / 'DLB_extreme_loads.nc')

    def test_statistics_only_does_not_need_other_files_or_weights(self):
        report = self.make_report({'statistics': 'statistics.nc'}, ['statistics'])
        self.assertEqual(
            report.case_files['run']['statistics'], TEST_FILES / 'statistics.nc')
        report.load_cases()
        report.resolve_component_ids()
        self.assertEqual(report.reference_case, 'run')
        self.assertEqual(set(report.case_data[report.reference_case]), {'statistics'})
        self.assertTrue(report.sections['statistics'].is_available())

    def test_cases_are_a_required_report_input(self):
        with self.assertRaises(TypeError):
            dlb_report.DLBReport(
                output_dir=Path(__file__).parent,
                report_title='Test',
                components=self.components('component', ['statistics']),
            )

    def test_report_registers_each_section_as_an_object(self):
        report = self.make_report({'statistics': 'statistics.nc'}, ['statistics'])
        self.assertIsInstance(
            report.sections['statistics'], statistics_section.StatisticsSection)
        self.assertIsInstance(
            report.sections['extreme'], extreme.ExtremeLoadsSection)
        self.assertIsInstance(
            report.sections['equivalent'], equivalent_section.EquivalentLoadsSection)
        self.assertIsInstance(
            report.sections['directional_extreme'], directional_section.DirectionalExtremeSection)
        self.assertIsInstance(
            report.sections['directional_equivalent'],
            directional_section.DirectionalEquivalentSection)

    def test_real_precomputed_arrays_are_used_without_recalculation(self):
        report = self.make_report({
            'dlb_equivalent': 'DLB_equivalent_loads.nc',
            'damage': 'DLB_damage_by_dlc_wsp.nc',
            'dlb_extreme': 'DLB_extreme_loads.nc',
        }, ['equivalent', 'extreme'])
        report.load_cases()
        report.resolve_component_ids()
        reference_data = report.case_data[report.reference_case]
        self.assertEqual(reference_data['damage'].sizes['dlc'], 1)
        self.assertEqual(reference_data['dlb_extreme'].sizes['driver'], 14)
        self.assertTrue(report.sections['equivalent'].is_available())

    def test_global_filter_does_not_modify_combined_dlb_arrays(self):
        report = self.make_report({'dlb_extreme': 'DLB_extreme_loads.nc'}, ['extreme'])
        report.dlcs = ['12']
        report.load_cases()
        self.assertEqual(
            report.case_data[report.reference_case]['dlb_extreme'].sizes['sensor_name'], 7)

    def test_comparison_cases_require_the_same_datasets(self):
        report = dlb_report.DLBReport(
            output_dir=Path(__file__).parent, report_title='Test', cases={
                'stats': {'statistics': TEST_FILES / 'statistics.nc'},
                'extremes': {'dlb_extreme': TEST_FILES / 'DLB_extreme_loads.nc'}},
            components=self.components('tower', ['statistics', 'extreme']))
        with self.assertRaisesRegex(ValueError, 'must provide the same datasets'):
            report.load_cases()

    def test_comparison_cases_can_differ_in_unrequested_datasets(self):
        report = dlb_report.DLBReport(
            output_dir=Path(__file__).parent, report_title='Test', cases={
                'first': {
                    'statistics': TEST_FILES / 'statistics.nc',
                    'dlb_extreme': TEST_FILES / 'DLB_extreme_loads.nc',
                },
                'second': {'statistics': TEST_FILES / 'statistics.nc'},
            },
            components=self.components('tower', ['statistics']))
        report.load_cases()
        self.assertEqual(report.reference_case, 'first')

    def test_comparison_cases_require_matching_sensor_order(self):
        first = xr.DataArray(
            np.ones((2,)), dims=('sensor_name',),
            coords={'sensor_name': ['Tower base Fx', 'Tower base Fy']})
        second = first.isel(sensor_name=[1, 0])
        report = dlb_report.DLBReport(
            output_dir=Path(__file__).parent, report_title='Test', cases={
                'first': {'statistics': 'first.nc'},
                'second': {'statistics': 'second.nc'}},
            components=self.components('tower', ['statistics']))
        with patch.object(dlb_report, 'load_data', side_effect=[
                {'statistics': first}, {'statistics': second}]):
            with self.assertRaisesRegex(ValueError, 'different sensor_name ordering'):
                report.load_cases()

    def test_precomputed_directional_results_need_no_weights(self):
        directional_extreme = xr.DataArray(
            np.ones((1, 1, 2)), dims=('dlc', 'sensor_name', 'angle'),
            coords={'dlc': ['12'], 'sensor_name': ['Tower base force'],
                    'sensor_unit': ('sensor_name', ['kN']), 'angle': [-30, 0]})
        directional_equivalent = xr.DataArray(
            np.ones((1, 2, 1)), dims=('sensor_name', 'angle', 'm'),
            coords={'sensor_name': ['Tower base force'],
                    'sensor_unit': ('sensor_name', ['kN']), 'angle': [-30, 0], 'm': [4]})
        report = dlb_report.DLBReport(
            output_dir=Path(__file__).parent, report_title='Test',
            cases={'run': {'dlb_directional_extreme': 'unused.nc',
                           'dlb_directional_equivalent': 'unused.nc'}},
            components=self.components(
                'tower', ['directional_extreme', 'directional_equivalent']))
        with patch.object(dlb_report, 'load_data', return_value={
                'dlb_directional_extreme': directional_extreme,
                'dlb_directional_equivalent': directional_equivalent}):
            report.load_cases()
        report.resolve_component_ids()
        with patch.object(report.sections['directional_extreme'], 'generate') as extreme_section, \
             patch.object(report.sections['directional_equivalent'], 'generate') as equivalent_section:
            report.generate_sections()
        self.assertEqual(extreme_section.call_count, 1)
        self.assertEqual(equivalent_section.call_count, 1)

    def test_fatigue_summary_uses_first_available_dlc_for_wind_speed_plot(self):
        damage = xr.DataArray(
            np.ones((1, 2, 1, 1)), dims=('dlc', 'wsp', 'sensor_name', 'm'),
            coords={'dlc': ['13'], 'wsp': [3., 5.], 'sensor_name': ['Tower base'],
                    'sensor_unit': ('sensor_name', ['kN']), 'm': [4]})
        with patch.object(equivalent_section, 'finish_figure') as finish:
            equivalent_section.plot_fatigue_summary_damage(
                damage, sensor_input=[0], m_values=[4], fatigue_dlcs=['13'], show=False)
        self.assertIn('DLC13 damage by wind speed', finish.call_args.args[0].axes[1].get_title())

    def test_subplot_grid_adapts_to_number_of_dlc_plots(self):
        figure, axes = dlb_utils.make_subplot_grid(n_panels=2)
        try:
            self.assertEqual(len(axes), 2)
            self.assertEqual(tuple(figure.get_size_inches()), (14.0, 5.0))
        finally:
            plt.close(figure)

    def test_short_term_equivalent_plot_uses_compact_grid_for_two_dlcs(self):
        equivalent = xr.DataArray(
            [[ [1.0] ], [ [2.0] ]], dims=('filename', 'sensor_name', 'm'),
            coords={'filename': ['DLC12_wsp03_s1001', 'DLC13_wsp05_s1001'],
                    'sensor_name': ['tower'], 'm': [4],
                    'dlc': ('filename', ['12', '13']),
                    'wsp': ('filename', [3.0, 5.0]),
                    'sensor_unit': ('sensor_name', ['kN'])})
        with patch.object(equivalent_section, 'make_subplot_grid', wraps=equivalent_section.make_subplot_grid) as grid:
            equivalent_section.plot_dlb_equivalent(
                equivalent, sensor_input=[0], m_value=[4], component='Tower', show=False)
        self.assertEqual(grid.call_args.kwargs['n_panels'], 2)

    def test_directional_envelope_keeps_title_after_squeezing_sensor_dimension(self):
        directional_extreme = xr.DataArray(
            np.ones((1, 1, 2)), dims=('dlc', 'sensor_name', 'angle'),
            coords={'dlc': ['12'], 'sensor_name': ['Tower base force'],
                    'angle': [-30, 0]})
        with patch.object(directional_section, 'finish_figure') as finish:
            directional_section.plot_load_envl(directional_extreme, sensor_input=[0], show=False)
        self.assertEqual(finish.call_args.args[0]._suptitle.get_text(), 'Tower base force')

    def test_table_labels_do_not_contain_latex_special_characters(self):
        latex = dlb_latex.fit_table_latex_to_page(
            r'\begin{table}\label{tab:extreme_loads_Tower_Base}\end{table}')
        self.assertIn(r'\label{tab:extreme-loads-Tower-Base}', latex)
        self.assertEqual(dlb_latex.latex_label('tab:partial_safety_factors'),
                         'tab:partial-safety-factors')

    def test_extreme_table_escapes_string_dtype_row_labels(self):
        loads = xr.DataArray(
            np.ones((1, 1, 1)), dims=('sensor_name', 'driver', 'load'),
            coords={'sensor_name': ['tower'], 'driver': ['Fx_max'], 'load': ['Fx']})
        with patch.object(extreme, 'write_text_file') as write:
            extreme.ExtremeLoadsSection.dlc_extreme_loads_to_latex(
                loads, sensor_input=[0], output_path='unused.tex')
        self.assertIn(r'\textbf{Fx\_max}', write.call_args.args[1])

    def test_diagonal_extreme_table_displays_filename_literally(self):
        loads = xr.DataArray(
            [[[1.0]]], dims=('filename', 'driver', 'load'),
            coords={'filename': [r'res\DLC12\DLC12_wsp23_wdir000_s2002.hdf5'],
                    'driver': ['Fx_max'], 'load': ['Fx'], 'dlc': ('filename', ['12']),
                    'sensor_name': 'tower'})
        with patch.object(extreme, 'write_text_file') as write:
            extreme.ExtremeLoadsSection.diagonal_extreme_loads_to_latex(
                loads.expand_dims(sensor_name=['tower']), sensor_input=0, output_path='unused.tex')
        self.assertIn(
            r'\texttt{\detokenize{res\DLC12\DLC12_wsp23_wdir000_s2002.hdf5}}',
            write.call_args.args[1],
        )

    def test_statistics_table_displays_filename_literally(self):
        statistics = xr.DataArray(
            [[[1.0, -1.0]]], dims=('filename', 'sensor_name', 'statistic'),
            coords={'filename': [r'res\DLC12\DLC12_wsp23_wdir000_s2002.hdf5'],
                    'sensor_name': ['tower'], 'statistic': ['max', 'min'],
                    'dlc': ('filename', ['12'])})
        with patch.object(statistics_section, 'write_text_file') as write:
            statistics_section.StatisticsSection.dlc_statistics_to_latex(
                statistics, sensor_input=[0], output_path='unused.tex')
        latex = write.call_args.args[1]
        self.assertIn(
            r'\texttt{\detokenize{res\DLC12\DLC12_wsp23_wdir000_s2002.hdf5}}', latex)
        self.assertNotIn(r'\textbackslash', latex)

    def test_report_renders_precomputed_fatigue_without_weights(self):
        files = {'equivalent': 'equivalent_loads.nc',
                 'dlb_equivalent': 'DLB_equivalent_loads.nc',
                 'damage': 'DLB_damage_by_dlc_wsp.nc'}
        report = self.make_report(files, ['equivalent'], 'shaft_rot angle speed')
        report.load_cases()
        report.resolve_component_ids()
        component = report.components['component']
        with patch.object(equivalent_section.EquivalentLoadsSection, 'dlb_equivalent_loads_to_latex') as table, \
             patch.object(equivalent_section, 'plot_dlb_equivalent_bar'), \
             patch.object(equivalent_section, 'plot_dlb_equivalent'), \
             patch.object(equivalent_section, 'plot_fatigue_summary_damage') as damage_plot, \
             patch.object(equivalent_section.ReportSection, 'clear_matching_files'), patch.object(Path, 'mkdir'):
            report.sections['equivalent'].generate('component', component)
        reference_data = report.case_data[report.reference_case]
        self.assertIs(table.call_args.args[0], reference_data['dlb_equivalent'])
        self.assertIs(damage_plot.call_args.args[0], reference_data['damage'])

    def test_extreme_table_uses_precomputed_matrix_with_no_raw_file(self):
        report = self.make_report({'dlb_extreme': 'DLB_extreme_loads.nc'}, ['extreme'])
        report.load_cases()
        report.resolve_component_ids()
        with patch.object(extreme.ExtremeLoadsSection, 'dlc_extreme_loads_to_latex') as table, \
             patch.object(extreme.ReportSection, 'clear_matching_files'):
            report.sections['extreme'].generate('component', report.components['component'])
        self.assertIs(
            table.call_args.args[0],
            report.case_data[report.reference_case]['dlb_extreme'],
        )

    def test_extreme_caption_does_not_repeat_matching_sensor_name(self):
        self.assertEqual(
            extreme.component_sensor_caption('Blade 1 root', 'Blade 1 root'),
            'Blade 1 root',
        )
        self.assertEqual(
            extreme.component_sensor_caption('Tower Base', 'Tower Base Fx'),
            'Tower Base Fx',
        )

    def test_precomputed_extremes_supply_safety_factor_table(self):
        precomputed_extreme = xr.DataArray(
            np.ones((1, 1, 1)), dims=('sensor_name', 'driver', 'load'),
            coords={'sensor_name': ['Tower base'], 'driver': ['Fx_max'], 'load': ['Fx']},
        )
        psf = xr.DataArray([1.35, 1.1], dims='dlc',
                           coords={'dlc': ['12', '22b']}, name='psf')
        report = self.make_report({'dlb_extreme': 'DLB_extreme_loads.nc'}, ['extreme'])
        output_dir = MagicMock()
        output_dir.__truediv__.return_value.exists.return_value = False
        with patch.object(dlb_report, 'make_component_table'), \
             patch.object(dlb_report, 'make_sensor_table'), \
             patch.object(dlb_report, 'make_psf_table') as psf_table:
            dlb_report.make_front_matter(
                {}, {'dlb_extreme': precomputed_extreme,
                     'dlb_extreme_psf': psf}, output_dir)
        self.assertEqual(psf_table.call_args.args[0], ['12', '22b'])
        self.assertEqual(psf_table.call_args.args[1], {'DLC12': 1.35, 'DLC22b': 1.1})

    def test_psf_table_wraps_dlc_factor_pairs_with_separators(self):
        factors = {f'DLC{dlc}': 1.35 for dlc in range(1, 8)}
        with patch.object(dlb_latex.os, 'makedirs'), \
             patch.object(dlb_latex, 'write_text_file') as write:
            dlb_latex.make_psf_table(
                list(range(1, 8)), factors, output_dir='unused')
        latex = write.call_args.args[1]
        self.assertEqual(latex.count('DLC &'), 2)
        self.assertIn(r'\midrule' + '\n' + r'\addlinespace[0.35em]', latex)
        self.assertNotIn(r'\resizebox{\textwidth}{!}{%', latex)


if __name__ == '__main__':
    unittest.main()
