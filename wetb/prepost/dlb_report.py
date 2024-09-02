# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:03:27 2024

@author: nicgo
"""

import numpy as np
import os
import glob
import openpyxl
from openpyxl.styles import Alignment
from wetb.hawc2.Hawc2io import ReadHawc2
from wetb.gtsdf import load_statistic

def get_filenames(path, ext):
    roots = [path] + [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return sorted([filename for root in roots for filename in glob.glob(os.path.join(root, ext))])

def get_statistic(time, data, statistics=['min', 'mean', 'max', 'std', 'eq3', 'eq4', 'eq6', 'eq8', 'eq10', 'eq12'], no_bins=46):
    def get_stat(stat):
        if hasattr(np, stat):
            return getattr(np, stat)(data, 0)
        elif (stat.startswith("eq") and stat[2:].isdigit()):
            from wetb.fatigue_tools.fatigue import eq_load
            m = float(stat[2:])
            return [eq_load(sensor, no_bins, m, time[-1] - time[0] + time[1] - time[0])[0][0] for sensor in data.T]
    return np.array([get_stat(stat) for stat in statistics]).T

def get_dlb_statistics(path, ext,
                       sensor_lst=["Ae rot. torque", "Ae rot. power", "Ae rot. thrust", "bea1 angle_speed"],
                       statistics=['min', 'mean', 'max', 'std', 'eq3', 'eq4', 'eq6', 'eq8', 'eq10', 'eq12']):
    filenames = get_filenames(path, ext)
    dlb_statistics = {}
    if ext == "*.hdf5":
        for f in filenames:
            file_statistics = load_statistic(f, xarray=True)
            dlb_statistics[os.path.basename(os.path.splitext(f)[0])] = {}
            for s in sensor_lst:
                dlb_statistics[os.path.basename(os.path.splitext(f)[0])][s] = {}
                for stat in file_statistics['stat'].values:
                    dlb_statistics[os.path.basename(os.path.splitext(f)[0])][s][stat] = file_statistics.sel(sensor_name = s, stat = stat).values[()]
    elif ext == "*.sel":
        for f in filenames:
            res_file = ReadHawc2(f)
            time = res_file.t
            data = res_file.ReadAll()
            info = res_file.ChInfo[0]
            file_statistics = get_statistic(time, data, statistics)
            dlb_statistics[os.path.basename(os.path.splitext(f)[0])] = {}
            for s in sensor_lst:
                dlb_statistics[os.path.basename(os.path.splitext(f)[0])][s] = {}
                for st in statistics:
                    dlb_statistics[os.path.basename(os.path.splitext(f)[0])][s][st] = file_statistics[info.index(s), statistics.index(st)]
    return dlb_statistics
    
def write_dlb_report_to_excel(dlb_statistics, excel_filename):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "DLB Statistics"
    central_headers = ['Simulation']
    inner_headers = ['']
    first_item = next(iter(dlb_statistics.values()))
    if isinstance(first_item, dict):
        for central_key, inner_dict in first_item.items():
            if isinstance(inner_dict, dict):
                for inner_key in inner_dict.keys():
                    central_headers.append(central_key)
                    inner_headers.append(inner_key)
            else:
                central_headers.append(central_key)
                inner_headers.append('')
    for col_num, header in enumerate(central_headers, start=1):
        sheet.cell(row=1, column=col_num, value=header)
    for col_num, header in enumerate(inner_headers, start=1):
        sheet.cell(row=2, column=col_num, value=header)
    current_central = central_headers[0]
    start_col = 1
    for col_num, central_header in enumerate(central_headers[1:], start=2):
        if central_header != current_central:
            if col_num - start_col > 1:
                sheet.merge_cells(start_row=1, start_column=start_col, end_row=1, end_column=col_num-1)
            start_col = col_num
            current_central = central_header
    if len(central_headers) - start_col > 0:
        sheet.merge_cells(start_row=1, start_column=start_col, end_row=1, end_column=len(central_headers))
    for col_num in range(1, len(central_headers) + 1):
        sheet.cell(row=1, column=col_num).alignment = Alignment(horizontal='center', vertical='center')
    for row_num, (outer_key, central_dict) in enumerate(dlb_statistics.items(), start=3):
        sheet.cell(row=row_num, column=1, value=outer_key)
        col_num = 2
        for central_key, inner_dict in central_dict.items():
            if isinstance(inner_dict, dict):
                for inner_key, inner_value in inner_dict.items():
                    sheet.cell(row=row_num, column=col_num, value=inner_value)
                    col_num += 1
            else:
                sheet.cell(row=row_num, column=col_num, value=inner_dict)
                col_num += 1
    workbook.save(excel_filename)

if __name__ == '__main__':        
    folder_path = r"C:\Users\nicgo\Documents\Turbine_Models\WindEnergyToolbox\res_hawc_binary"
    dlb_statistics = get_dlb_statistics(folder_path, "*.sel")
    excel_filename = 'DLB_Statistics_Report_binary.xlsx'
    write_dlb_report_to_excel(dlb_statistics, excel_filename)
