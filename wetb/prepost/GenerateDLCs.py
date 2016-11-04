# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:11:06 2015

@author: tlbl
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

# arctan and pi are required because they are in the formulas that are
# evaluated
from numpy import floor, arctan, pi
import pandas as pd
import xlrd
from argparse import ArgumentParser
import os


def multi_for(iterables):
    """
    Routine to create list with combination of elements.
    """
    if not iterables:
        yield ()
    else:
        for item in iterables[0]:
            for rest_tuple in multi_for(iterables[1:]):
                yield (item,) + rest_tuple


class GeneralDLC(object):
    """
    Basic class to generate the DLC spreadsheets. It contains routines to
    handle the different types of tags.

    * Constants: are fixed in the current DLC, e.g. reference turbulence\
        intensity, rotor radius, reference wind speed, ....

    * Variables: define the number of cases in a DLC through their combination\
        e.g. wind speed, number of turbulence seeds, yaw angle, ....


    * Functions: depend on other tags e.g turbulence intensity, file name, ....

    """

    def __init__(self):

        pass

    def remove_from_dict(self, non_defaults, defaults):

        for key in non_defaults.keys():
            try:
                del defaults[key]
            except:
                pass
        return defaults

    def add_variables_tag(self, dlc, variables, variables_order):

        cases_len = []
        for tag in variables_order:
            dlc[tag] = []
            v = variables[tag]
            for i in range(len(v)-1):
                try:
                    v.remove('')
                except:
                    pass
            if tag == '[seed]':
                cases_len.append(int(v[0]))
            else:
                cases_len.append(len(v))
        cases_index = multi_for(list(map(range, cases_len)))

        for irow, row in enumerate(cases_index):
            counter = floor(irow/len(variables['[wsp]']))+1
            for icol, col in enumerate(row):
                if variables_order[icol] == '[seed]':
                    value = '%4.4i' % (1000*counter + row[variables_order.index('[wsp]')]+1)
                else:
                    value = variables[variables_order[icol]][col]
                    if not isinstance(value, float) and not isinstance(value, int):
                        value = str(value)
                dlc[variables_order[icol]].append(value)

    def add_constants_tag(self, dlc, constants):

        for key in constants.keys():
            dlc[key] = [constants[key]]*len(dlc['[wsp]'])

    def sort_formulas(self, formulas):
        # sort formulas based on their dependency

        keys_list = sorted(formulas)
        for i in range(len(keys_list)):
            for ikey, key in enumerate(keys_list):
                formula = formulas[key]
                for ikey2, key2 in enumerate(keys_list):
                    if key2 in formula:
                        if ikey < ikey2:
                            keys_list.pop(ikey)
                            keys_list.insert(ikey2, key)
                            break
        return keys_list

    def eval_formulas(self, dlc):

        for key in dlc.keys():
            if isinstance(dlc[key][0], str):
                if "[" in dlc[key][0]:
                    for key2 in dlc.keys():
                        for iformula, formula in enumerate(dlc[key]):
                            if key2 in formula:
                                dlc[key][iformula] = dlc[key][iformula].replace(key2, '%s'%dlc[key2][iformula])
                    for iformula, formula in enumerate(dlc[key]):
                        formula = formula.replace(',', '.')
                        formula = formula.replace(';', ',')
                        dlc[key][iformula] = eval(formula)

    def add_formulas(self, dlc, formulas):

        keys_list = self.sort_formulas(formulas)

        for fkey in keys_list:
            flist = []
            for i in range(len(dlc['[wsp]'])):
                formula = formulas[fkey]
                for key in dlc.keys():
                    if key in formula:
                        if formula[0] == '"':
                            if key == '[wsp]' or key == '[gridgustdelay]':
                                fmt = '%2.2i'
                                formula = formula.replace(key, fmt%int(dlc[key][i]))
                            elif key == '[wdir]' or key == '[G_phi0]':
                                fmt = '%3.3i'
                                formula = formula.replace(key, fmt%int(dlc[key][i]))
                            elif key == '[sign]':
                                fmt = '%s'
                                formula = formula.replace(key, fmt%dlc[key][i])
                            else:
                                fmt = '%4.4i'
                                formula = formula.replace(key, fmt % int(dlc[key][i]))
                        elif key in formula:
                            formula = formula.replace(key, '%s' % dlc[key][i])
                formula = formula.replace(',', '.')
                formula = formula.replace(';', ',')
                flist.append(eval(formula))

            dlc[fkey] = flist


class GenerateDLCCases(GeneralDLC):
    """
    Class to generate Excell sheets for each DLB case starting from a single
    Excell sheet.

    Parameters
    ----------

    filename: str
        Name of the excel spreadsheet containing the definition of all the
        cases to generate.

    folder: str
        Name of the folder in which to save the DLB cases.

    Example
    -------
        DLB = GenerateDLCCases()
        DLB.execute()


    """

    def execute(self, filename='DLCs.xlsx', folder=''):

        book = xlrd.open_workbook(filename)

        nsheets = book.nsheets

        # Loop through all the sheets. Each sheet correspond to a DLC.
        for isheet in range(1, nsheets):

            # Read all the initialization constants and functions in the
            # first sheet
            general_constants = {}
            general_functions = {}
            sheet = book.sheets()[0]
            for i in range(1, sheet.ncols):
                if sheet.cell_value(9, i) != '':
                    general_constants[str(sheet.cell_value(9, i))] = \
                        sheet.cell_value(10, i)
                if sheet.cell_value(13, i) != '':
                    general_functions[str(sheet.cell_value(13, i))] = \
                        sheet.cell_value(14, i)

            sheet = book.sheets()[isheet]

            print('Sheet #%i' % isheet, sheet.name)

            # Read the actual sheet.
            constants = {}
            variables = {}
            formulas = {}
            variables_order = []
            # Loop through the columns
            for i in range(sheet.ncols):
                if sheet.cell_value(1, i) is not None:
                    tag = str(sheet.cell_value(1, i))
                    if tag is not '':
                        if sheet.cell_value(0, i) == 'C':
                            constants[tag] = sheet.cell_value(2, i)
                        if sheet.cell_value(0, i) == 'V':
                            variables_order.append(tag)
                            variables[tag] = \
                                [sheet.cell_value(j, i) for j in range(2, sheet.nrows)]
                        if sheet.cell_value(0, i) == 'F':
                            formulas[tag] = str(sheet.cell_value(2, i))

            dlc = {}

            general_constants = self.remove_from_dict(variables,
                                                      general_constants)
            general_constants = self.remove_from_dict(constants,
                                                      general_constants)
            general_functions = self.remove_from_dict(formulas,
                                                      general_functions)

            self.add_variables_tag(dlc, variables, variables_order)
            self.add_constants_tag(dlc, general_constants)
            self.add_constants_tag(dlc, constants)
            self.add_formulas(dlc, formulas)
            self.add_formulas(dlc, general_functions)
            self.eval_formulas(dlc)
            df = pd.DataFrame(dlc)
            if not os.path.exists(folder):
                os.makedirs(folder)
            df.to_excel(os.path.join(folder, sheet.name+'.xlsx'), index=False)


class RunTest():
    """
    Class to perform basic testing of the GenerateDLCCases class. It writes the
    spreadsheets and compare them with a reference set.
    """
    def execute(self):

        from pandas.util.testing import assert_frame_equal
        a = GenerateDLCCases()
        a.execute()

        book = xlrd.open_workbook('DLCs.xlsx')
        nsheets = book.nsheets
        for isheet in range(1, nsheets):
            sheet = book.sheets()[isheet]
            print('Sheet #%i' % isheet, sheet.name)
            book1 = pd.read_excel('Reference/'+sheet.name+'.xlsx')

            book2 = pd.read_excel(sheet.name+'.xls')

            book2 = book2[book1.columns]
            assert_frame_equal(book1, book2, check_dtype=False)

if __name__ == '__main__':

    parser = ArgumentParser(description = "generator of DLB spreadsheets")
    parser.add_argument('--master', type=str, default='DLCs.xlsx', action='store',
                        dest='filename', help='Master spreadsheet file')
    parser.add_argument('--folder', type=str, default='', action='store',
                        dest='folder', help='Destination folder name')
    opt = parser.parse_args()
    DLB = GenerateDLCCases()
    DLB.execute(filename=opt.filename, folder=opt.folder)

