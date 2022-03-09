# -*- coding: utf-8 -*-
"""
Create a master Excel sheet from a series of text files

Command-line usage
------------------
$ python write_master.py [OPTIONS]

Command line options
--------------------
 --folder (default='DLCs')
    folder containing text files
 --filename (default='DLCs.xlsx')
    name of Excel file to save incl. extension
 --fileend (default='.txt')
    text file extention
 --delimiter (default='\t')
    character separating columns in text files

Author: Jenni Rinker, rink@dtu.dk
"""
from   argparse import ArgumentParser
import numpy as np
import os
import pandas as pd


def write_master(path_to_texts,
                 excel_name='DLCs.xlsx', file_end='.txt',
                 delimiter='\t'):
    """ Write a master Excel sheet from a series of text files

    Parameters
    ----------

    path_to_texts : str
        path to directory with text files

    excel_name : str
        filename of generated master Excel file

    file_end : str
        file ending of text files

    delimiter : str
        column delimiter in text files
    """

    # formatting for header cells
    header_dict = {'bold': True, 'font_color': '#1F497D',
                   'bottom': 2, 'bottom_color': '#95B3D7'}

    # get list of text files
    text_files = [f for f in os.listdir(path_to_texts) \
                                  if f.endswith(file_end)]

    # check if main text file in the specified directory
    if 'Main'+file_end not in text_files:
        raise ValueError('\"Main\" file not in CSV directory')

    # rearrange text files so main page is first and everything
    #   else is alphabetical
    text_files.remove('Main'+file_end)
    text_files = ['Main'+file_end] + sorted(text_files)

    # open excel file
    writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')

    # create workbook and add formast
    workbook  = writer.book
    header    = workbook.add_format(header_dict)

    # loop through text files
    for text_name in text_files:

        # define path to csv file
        text_path = os.path.join(path_to_texts,text_name)

        # read data, write to Excel file, and define worksheet handle
        text_df = pd.read_table(text_path,
                                delimiter=delimiter, dtype=str,
                                header=None)
        text_df.to_excel(writer, sheet_name=text_name.rstrip(file_end),
                         index=False, header=False)
        worksheet = writer.sheets[text_name.rstrip(file_end)]

        # get column widths by calculating max string lenths
        col_widths = text_df.apply(lambda x: np.max([len(str(s)) for s in x]))

        # add formatting
        for i_col, width in enumerate(col_widths):
            worksheet.set_column(i_col, i_col, width)
        if 'Main' in text_name:  # hardcode first column on main tab
            worksheet.set_column('A:A',  16.56)
            for i_row in [6,9,13,18,24,28]:
                worksheet.set_row(i_row, cell_format=header)
        else:
            worksheet.set_row(1, cell_format=header)
        worksheet.set_zoom(zoom=85)

    # save worksheet
    writer.save()


if __name__ == '__main__':

    # define argument parser
    parser = ArgumentParser(description="generator of master excel sheet")
    parser.add_argument('--folder', type=str, default='DLCs', action='store',
                        dest='folder', help='Destination folder name')
    parser.add_argument('--filename', type=str, default='DLCs.xlsx',
                        action='store',  dest='filename',
                        help='Master spreadsheet file name')
    parser.add_argument('--fileend', type=str, default='.txt',
                        action='store',  dest='fileend',
                        help='File extension for fileend files')
    parser.add_argument('--delimiter', type=str, default='\t',
                        action='store',  dest='delimiter',
                        help='Text delimiter in files')
    opt = parser.parse_args()

    # write master Excel files
    write_master(opt.folder,
                 excel_name=opt.filename, file_end=opt.fileend,
                 delimiter=opt.delimiter)
