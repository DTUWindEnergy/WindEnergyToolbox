'''
Created on 01/10/2014

@author: MMPE
'''
import xlrd
import pandas as pd
import numpy as np
import glob
import os
import functools

from wetb.hawc2.sel_file import SelFile
from wetb.functions.caching import cache_function
HOURS_PR_20YEAR = 20 * 365 * 24

def Weibull(u, k, start, stop, step):
    C = 2 * u / np.sqrt(np.pi)
    cdf = lambda x :-np.exp(-(x / C) ** k)
    return {wsp:-cdf(wsp - step / 2) + cdf(wsp + step / 2) for wsp in np.arange(start, stop + step, step)}


class DLCHighLevel(object):
    def __init__(self, filename):
        self.filename = filename
        wb = xlrd.open_workbook(self.filename)

        # Variables
        sheet = wb.sheet_by_name("Variables")
        for row_index in range(1, sheet.nrows):
            name = str(sheet.cell(row_index, 0).value).lower()
            value = sheet.cell(row_index, 1).value
            setattr(self, name, value)
        if not hasattr(self, "res_path"):
            raise Warning("The 'Variables' sheet of '%s' must contain the variable 'res_path' specifying the path to the result folder" % self.filename)
        self.res_path = os.path.join(os.path.dirname(self.filename), self.res_path)

        #DLC sheet
        sheet = wb.sheet_by_name("DLC")
        self.dlc_df = pd.DataFrame({sheet.cell(0, col_index).value.lower(): [sheet.cell(row_index, col_index).value for row_index in range(2, sheet.nrows) if sheet.cell(row_index, 0).value != ""] for col_index in range(sheet.ncols)})
        for k in ['name', 'load', 'wsp', 'wdir', 'dlc_dist', 'wsp_dist', 'wdir_dist']:
            assert k.lower() in self.dlc_df.keys(), "DLC sheet must have a '%s' column" % k
        self.dlc_df['name'] = [n.lower().replace("dlc", "") for n in self.dlc_df['name']]
        if 'psf' not in self.dlc_df:
            self.dlc_df['psf'] = 1

        # Sensors sheet
        sheet = wb.sheet_by_name("Sensors")
        name_col_index = [sheet.cell(0, col_index).value.lower() for col_index in range(0, sheet.ncols)].index("name")
        self.sensor_df = pd.DataFrame({sheet.cell(0, col_index).value.lower(): [sheet.cell(row_index, col_index).value for row_index in range(1, sheet.nrows) if sheet.cell(row_index, name_col_index).value != ""] for col_index in range(sheet.ncols)})
        for k in ['Name', 'Nr']:
            assert k.lower() in self.sensor_df.keys(), "Sensor sheet must have a '%s' column" % k
        assert not any(self.sensor_df['name'].duplicated()), "Duplicate sensor names: %s" % ",".join(self.sensor_df['name'][self.sensor_df['name'].duplicated()].values)
        for k in ['description', 'unit', 'statistic', 'ultimate', 'fatigue', 'm', 'neql', 'extremeload', 'bearingdamage', 'mindistance', 'maxdistance']:
            if k not in self.sensor_df.keys():
                self.sensor_df[k] = ""
        for _, row in self.sensor_df[self.sensor_df['fatigue'] != ""].iterrows():
            assert isinstance(row['m'], (int, float)), "Invalid m-value for %s (m='%s')" % (row['name'], row['m'])
            assert isinstance(row['neql'], (int, float)), "Invalid NeqL-value for %s (NeqL='%s')" % (row['name'], row['neql'])
        for name, nrs in zip(self.sensor_info("extremeload").name, self.sensor_info("extremeload").nr):
            assert (np.atleast_1d((eval(str(nrs)))).shape[0] == 6), "'Nr' for Extremeload-sensor '%s' must contain 6 sensors (Fx,Fy,Fz,Mx,My,Mz)" % name


    def __str__(self):
        return self.filename


    def sensor_info(self, sensors=[]):
        if sensors != []:
            return self.sensor_df[functools.reduce(np.logical_or, [((self.sensor_df.get(f, np.array([""] * len(self.sensor_df.name))).values != "") | (self.sensor_df.name == f)) for f in np.atleast_1d(sensors)])]
        else:
            return self.sensor_df

    def dlc_variables(self, dlc):
        dlc_row = self.dlc_df[self.dlc_df['name'] == dlc]
        def get_lst(x):
            if isinstance(x, pd.Series):
                x = x.iloc[0]
            if ":" in str(x):
                start, step, stop = [float(eval(v, globals(), self.__dict__)) for v in x.lower().split(":")]
                return list(np.arange(start, stop + step, step))
            else:
                return [float(eval(v, globals(), self.__dict__)) for v in str(x).lower().replace("/", ",").split(",")]
        wsp = get_lst(dlc_row['wsp'])
        wdir = get_lst(dlc_row['wdir'])
        return wsp, wdir

    def fatigue_distribution(self):
        fatigue_dist = {}
        data = self.dlc_df  #[[sheet.cell(row_index, col_index).value for row_index in range(1, sheet.nrows)] for col_index in range(sheet.ncols)]
        for i, load in enumerate(data['load']):
            if "F" in str(load).upper():
                dlc = data['name'][i].lower().replace("dlc", "")
                def fmt(v):
                    if "#" in str(v):
                        return v
                    else:
                        if v == "":
                            return 0
                        else:
                            return float(v) / 100
                dlc_dist = fmt(data['dlc_dist'][i])
                wsp_dist = data['wsp_dist'][i]
                wsp = data['wsp'][i]
                if wsp_dist.lower() == "weibull" or wsp_dist.lower() == "rayleigh":
                    start, step, stop = [float(eval(v, globals(), self.__dict__)) for v in wsp.lower().split(":")]
                    wsp_dist = Weibull(self.vref * .2, 2, start, stop, step)
                else:
                    wsp = [(eval(v, globals(), self.__dict__)) for v in str(wsp).lower().replace("/", ",").split(",")]
                    wsp_dist = [fmt(v) for v in str(wsp_dist).lower().replace("/", ",").split(",")]
                    assert len(wsp) == len(wsp_dist), "\nWsp: %s\nWsp_dist: %s" % (wsp , wsp_dist)
                    wsp_dist = {k:v  for k, v in zip(wsp, wsp_dist)}
                wdir_dist = [float(v) for v in str(data['wdir_dist'][i]).replace("/", ",").split(",")]
                wdir = [float(v) for v in str(data['wdir'][i]).replace("/", ",").split(",")]
                fatigue_dist[dlc] = (dlc_dist, wsp_dist, {k:v / 100 for k, v in zip(wdir, wdir_dist)})
        return fatigue_dist

    def file_hour_lst(self):
        """Create a list of (filename, hours_pr_year) that can be used as input for LifeTimeEqLoad

        Returns
        -------
        file_hour_lst : list
            [(filename, hours),...] where\n
            - filename is the name of the file, including path
            - hours is the number of hours pr. 20 year of this file
        """

        fh_lst = []
        dlc_dict = self.fatigue_distribution()
        for dlc_id in sorted(dlc_dict.keys()):
            dlc_dist, wsp_dict, wdir_dict = dlc_dict[dlc_id]
            for wsp in sorted(wsp_dict.keys()):
                wsp_dist = wsp_dict[wsp]
                for wdir in sorted(wdir_dict.keys()):
                    wdir_dist = wdir_dict[wdir]
                    if not hasattr(self, "res_folder"):
                        folder = ""
                    elif "%" in self.res_folder:
                        folder = self.res_folder % dlc_id
                    else:
                        folder = self.res_folder
                    files = glob.glob(os.path.join(self.res_path , folder, "dlc%s_wsp%02d_wdir%03d*.sel" % (dlc_id, wsp, wdir % 360)))
                    for f in sorted(files):
                        if "#" in str(dlc_dist):
                            duration = SelFile(f).duration
                            dlc_dist = float(dlc_dist[1:]) * duration / (60 * 60 * 24 * 365)
                        if "#" in str(wsp_dist):
                            total = sum([float(v[1:]) for v in wsp_dict.values()])
                            wsp_dist = float(wsp_dist[1:]) / total
                        f_prob = dlc_dist * wsp_dist * wdir_dist / len(files)
                        f_hours_pr_20year = HOURS_PR_20YEAR * f_prob
                        fh_lst.append((f, f_hours_pr_20year))
        return fh_lst

    def dlc_lst(self, load='all'):
        dlc_lst = np.array(self.dlc_df['name'])[np.array([load == 'all' or load.lower() in d.lower() for d in self.dlc_df['load']])]
        return [v.lower().replace('dlc', '') for v in dlc_lst]

    @cache_function
    def psf(self):
        return {dlc.lower().replace('dlc', ''): float((psf, 1)[psf == ""]) for dlc, psf in zip(self.dlc_df['name'], self.dlc_df['psf']) if dlc != ""}

if __name__ == "__main__":
    dlc_hl = DLCHighLevel(r'X:\NREL5MW\dlc.xlsx')
    #print (DLCHighLevelInputFile(r'C:\mmpe\Projects\DLC.xlsx').sensor_info(0, 0, 1)['Name'])
    #print (dlc_dict()['64'])
    #print (dlc_hl.fatigue_distribution()['64'])
    print (dlc_hl.file_hour_lst(r"X:\NREL5MW/C0008/res/"))


