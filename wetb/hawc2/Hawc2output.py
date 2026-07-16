# -*- coding: utf-8 -*-
"""
Author:
    Bjarne S. Kallesoee


Description:
    Reads HAWC2 output data and returns metadata-labelled DataFrames.

call ex.:
    # creat data file object, call without extension, but with parth
    file = Hawc2Output("HAWC2ex/test")
    # if called with ReadOnly = 1 as
    file = Hawc2Output("HAWC2ex/test",ReadOnly=1)
    # no channels a stored in memory, otherwise read channels are stored for reuse

    # channels are called by a list
    file([0,2,1,1])  => channels 1,3,2,2
    # if empty all channels are returned
    file()  => all channels as 1,2,3,...
    file.t => time vector

1. version: 19/4-2011
2. version: 5/11-2015 fixed columns to get description right, fixed time vector (mmpe@dtu.dk)

Need to be done:
    * add error handling for allmost every thing

"""
import numpy as np
import os

from wetb import gtsdf
from wetb.prepost import misc
from wetb.hawc2.sensor_search import SensorSearch

import pandas as pd

################################################################################
################################################################################
################################################################################
# Read HAWC2 class
################################################################################


class Hawc2Output(object):
    """
    """
################################################################################
# read *.sel file

    def _ReadSelFile(self):
        """
        Some title
        ==========

        Using docstrings formatted according to the reStructuredText specs
        can be used for automated documentation generation with for instance
        Sphinx: http://sphinx.pocoo.org/.

        Parameters
        ----------
        signal : ndarray
            some description

        Returns
        -------
        output : int
            describe variable
        """

        # read *.sel hawc2 output file for result info
        if self.FileName.lower().endswith('.sel'):
            self.FileName = self.FileName[:-4]
        Lines = misc.readlines_try_encodings(self.FileName + '.sel')
        # findes general result info (number of scans, number of channels,
        # simulation time and file format)
        temp = Lines[8].split()
        self.NrSc = int(temp[0])
        self.NrCh = int(temp[1])
        self.Time = float(temp[2])
        self.Freq = self.NrSc / self.Time
        self.t = np.linspace(0, self.Time, self.NrSc + 1)[1:]
        Format = temp[3]
        # reads channel info (name, unit and description)
        Name = []
        Unit = []
        Description = []
        for i in range(0, self.NrCh):
            temp = str(Lines[i + 12][12:43])
            Name.append(temp.strip())
            temp = str(Lines[i + 12][43:54])
            Unit.append(temp.strip())
            temp = str(Lines[i + 12][54:-1])
            Description.append(temp.strip())
        self.ChInfo = [Name, Unit, Description]
        # if binary file format, scaling factors are read
        if Format.lower() == 'binary':
            self.ScaleFactor = np.zeros(self.NrCh)
            self.FileFormat = 'HAWC2_BINARY'
            for i in range(0, self.NrCh):
                self.ScaleFactor[i] = float(Lines[i + 12 + self.NrCh + 2])
        else:
            self.FileFormat = 'HAWC2_ASCII'
################################################################################
# read sensor file for FLEX format

    def _ReadSensorFile(self):
        # read sensor file used if results are saved in FLEX format
        DirName = os.path.dirname(self.FileName + ".int")
        try:
            Lines = misc.readlines_try_encodings(os.path.join(DirName, r"sensor"))
        except IOError:
            print("can't finde sensor file for FLEX format")
            return
        # reads channel info (name, unit and description)
        self.NrCh = 0
        Name = []
        Unit = []
        Description = []
        for i in range(2, len(Lines)):
            temp = Lines[i]
            if not temp.strip():
                break
            self.NrCh += 1
            temp = str(Lines[i][38:45])
            Unit.append(temp.strip())
            temp = str(Lines[i][45:53])
            Name.append(temp.strip())
            temp = str(Lines[i][53:])
            Description.append(temp.strip())
        self.ChInfo = [Name, Unit, Description]
        # read general info from *.int file
        fid = open(self.FileName + ".int", 'rb')
        fid.seek(4 * 19)
        if not np.fromfile(fid, 'int32', 1) == self.NrCh:
            print("number of sensors in sensor file and data file are not consisten")
        fid.seek(4 * (self.NrCh) + 8, 1)
        time_start, time_step = np.fromfile(fid, 'f', 2)
        self.Freq = 1 / time_step
        self.ScaleFactor = np.fromfile(fid, 'f', self.NrCh)
        fid.seek(2 * 4 * self.NrCh + 48 * 2)
        self.NrSc = int(len(np.fromfile(fid, 'int16')) / self.NrCh)
        self.Time = self.NrSc * time_step
        self.t = np.arange(0, self.Time, time_step) + time_start
        fid.close()
################################################################################
# init function, load channel and other general result file info

    def __init__(self, FileName, ReadOnly=0):
        self.FileName = FileName
        self.ReadOnly = ReadOnly
        self.Iknown = []  # to keep track of what has been read all ready
        self.Data = np.zeros(0)

        if FileName.lower().endswith('.sel') or os.path.isfile(FileName + ".sel"):
            self._ReadSelFile()
        elif FileName.lower().endswith('.int') or os.path.isfile(self.FileName + ".int"):
            self.FileFormat = 'FLEX'
            self._ReadSensorFile()
        elif FileName.lower().endswith('.hdf5') or os.path.isfile(self.FileName + ".hdf5"):
            self.FileFormat = 'GTSDF'
            self.ReadGtsdf()
        else:
            raise ValueError(f"Unknown file: {FileName}")

        self.sensor_search = SensorSearch(*self.ChInfo)
################################################################################
# Read results in binary format

    def ReadBinary(self, ChVec=None):
        ChVec = self._normalize_chvec(ChVec)
        with open(self.FileName + '.dat', 'rb') as fid:
            data = np.zeros((self.NrSc, len(ChVec)))
            j = 0
            for i in ChVec:
                fid.seek(i * self.NrSc * 2, 0)
                data[:, j] = np.fromfile(fid, 'int16', self.NrSc) * self.ScaleFactor[i]
                j += 1
        return data
################################################################################
# Read results in ASCII format

    def ReadAscii(self, ChVec=None):
        ChVec = self._normalize_chvec(ChVec)
        temp = np.loadtxt(self.FileName + '.dat', usecols=ChVec)
        return temp.reshape((temp.shape[0], len(ChVec)))
################################################################################
# Read results in FLEX format

    def ReadFLEX(self, ChVec=None):
        ChVec = self._normalize_chvec(ChVec)
        fid = open(self.FileName + ".int", 'rb')
        fid.seek(2 * 4 * self.NrCh + 48 * 2)
        temp = np.fromfile(fid, 'int16')
        temp = temp.reshape(self.NrSc, self.NrCh)
        fid.close()
        return np.dot(temp[:, ChVec], np.diag(self.ScaleFactor[ChVec]))
################################################################################
# Read results in GTSD format

    def ReadGtsdf(self, ChVec=None):
        fn = self.FileName
        if fn[-5:].lower() != '.hdf5':
            fn += '.hdf5'
        self.t, data, info = gtsdf.load(fn)
        self.Time = self.t
        self.ChInfo = [['Time'] + info['attribute_names'],
                       ['s'] + info['attribute_units'],
                       ['Time'] + info['attribute_descriptions']]
        if 'htc_input' in info:
            self.ChInfo.append(['Time'] + info['htc_input'])

        self.NrCh = data.shape[1] + 1
        self.NrSc = data.shape[0]
        self.Freq = self.NrSc / self.Time
        self.FileFormat = 'GTSDF'
        self.gtsdf_description = info['description']
        self.gtsdf_dtype = info['dtype']
        data = np.hstack([self.Time[:, np.newaxis], data])
        
        ChVec = self._normalize_chvec(ChVec)
        data = data[:, ChVec]
        return data

################################################################################
    # One stop call for reading all data formats

    def ReadAll(self, ChVec=None):
        if ChVec is not None and np.asarray(ChVec).size == 0:
            ChVec = None

        if self.FileFormat == 'HAWC2_BINARY':
            return self.ReadBinary(ChVec)
        elif self.FileFormat == 'HAWC2_ASCII':
            return self.ReadAscii(ChVec)
        elif self.FileFormat == 'GTSDF':
            return self.ReadGtsdf(ChVec)
        else:
            return self.ReadFLEX(ChVec)

    def get_sensor_id(self, **kwargs):
        return self.sensor_search.get_sensor_id(**kwargs)

    def _normalize_chvec(self, ChVec):
        """Return channel numbers as a one-dimensional integer array."""
        if ChVec is None:
            return np.arange(self.NrCh, dtype=int)

        ChVec = np.asarray(ChVec, dtype=int)
        if ChVec.ndim == 0:
            ChVec = ChVec.reshape(1)
        if ChVec.ndim != 1:
            raise ValueError("ChVec must be one-dimensional")
        
        if ChVec.size and (ChVec.min() < 0 or ChVec.max() >= self.NrCh):
            raise ValueError("Channel number out of range")
        
        return ChVec


    def _get_data(self, ChVec):
        if ChVec.size == 0:
            return np.empty((self.NrSc, 0))

        # if ReadOnly, read data but no storeing in memory
        if self.ReadOnly:
            return self.ReadAll(ChVec)

        # if not ReadOnly, sort in known and new channels, read new channels
        # and return all requested channels
        I1 = []
        I2 = []  # I1=Channel mapping, I2=Channels to be read
        for i in ChVec:
            try:
                I1.append(self.Iknown.index(i))
            except Exception:
                self.Iknown.append(i)
                I2.append(i)
                I1.append(len(self.Iknown) - 1)
        # read new channels
        if I2:
            temp = self.ReadAll(I2)
            # add new channels to Data
            if self.Data.any():
                self.Data = np.append(self.Data, temp, axis=1)
            # if first call, so Data is empty
            else:
                self.Data = temp
        return self.Data[:, tuple(I1)]

    def __call__(
        self,
        ChVec=None,
        htc=None,
        name=None,
        unit=None,
        desc=None,
        label=None,
    ):

        if htc is not None or name is not None or unit is not None or desc is not None or label is not None:
            sensor_df = self.sensor_search(
                htc=htc,
                name=name,
                unit=unit,
                desc=desc,
                label=label,
            )
            ChVec = sensor_df["id"].to_numpy(dtype=int)
        else:
            ChVec = self._normalize_chvec(ChVec)
            sensor_df = self.sensor_search(id=ChVec)

        data = self._get_data(ChVec)
        columns = pd.MultiIndex.from_frame(sensor_df)
        df = pd.DataFrame(data, columns=columns)
        for i, name in enumerate(df.columns.names):
            values = [column[i] for column in df.columns]
            object.__setattr__(df, name, values)
        return df


