from datetime import datetime
import os
import numpy as np


BINARY = "BINARY"
ASCII = "ASCII"
class SelFile(object):
    """Class for reading HAWC2 sel-files
    Attributes
    ----------
    version_id : str
        Version id
    created : datetime.datetime
        Date and Time of creation
    result_file : str
        Result file name read from file. Note that dat-filename are often used instead
    dat_filename : str
        Result file name (sel-basename + ".dat")
    scans : int
        Number of observations/rows/scans
    channels : int
        Number of channels/sensors/columns (same as 'no_sensors')
    no_sensors : int
        Number of channels/sensors/columns (same as 'channels')
    time : float
        Duration of simulation (same as duration)
    duration : float
        Duration of simulation (same as time)
    format : {"BINARY", "ASCII"}
        Result file format
    sensors : array_like
        List of (id, name, unit, description)-tuples
    scalefactors : array_like
        Array of scale factors. Only if format is BINARY
    """

    def __init__(self, sel_filename):
        if not os.path.isfile(sel_filename) or os.path.splitext(sel_filename)[1] != ".sel":
            raise Warning("%s cannot be found or is not a legal *.sel file" % os.path.realpath(sel_filename))
        with open(sel_filename, encoding='utf-8') as f:
            lines = f.readlines()


        # parse info lines
        def val(line):
            return line.split(" : ")[1].strip()
        try:
            self.version_id = val(lines[1])
            _created = "%s %s" % (val(lines[3]), val(lines[2]))
            self.created = datetime.strptime(_created, "%d:%m.%Y %H:%M:%S")
            self.result_file = os.path.basename(val(lines[5]))

        except:
            pass

        self.dat_filename = sel_filename[:-4] + ".dat"

        _info = lines[8].split()
        self.scans = int(_info[0])
        self.no_sensors = self.channels = int(_info[1])
        self.duration = self.time = float(_info[2])
        if len(_info) > 3:
            self.format = _info[3].upper()
        elif self.no_sensors * self.scans * 2 == os.path.getsize(self.dat_filename):
            self.format = "BINARY"
        else:
            self.format = "ASCII"

        # parse sensor info
        self.sensors = []
        for i, line in enumerate(lines[12:12 + self.no_sensors]):
            try:
                ch = int(line[:7])
                name = str(line[7:43]).strip()
                unit = str(line[43:48]).strip()
                description = str(line[49:]).strip()
                self.sensors.append((ch, name, unit, description))
            except ValueError:
                raise Warning("Value error in '%s'\nLine %d supposed to contain number, name, unit and description of sensor %d, but contained:\n%s" % (sel_filename, i + 13, i, line.strip()))

        # parse scalefactors
        if self.format == "BINARY":
            self.scale_factors = np.array([float(factor) for factor in lines[12 + self.no_sensors + 2:]])


def save(sel_filename, version, time, scans, no_sensors, duration, sensors, scale_factors=None):
    """Create HAWC2 sel-file

    Parameters
    ----------
    sel_filename : str
        Filename
    version : str
        Version tag
    time : datetime.datetime
        Date and time of creation
    scans : int
        Number of scans/observations/rows
    no_sensors : int
        Number of sensors/channels/columns
    duration : int, float
        Duration of simulation
    sensors : array_like
        List of (name, unit, description)-tuples
    scale_factors : array_like, optional
        Array of scale factors. Only for BINARY format
    """
    lines = []
    linesep = "_" * 120
    lines.append(linesep)
    lines.append("  Version ID : %s" % version)
    lines.append(" " * 60 + "Time : %s" % time.strftime("%H:%M:%S"))
    lines.append(" " * 60 + "Date : %s" % time.strftime("%d:%m.%Y"))
    lines.append(linesep)
    lines.append("  Result file : %s.dat" % sel_filename[:-4])
    lines.append(linesep)
    lines.append("   Scans    Channels    Time [sec]      Format")
    lines.append("   %8s    %3s       %8.3f       %s" % (scans, no_sensors, duration, ("BINARY", "ASCII")[scale_factors is None]))
    lines.append("")
    lines.append('  Channel   Variable Description               ')
    lines.append("")
    for nr, sensor in enumerate(sensors, 1):
        name, unit, description = sensor
        lines.append("  %4s      %-30s %-10s %s" % (nr, name[:30], unit[:10], description[:512]))
    lines.append(linesep)
    if scale_factors is not None:
        lines.append("Scale factors:")
        for sf in scale_factors:
            lines.append("  %.5E" % sf)

    with open(sel_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

