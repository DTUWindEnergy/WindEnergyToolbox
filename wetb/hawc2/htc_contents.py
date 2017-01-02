'''
Created on 20/01/2014

@author: MMPE

See documentation of HTCFile below

'''
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import int
from builtins import str
from future import standard_library
standard_library.install_aliases()
from collections import OrderedDict
import collections


class OrderedDict(collections.OrderedDict):
    pass

    def __str__(self):
        return "\n".join(["%-30s\t %s" % ((str(k) + ":"), str(v)) for k, v in self.items()])


def parse_next_line(lines):
    _3to2list = list(lines.pop(0).split(";"))
    line, comments, = _3to2list[:1] + [_3to2list[1:]]
    comments = ";".join(comments).rstrip()
    while lines and lines[0].lstrip().startswith(";"):
        comments += "\n%s" % lines.pop(0).rstrip()
    return line.strip(), comments

def fmt_value(v):
    try:
        if int(float(v)) == float(v):
            return int(float(v))
        return float(v)
    except ValueError:
        return v

c = 0
class HTCContents(object):
    lines = []
    contents = None
    name_ = ""

    def __setitem__(self, key, value):
        self.contents[key] = value

    def __getitem__(self, key):
        if isinstance(key, str):
            key = key.replace(".", "/")
            if "/" in key:
                keys = key.split('/')
                val = self.contents[keys[0]]
                for k in keys[1:]:
                    val = val[k]
                return val
            return self.contents[key]
        else:
            return self.values[key]

    def __getattr__(self, *args, **kwargs):
        try:
            return object.__getattribute__(self, *args, **kwargs)
        except:
            return self.contents[args[0]]

    def __setattr__(self, *args, **kwargs):
        _3to2list1 = list(args)
        k, v, = _3to2list1[:1] + _3to2list1[1:]
        if k in dir(self):  # in ['section', 'filename', 'lines']:
            return object.__setattr__(self, *args, **kwargs)
        if isinstance(v, str):
            v = [fmt_value(v) for v in v.split()]
        if not isinstance(v, (list, tuple)):
            v = [v]
        self.contents[k] = HTCLine(k, v, "")

    def __delattr__(self, *args, **kwargs):
        k, = args
        if k in self:
            del self.contents[k]

    def __iter__(self):
        return iter(self.contents.values())


    def __contains__(self, key):
        if self.contents is None:
            return False
        return key in self.contents

    def get(self, section, default=None):
        try:
            return self[section]
        except KeyError:
            return default

    def keys(self):
        return list(self.contents.keys())

    def _add_contents(self, contents):
        if contents.name_ not in self:
            self[contents.name_] = contents
        else:
            ending = "__2"
            while contents.name_ + ending in self:
                ending = "__%d" % (1 + float("0%s" % ending.replace("__", "")))
            self[contents.name_ + ending] = contents

    def add_section(self, name, allow_duplicate_section=False):
        if name in self and allow_duplicate_section is False:
            return self[name]
        section = HTCSection(name)
        self._add_contents(section)
        return section

    def add_line(self, name, values, comments):
        line = HTCLine(name, values, comments)
        self._add_contents(line)
        return line




class HTCSection(HTCContents):
    end_comments = ""
    begin_comments = ""
    def __init__(self, name, begin_comments="", end_comments=""):
        self.name_ = name
        self.begin_comments = begin_comments
        self.end_comments = end_comments
        self.contents = OrderedDict()

    @staticmethod
    def from_lines(lines):
        line, begin_comments = parse_next_line(lines)
        name = line[6:].lower()
        if name == "output":
            section = HTCOutputSection(name, begin_comments)
        elif name.startswith("output_at_time"):
            section = HTCOutputAtTimeSection(name, begin_comments)
        else:
            section = HTCSection(name, begin_comments)
        while lines:
            if lines[0].lower().startswith("begin"):
                section._add_contents(HTCSection.from_lines(lines))
            elif lines[0].lower().startswith("end"):
                line, section.end_comments = parse_next_line(lines)
                break
            else:
                section._add_contents(section.line_from_line(lines))
        return section

    def line_from_line(self, lines):
        return HTCLine.from_lines(lines)

    def __str__(self, level=0):
        s = "%sbegin %s;%s\n" % ("  "*level, self.name_, ("", "\t" + self.begin_comments)[bool(self.begin_comments.strip())])
        s += "".join([c.__str__(level + 1) for c in self])
        s += "%send %s;%s\n" % ("  "*level, self.name_, ("", "\t" + self.end_comments)[self.end_comments.strip() != ""])
        return s

class HTCLine(HTCContents):
    values = None
    comments = ""
    def __init__(self, name, values, comments):
        if "__" in name:
            name = name[:name.index("__")]
        self.name_ = name
        self.values = list(values)
        self.comments = comments

    def __repr__(self):
        return str(self)

    def __str__(self, level=0):
        if self.name_ == "":
            return ""
        return "%s%s%s;%s\n" % ("  "*(level), self.name_,
                                  ("", "\t" + self.str_values())[bool(self.values)],
                                  ("", "\t" + self.comments)[bool(self.comments.strip())])
    def str_values(self):
        return " ".join([str(v).lower() for v in self.values])

    def __getitem__(self, key):
        return self.values[key]

    @staticmethod
    def from_lines(lines):
        line, end_comments = parse_next_line(lines)
        if len(line.split()) > 0:
            _3to2list3 = list(line.split())
            name, values, = _3to2list3[:1] + [_3to2list3[1:]]
        else:
            name = line
            values = []

        values = [fmt_value(v) for v in values]
        return HTCLine(name, values, end_comments)


    def remove(self):
        self.name_ = ""
        self.values = []
        self.comments = ""


class HTCOutputSection(HTCSection):
    sensors = None
    def __init__(self, name, begin_comments="", end_comments=""):
        HTCSection.__init__(self, name, begin_comments=begin_comments, end_comments=end_comments)
        self.sensors = []

    def add_sensor(self, type, sensor, values=[], comment="", nr=None):
        self._add_sensor(HTCSensor(type, sensor, values, comment), nr)


    def _add_sensor(self, htcSensor, nr=None):
        if nr is None:
            nr = len(self.sensors)
        self.sensors.insert(nr, htcSensor)


    def line_from_line(self, lines):
        while lines[0].strip() == "":
            lines.pop(0)
        name = lines[0].split()[0].strip()
        if name in ['filename', 'data_format', 'buffer', 'time']:
            return HTCLine.from_lines(lines)
        else:
            return HTCSensor.from_lines(lines)

    def _add_contents(self, contents):
        if isinstance(contents, HTCSensor):
            self._add_sensor(contents)
        else:
            return HTCSection._add_contents(self, contents)


    def __str__(self, level=0):
        s = "%sbegin %s;%s\n" % ("  "*level, self.name_, ("", "\t" + self.begin_comments)[len(self.begin_comments.strip()) > 0])
        s += "".join([c.__str__(level + 1) for c in self])
        s += "".join([s.__str__(level + 1) for s in self.sensors])
        s += "%send %s;%s\n" % ("  "*level, self.name_, ("", "\t" + self.end_comments)[self.end_comments.strip() != ""])
        return s

class HTCOutputAtTimeSection(HTCOutputSection):
    type = None
    time = None
    def __init__(self, name, begin_comments="", end_comments=""):
        if len(name.split()) < 3:
            raise ValueError('"keyword" and "time" arguments required for output_at_time command:\n%s' % name)
        name, self.type, time = name.split()
        self.time = float(time)
        HTCOutputSection.__init__(self, name, begin_comments=begin_comments, end_comments=end_comments)

    def __str__(self, level=0):
        s = "%sbegin %s %s %s;%s\n" % ("  "*level, self.name_, self.type, self.time, ("", "\t" + self.begin_comments)[len(self.begin_comments.strip())])
        s += "".join([c.__str__(level + 1) for c in self])
        s += "".join([s.__str__(level + 1) for s in self.sensors])
        s += "%send %s;%s\n" % ("  "*level, self.name_, ("", "\t" + self.end_comments)[self.end_comments.strip() != ""])
        return s

class HTCSensor(HTCLine):
    type = ""
    sensor = ""
    values = []

    def __init__(self, type, sensor, values, comments):
        self.type = type
        self.sensor = sensor
        self.values = values
        self.comments = comments

    @staticmethod
    def from_lines(lines):
        line, comments = parse_next_line(lines)
        if len(line.split()) > 2:
            _3to2list5 = list(line.split())
            type, sensor, values, = _3to2list5[:2] + [_3to2list5[2:]]
        elif len(line.split()) == 2:
            type, sensor = line.split()
            values = []
        else:
            type, sensor, values = "", "", []
        def fmt(v):
            try:
                if int(float(v)) == float(v):
                    return int(float(v))
                return float(v)
            except ValueError:
                return v
        values = [fmt(v) for v in values]
        return HTCSensor(type, sensor, values, comments)

    def __str__(self, level=0):
        return "%s%s %s%s;%s\n" % ("  "*(level),
                                self.type,
                                self.sensor,
                                ("", "\t" + self.str_values())[bool(self.values)],
                                ("", "\t" + self.comments)[bool(self.comments.strip())])

class HTCDefaults(object):


    empty_htc = """begin simulation;
        time_stop 600;
        solvertype    1;    (newmark)
        on_no_convergence continue;
        convergence_limits 1E3 1.0 1E-7; ; . to run again, changed 07/11
        begin newmark;
          deltat    0.02;
        end newmark;
    end simulation;
    ;
    ;----------------------------------------------------------------------------------------------------------------------------------------------------------------
    ;
    begin new_htc_structure;
      begin orientation;
      end orientation;
      begin constraint;
      end constraint;
    end new_htc_structure;
    ;
    ;----------------------------------------------------------------------------------------------------------------------------------------------------------------
    ;
    begin wind ;
      density                 1.225 ;
      wsp                     10   ;
      tint                    1;
      horizontal_input        1     ;            0=false, 1=true
      windfield_rotations     0 0.0 0.0 ;    yaw, tilt, rotation
      center_pos0             0 0 -30 ; hub heigth
      shear_format            1   0;0=none,1=constant,2=log,3=power,4=linear
      turb_format             0     ;  0=none, 1=mann,2=flex
      tower_shadow_method     0     ;  0=none, 1=potential flow, 2=jet
    end wind;
    ;
    ;----------------------------------------------------------------------------------------------------------------------------------------------------------------
    ;
    begin dll;
    end dll;
    ;
    ;----------------------------------------------------------------------------------------------------------------------------------------------------------------
    ;
    begin output;
      general time;
    end output;
    exit;"""


    def add_mann_turbulence(self, L=29.4, ae23=1, Gamma=3.9, seed=1001, high_frq_compensation=True,
                            filenames=None,
                            no_grid_points=(4096, 32, 32), box_dimension=(6000, 100, 100),
                            std_scaling=(1, .8, .5)):
        wind = self.add_section('wind')
        wind.turb_format = 1
        mann = wind.add_section('mann')
        if 'create_turb_parameters' in mann:
            mann.create_turb_parameters.values = [L, ae23, Gamma, seed, int(high_frq_compensation)]
        else:
            mann.add_line('create_turb_parameters', [L, ae23, Gamma, seed, int(high_frq_compensation)], "L, alfaeps, gamma, seed, highfrq compensation")
        if filenames is None:
            fmt = "mann_l%.1f_ae%.2f_g%.1f_h%d_%dx%dx%d_%.3fx%.2fx%.2f_s%04d%c.turb"
            import numpy as np
            dxyz = tuple(np.array(box_dimension) / no_grid_points)
            filenames = ["./turb/" + fmt % ((L, ae23, Gamma, high_frq_compensation) + no_grid_points + dxyz + (seed, uvw)) for uvw in ['u', 'v', 'w']]
        if isinstance(filenames, str):
            filenames = ["./turb/%s_s%04d%s.bin" % (filenames, seed, c) for c in ['u', 'v', 'w']]
        for filename, c in zip(filenames, ['u', 'v', 'w']):
            setattr(mann, 'filename_%s' % c, filename)
        for c, n, dim in zip(['u', 'v', 'w'], no_grid_points, box_dimension):
            setattr(mann, 'box_dim_%s' % c, "%d %.4f" % (n, dim / (n - 1)))
        if std_scaling is None:
            mann.dont_scale = 1
        else:
            try:
                del mann.dont_scale
            except KeyError:
                pass
            mann.std_scaling = "%f %f %f" % std_scaling


    def import_dtu_we_controller_input(self, filename):
        dtu_we_controller = [dll for dll in self.dll if dll.name[0] == 'dtu_we_controller'][0]
        with open (filename) as fid:
            lines = fid.readlines()
        K_r1 = float(lines[1].replace("K = ", '').replace("[Nm/(rad/s)^2]", ''))
        Kp_r2 = float(lines[4].replace("Kp = ", '').replace("[Nm/(rad/s)]", ''))
        Ki_r2 = float(lines[5].replace("Ki = ", '').replace("[Nm/rad]", ''))
        Kp_r3 = float(lines[7].replace("Kp = ", '').replace("[rad/(rad/s)]", ''))
        Ki_r3 = float(lines[8].replace("Ki = ", '').replace("[rad/rad]", ''))
        KK = lines[9].split("]")
        KK1 = float(KK[0].replace("K1 = ", '').replace("[deg", ''))
        KK2 = float(KK[1].replace(", K2 = ", '').replace("[deg^2", ''))
        cs = dtu_we_controller.init
        cs.constant__11.values[1] = "%.6E" % K_r1
        cs.constant__12.values[1] = "%.6E" % Kp_r2
        cs.constant__13.values[1] = "%.6E" % Ki_r2
        cs.constant__16.values[1] = "%.6E" % Kp_r3
        cs.constant__17.values[1] = "%.6E" % Ki_r3
        cs.constant__21.values[1] = "%.6E" % KK1
        cs.constant__22.values[1] = "%.6E" % KK2


