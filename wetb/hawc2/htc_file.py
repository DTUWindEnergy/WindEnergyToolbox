'''
Created on 20/01/2014

@author: MMPE
'''
from collections import OrderedDict
import os
import time
import numpy as np
from wetb.functions.process_exec import pexec

class HTCSection(object):
    section = []
    def __init__(self, section):
        self.section = section

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.section2str(self.section)

    def section2str(self, section, level=0):
        #list(list(section.items())[5][1][0].items())[8]
        s = ""
        endcomment = ""
        if isinstance(section, tuple):
            section, endcomment = section
        for k, vs in section.items():
            for v, comment in ([vs], vs)[isinstance(vs, list)]:
                if isinstance(v, OrderedDict):
                    s += "%sbegin %s;\t%s\n" % ("  "*level, k, comment[0])
                    s += self.section2str(v, level + 1)
                    s += "%send %s;\t%s\n" % ("  "*level, k, comment[1])
                elif isinstance(v, list):
                    #s += "%sbegin %s;%s\n" % ("  "*level, k, comment[0])
                    v = [_v.strip() for _v in v]
                    s += "  "*(level) + ("\n%s" % ("  "*(level))).join(v) + ("", "\n")[len(v) > 0]
                    #s += "%send %s;%s\n" % ("  "*level, k, comment[1])
                elif v is None:
                    pass
                else:
                    if k.startswith(";"):
                        s += ("%s;\t%s" % (v, comment)).rstrip() + "\n"
                    else:
                        s += ("%s%s\t%s;\t%s" % ("  "*(level), k, v, comment)).rstrip() + "\n"
        return s + endcomment


    def __getattribute__(self, *args, **kwargs):
        try:
            return object.__getattribute__(self, *args, **kwargs)
        except:
            #if args[0] not in self.section:
            #    return ""
            if isinstance(self.section[args[0]][0], OrderedDict):
                return HTCSection(self.section[args[0]][0])
            return self.section[args[0]][0]

    def __setattr__(self, *args, **kwargs):
        k, v = args
        if k in dir(self):  # in ['section', 'filename', 'lines']:
            return object.__setattr__(self, *args, **kwargs)
        if k not in self.section:
            if isinstance(v, (tuple, list)) and len(v) == 2:
                self.section[k] = v
            else:
                self.section[k] = (v, "")
        else:
            if isinstance(v, (tuple, list)) and len(v) == 2:
                self.section[k] = (str(v_) for v_ in v)
            else:
                comment = self.section[k][1]
                self.section[args[0]] = (str(args[1]), comment)

    def __delattr__(self, *args, **kwargs):
        k, = args
        if k in self.section:
            del self.section[k]

    def __eq__(self, other):
        return str(self) == other

    def __getitem__(self, section):
        if "/" in section:
            sections = section.split('/')
            val = self.section[sections[0]][0]
            for s in sections[1:]:
                val = val[s][0]
            return val
        return self.section[section][0]

    def add_section(self, name):
        if name not in self.section:
            self.section[name] = (OrderedDict(), ("", ""))
        return HTCSection(self.section[name][0])



class HTCFile(HTCSection):
    """Htc file wrapper

    Examples
    --------
    # load
    >>> htcfile = HTCFile("test.htc")

    # access (3 methods)
    >>> print (htcfile['simulation']['time_stop'][0])
    100
    >>> print (htcfile['simulation/time_stop'])
    100
    >>> print (htcfile.simulation.time_stop)
    100


    # set values
    >>> htcfile.simulation.time_stop = 200

    # delete keys
    >>> del htcfile.simulation.logfile

    # safe key deleting
    >>> try:
    >>>     del htcfile.hydro.water_properties.water_kinematics_dll
    >>> except KeyError:
    >>>     pass


    # save
    >>> htcfile.save() # replace existing file
    >>> htcfile.save('newfilename.htc')

    # simulate
    >>> htcfile.simulate('<hawc2path>/hawc2mb.exe')
    """


    filename = None
    lines = []
    def __init__(self, filename=None):
        if filename is None:
            self.filename = 'empty.htc'
            self.lines = empty_htc.split("\n")
        else:
            self.filename = filename

            with open(filename) as fid:
                self.lines = fid.readlines()
        self.section = self.parse_section()

    def set_name(self, name):
        self.filename = "%s.htc" % name
        self.simulation.logfile = "./log/%s.log" % name
        self.output.filename = "./res/%s" % name


    def parse_line(self):
        global curr_line
        line, *comments = self.lines[curr_line].split(";")
        comments = ";".join(comments).rstrip()
        while curr_line + 1 < len(self.lines) and self.lines[curr_line + 1].strip().startswith(";"):
            curr_line += 1
            comments += "\n%s" % self.lines[curr_line].rstrip()
        return line, comments

    def key_value(self, line):
        if " " in line.strip() or "\t" in line.strip():
            d = 9999
            if " " in line.strip():
                d = line.strip().index(" ")
            if "\t" in line.strip() and line.strip().index('\t') < d:
                d = line.strip().index('\t')
            key = line.strip()[:d]
            value = line.strip()[d + 1:]
            return key, value
        else:
            return None, None

    def parse_section(self, startline=0):
        global curr_line
        section = OrderedDict()
        curr_line = startline
        while curr_line < len(self.lines):
            line, comments = self.parse_line()
            if line.strip().lower().startswith("begin "):
                key = line.strip()[6:]
                if key == "output":
                    keys = ['filename', 'data_format', 'buffer', 'time']
                    sensors = []
                    output = OrderedDict()
                    for k in ['filename', 'data_format', 'buffer', 'time']:
                        output[k] = (None, "")
                    while not self.lines[curr_line + 1].lower().strip().startswith('end'):
                        curr_line += 1
                        line, comment = self.parse_line()
                        k, v = self.key_value(line)
                        if k in keys:
                            output[k] = (v, comment)
                        else:
                            sensors.append("%s;%s" % (line, comment))
                    curr_line += 1
                    line, endcomments = self.parse_line()
                    output['sensors'] = (sensors, ("", ""))
                    section[key] = (output, (comments, endcomments))

                else:
                    curr_line += 1
                    value, end_comments = self.parse_section(curr_line)
                    while self.lines[curr_line + 1].strip().startswith(";"):
                        curr_line += 1
                        end_comments += "\n%s" % self.lines[curr_line].strip()
                    if key in section:
                        if not isinstance(section[key], list):
                            section[key] = [section[key], (value, (comments, end_comments))]
                        else:
                            section[key].append((value, (comments, end_comments)))
                    else:
                        section[key] = (value, (comments, end_comments))
            elif line.lower().strip().startswith("end "):
                return section, comments
            elif line.lower().strip().startswith('exit'):
                pass
            elif " " in line.strip() or "\t" in line.strip():

                key, value = self.key_value(line)
                if key in section:
                    if not isinstance(section[key], list):
                        section[key] = [section[key], (value, comments)]
                    else:
                        section[key].append((value, comments))
                else:
                    section[key] = (value, comments)
            else:
                section[';%d' % (curr_line + 1)] = (line, comments)
            curr_line += 1
        return section

    def __str__(self):
        return self.section2str(self.section) + "exit;"


    def save(self, filename=None):
        if filename is None:
            filename = self.filename
        else:
            self.filename = filename
        with open(filename, 'w') as fid:
            fid.write(str(self))

    def simulate(self, hawc2_path):
        self.save(os.path.join(os.path.dirname(self.filename), "auto.htc"))
        errorcode, stdout, stderr, cmd = pexec([hawc2_path, "./htc/auto.htc"], os.path.realpath("..", os.path.dirname(self.filename)))

        if 'logfile' in self['simulation']:
            with open(os.path.join(os.path.realpath(os.path.dirname(self.filename)), "../", self['simulation']['logfile'])) as fid:
                log = fid.read()
        else:
            log = stderr
        if "error" in log.lower():
            raise Exception ("Simulation failed: %s" % (log[:log.lower().index("error") + 1000]))

    def add_sensor(self, sensor, nr=None):
        if nr is None:
            nr = len(self.output.sensors)
        line, *comments = sensor.split(";")
        comments = ";".join(comments).rstrip()
        self.output.sensors.insert(nr, ("%s;\t%s" % (line.strip(), comments)).rstrip())


    def add_mann_turbulence(self, L=29.4, ae23=1, Gamma=3.9, seed=1001, high_frq_compensation=True,
                            filenames=None,
                            no_grid_points=(4096, 32, 32), box_dimension=(6000, 100, 100),
                            std_scaling=(1, .8, .5)):
        wind = self.add_section('wind')
        wind.turb_format = (1, "0=none, 1=mann,2=flex")
        mann = wind.add_section('mann')
        mann.create_turb_parameters = ("%.2f %.3f %.2f %d %d" % (L, ae23, Gamma, seed, high_frq_compensation), "L, alfaeps, gamma, seed, highfrq compensation")
        if filenames is None:
            filenames = ["./turb/turb_wsp%d_s%04d%s.bin" % (float(self.wind.wsp), seed, c) for c in ['u', 'v', 'w']]
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

if "__main__" == __name__:
    f = HTCFile(r"C:\mmpe\hawc2\models\DTU10MWRef\htc\dtu_10mw_rwt.htc")
    print (f.section2str(f['new_htc_structure']['main_body'][0]))
    f.simulate("hawc2_path")



