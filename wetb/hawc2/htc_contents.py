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
import os
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
        return v.replace("\\", "/")


c = 0


class HTCContents(object):
    lines = []
    contents = None
    name_ = ""
    parent = None

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.contents[key] = value
        elif isinstance(key, int):
            self.values[key] = value
        else:
            raise NotImplementedError
        value.parent = self

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
        if args[0] in ['__members__', '__methods__']:
            # fix python2 related issue. In py2, dir(self) calls
            # __getattr__(('__members__',)), and this call must fail unhandled to work
            return object.__getattribute__(self, *args, **kwargs)
        try:
            return object.__getattribute__(self, *args, **kwargs)
        except:
            k = args[0]
            if k.endswith("__1"):
                k = k[:-3]
            return self.contents[k]

    def __setattr__(self, *args, **kwargs):
        _3to2list1 = list(args)
        k, v, = _3to2list1[:1] + _3to2list1[1:]
        if k in dir(self):  # in ['section', 'filename', 'lines']:
            return object.__setattr__(self, *args, **kwargs)
        if isinstance(v, str):
            v = [fmt_value(v) for v in v.split()]
        if isinstance(v, HTCContents):
            self.contents[k] = v
        else:
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

    def add_section(self, name, allow_duplicate=False):
        if name in self and allow_duplicate is False:
            return self[name]
        if name == "output":
            section = HTCOutputSection(name)
        elif name.startswith("output_at_time"):
            section = HTCOutputAtTimeSection(name)
        else:
            section = HTCSection(name)
        self._add_contents(section)
        return section

    def add_line(self, name, values, comments=""):
        line = HTCLine(name, values, comments)
        self._add_contents(line)
        return line

    def location(self):
        if self.parent is None:
            return os.path.basename(self.filename)
        else:
            return self.parent.location() + "/" + self.name_

    def compare(self, other):
        my_keys = self.keys()
        other_keys = other.keys()
        s = ""
        while my_keys or other_keys:
            if my_keys:
                if (my_keys[0] in other_keys):
                    while other_keys[0] != my_keys[0]:
                        s += "\n".join(["+ %s" % l for l in str(other[other_keys.pop(0)]).strip().split("\n")]) + "\n\n"

                    s += self[my_keys.pop(0)].compare(other[other_keys.pop(0)])

                else:
                    s += "\n".join(["- %s" % l for l in str(self[my_keys.pop(0)]).strip().split("\n")]) + "\n\n"
            else:
                s += "\n".join(["+ %s" % l for l in str(other[other_keys.pop(0)]).strip().split("\n")]) + "\n\n"
        return s


class HTCSection(HTCContents):
    end_comments = ""
    begin_comments = ""

    def __init__(self, name, begin_comments="", end_comments=""):
        self.name_ = name
        self.begin_comments = begin_comments.strip(" \t")
        self.end_comments = end_comments.strip(" \t")
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
        else:
            raise Exception("Section '%s' has not end" % section.name_)
        return section

    def line_from_line(self, lines):
        return HTCLine.from_lines(lines)

    def __str__(self, level=0):
        s = "%sbegin %s;%s\n" % ("  " * level, self.name_, (("", "\t" + self.begin_comments)
                                                            [bool(self.begin_comments.strip())]).replace("\t\n", "\n"))
        s += "".join([c.__str__(level + 1) for c in self])
        s += "%send %s;%s\n" % ("  " * level, self.name_, (("", "\t" + self.end_comments)
                                                           [self.end_comments.strip() != ""]).replace("\t\n", "\n"))
        return s

    def get_subsection_by_name(self, name, field='name'):
        lst = [s for s in self if field in s and s[field][0] == name]
        if len(lst) == 1:
            return lst[0]
        else:
            if len(lst) == 0:
                raise ValueError("subsection '%s' not found" % name)
            else:
                raise NotImplementedError()


class HTCLine(HTCContents):
    values = None
    comments = ""

    def __init__(self, name, values, comments):
        if "__" in name:
            name = name[:name.index("__")]
        self.name_ = name
        self.values = list(values)
        self.comments = comments.strip(" \t")

    def __repr__(self):
        return str(self)

    def __str__(self, level=0):
        if self.name_ == "":
            return ""
        return "%s%s%s;%s\n" % ("  " * (level), self.name_,
                                ("", "\t" + self.str_values())[bool(self.values)],
                                ("", "\t" + self.comments)[bool(self.comments.strip())])

    def str_values(self):
        return " ".join([str(v).lower() for v in self.values])

    def __getitem__(self, key):
        try:
            return self.values[key]
        except:
            raise IndexError("Parameter %s does not exists for %s" % (key + 1, self.location()))

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

    def compare(self, other):
        s = ""
        if self.values != other.values:
            s += "\n".join(["+ %s" % l for l in str(self).strip().split("\n")]) + "\n"
            s += "\n".join(["- %s" % l for l in str(other).strip().split("\n")]) + "\n"
            s += "\n"
        return s


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
        while len(lines) and lines[0].strip() == "":
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
        s = "%sbegin %s;%s\n" % ("  " * level, self.name_, ("", "\t" + self.begin_comments)
                                 [len(self.begin_comments.strip()) > 0])
        s += "".join([c.__str__(level + 1) for c in self])
        s += "".join([s.__str__(level + 1) for s in self.sensors])
        s += "%send %s;%s\n" % ("  " * level, self.name_, ("", "\t" + self.end_comments)
                                [self.end_comments.strip() != ""])
        return s

    def compare(self, other):
        s = HTCContents.compare(self, other)
        for s1, s2 in zip(self.sensors, other.sensors):
            s += s1.compare(s2)
        for s1 in self.sensors[len(other.sensors):]:
            s += "\n".join(["- %s" % l for l in str(s1).strip().split("\n")]) + "\n"
        for s2 in self.sensors[len(self.sensors):]:
            s += "\n".join(["- %s" % l for l in str(s2).strip().split("\n")]) + "\n"

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
        s = "%sbegin %s %s %s;%s\n" % ("  " * level, self.name_, self.type, self.time,
                                       ("", "\t" + self.begin_comments)[len(self.begin_comments.strip())])
        s += "".join([c.__str__(level + 1) for c in self])
        s += "".join([s.__str__(level + 1) for s in self.sensors])
        s += "%send %s;%s\n" % ("  " * level, self.name_, ("", "\t" + self.end_comments)
                                [self.end_comments.strip() != ""])
        return s


class HTCSensor(HTCLine):
    type = ""
    sensor = ""
    values = []

    def __init__(self, type, sensor, values, comments):
        self.type = type
        self.sensor = sensor
        self.values = values
        self.comments = comments.strip(" \t")

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
        return "%s%s %s%s;%s\n" % ("  " * (level),
                                   self.type,
                                   self.sensor,
                                   ("", "\t" + self.str_values())[bool(self.values)],
                                   ("", "\t" + self.comments)[bool(self.comments.strip())])

    def compare(self, other):
        s = ""
        if self.sensor != other.sensor or self.values != other.values:
            s += "\n".join(["+ %s" % l for l in str(self).strip().split("\n")]) + "\n"
            s += "\n".join(["- %s" % l for l in str(other).strip().split("\n")]) + "\n"
            s += "\n"
        return s
