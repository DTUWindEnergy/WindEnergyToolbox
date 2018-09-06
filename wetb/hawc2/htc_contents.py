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
import copy
import numpy as np


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

def parse_next_line_and_raw_line(lines):
    raw_line = lines.pop(0)
    _3to2list = list(raw_line.split(";"))
    line, comments, = _3to2list[:1] + [_3to2list[1:]]
    comments = ";".join(comments).rstrip()
    while lines and lines[0].lstrip().startswith(";"):
        comments += "\n%s" % lines.pop(0).rstrip()
    return line.strip(), comments, raw_line

def fmt_value(v):
    try:
        if int(float(v)) == float(v):
            return int(float(v))
        return float(v)
    except ValueError:
        return v

def is_int(v):
    try:
        int(v)
        return True
    except ValueError:
        return False

class HTCContents(object):
    lines = []
    contents = None
    name_ = ""
    on = True
    parent=None

    def __setitem__(self, key, value):
        if key == '__on__':
            self.on = bool(value)
        if not is_int(key) and isinstance(key, str):
            key = key.replace(".", "/")
            if "/" in key:
                keys = key.split('/')
                main_key = keys[0]
                rem_key = ".".join(keys[1:])
                obj = self.contents[main_key]
                obj[rem_key] = value
            else:
                self.contents[key] = value
        else:
            if isinstance(key, str):
                key = int(key)
            self.values[key] = value
        value.parent=self

    def __getitem__(self, key):
        if key == '__on__':
            return self.on
        if isinstance(key, str):
            key = key.replace(".", "/")
            if "/" in key:
                keys = key.split('/')
                sub_obj = self.contents[keys[0]]
                rem_key = '.'.join(keys[1:])
                return sub_obj[rem_key]
            else:
                return self.contents[key]
        else:
            return self.values[key]

    def __getattr__(self, *args, **kwargs):
        if args[0] in ['__members__','__methods__']:
            # fix python2 related issue. In py2, dir(self) calls __getattr__(('__members__',)), and this call must fail unhandled to work
            return object.__getattribute__(self, *args, **kwargs)
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
        if isinstance(v,HTCContents):
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
        retval = ['__on__']
        retval.extend(list(self.contents.keys()))
        return retval

    def all_keys(self):
        my_keys = self.keys()
        retval = copy.copy(my_keys)
        for key in my_keys:
            obj = self[key]
            if hasattr(obj,'all_keys'):
                obj_keys = obj.all_keys()
                for obj_k in obj_keys:
                    retval.append(key+'.'+str(obj_k))
        return retval

    def leaf_keys(self):
        my_keys = self.keys()
        retval = []
        loop_cnt = 0
        for key in my_keys:
            obj = self[key]
            if hasattr(obj,'leaf_keys'):
                obj_keys = obj.leaf_keys()
                for obj_k in obj_keys:
                    retval.append(key+'.'+str(obj_k))
            else:
                retval.append(key)
            loop_cnt += 1
        return retval

    def get_key_name_for_object(self):
        return self.name_

    def _add_contents(self, contents):
        key_for_object = contents.get_key_name_for_object()
        if not (key_for_object in self or key_for_object+'__1' in self):
            self[key_for_object] = contents
        else:
            # If this is a duplicate, replace the original as an enumerated key
            if key_for_object in self:
                self[key_for_object+'__1'] = self[key_for_object]
                del self[key_for_object]
            ending = "__2"
            while key_for_object + ending in self:
                ending = "__%d" % (1 + float("0%s" % ending.replace("__", "")))
            self[key_for_object + ending] = contents

    def add_section(self, name, allow_duplicate_section=False):
        if name in self and allow_duplicate_section is False:
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

    def __delitem__(self, item):
        del self.contents[item]

    def location(self):
        if self.parent is None:
            return os.path.basename(self.filename)
        else:
            return self.parent.location() + "/" + self.name_

class HTCSection(HTCContents):
    end_comments = ""
    begin_comments = ""
    def __init__(self, name, begin_comments="", end_comments=""):
        self.name_ = name
        self.begin_comments = begin_comments.strip(" \t")
        self.end_comments = end_comments.strip(" \t")
        self.contents = OrderedDict()
        self.on = True

    @staticmethod
    def from_lines(lines):
        line, begin_comments = parse_next_line(lines)
        name = line[6:].lower()
        if name == "output":
            section = HTCOutputSection(name, begin_comments)
        elif name.startswith("output_at_time"):
            section = HTCOutputAtTimeSection(name, begin_comments)
        elif name.startswith("c2_def"):
            section = HTCC2DefSection(name, begin_comments)
            section.from_lines(lines)
            return section
        elif name.startswith("timoschenko_input"):
            section = HTCTimoschenkoInputSection(name, begin_comments)
        elif name == "aero":
            section = HTCAeroInputSection(name, begin_comments)
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

    def get_key_name_for_object(self):
        if self.name_ in ('main_body', 'type2_dll', 'hawc2_dll') and 'name' in self:
            return self.name_+'_'+self['name.0']
        return self.name_

    def line_from_line(self, lines):
        return HTCLine.from_lines(lines)

    def __str__(self, level=0):
        if not self.on:
            return ""
        s = "%sbegin %s;%s\n" % ("  "*level, self.name_, (("", "\t" + self.begin_comments)[bool(self.begin_comments.strip())]).replace("\t\n","\n"))
        for c in self:
            s += c.__str__(level + 1)
        s += "%send %s;%s\n" % ("  "*level, self.name_, (("", "\t" + self.end_comments)[self.end_comments.strip() != ""]).replace("\t\n","\n"))
        return s
    
    def get_subsection_by_name(self, name, field='name'):
        lst = [s for s in self if field in s and s[field][0]==name]
        if len(lst)==1:
            return lst[0]
        else:
            if len(lst)==0:
                raise ValueError("subsection '%s' not found"%name)
            else:
                raise NotImplementedError()

class HTCTimoschenkoInputSection(HTCSection):
    
    st_object = None

    def __init__(self, name, begin_comments="", end_comments=""):

        super(HTCTimoschenkoInputSection, self).__init__(name, begin_comments, end_comments)
        self.st_object = None

    def __setitem__(self, key, value):
        words = key.split('.')
        if words[0] != 'st_object':
            HTCSection.__setitem__(self, key, value)
            return
        if len(words)==1:
            self.st_object = value
            return
        if self.st_object is None:
            raise KeyError('There is no st_object in the object')
        if not 'set' in self:
            raise Exception('Object is not complete')
        if len(words)!=2:
            raise KeyError('That key does not exist')
        st_object_key = '%d.%d.%s'%(self['set.0'], self['set.1'], words[1])
        self.st_object[st_object_key]=value

    def __getitem__(self, key):
        words = key.split('.')
        if words[0] in HTCSection.keys(self):
            return HTCSection.__getitem__(self, key)
        if words[0] != 'st_object':
            raise KeyError('That key does not exist')
        if len(words)==1:
            return self.st_object
        if self.st_object is None:
            raise KeyError('There is no st_object in the object')
        if not 'set' in self:
            raise Exception('Object is not complete')
        if len(words)!=2:
            raise KeyError('That key does not exist')
        st_object_key = '%d.%d.%s'%(self['set.0'], self['set.1'], words[1])
        return self.st_object[st_object_key]

    def keys(self):
        retval = HTCSection.keys(self)
        retval.append('st_object')
        return retval

    def all_keys(self):
        retval = HTCSection.all_keys(self)
        if not self.st_object is None:
            if not 'set' in self:
                raise Exception('Object is not complete')
            int_0 = self['set.0']
            int_1 = self['set.1']
            set_key = '%d.%d.'%(int_0, int_1)
            st_object_keys = self.st_object.keys()
            for st_key in st_object_keys:
                if st_key.startswith(set_key):
                    retval.append('st_object.%s'%(st_key[len(set_key):]))
        return retval

    def leaf_keys(self):
        retval = HTCSection.leaf_keys(self)
        if not self.st_object is None:
            if not 'set' in self:
                raise Exception('Object is not complete')
            set_key = '%d.%d.'%(self['set.0'], self['set.1'])
            st_object_keys = self.st_object.keys()
            for st_key in st_object_keys:
                if st_key.startswith(set_key):
                    retval.append('st_object.%s'%(st_key[len(set_key):]))
        return retval

class HTCAeroInputSection(HTCSection):
    
    ae_object = None
    pc_object = None

    def __init__(self, name, begin_comments="", end_comments=""):

        super(HTCAeroInputSection, self).__init__(name, begin_comments, end_comments)
        self.ae_object = None
        self.pc_object = None

    def __setitem__(self, key, value):
        words = key.split('.')
        if not words[0] in ('ae_object', 'pc_object'):
            HTCSection.__setitem__(self, key, value)
            return
        if len(words)==1:
            if words[0] == 'ae_object':
                self.ae_object = value
            else:
                self.pc_object = value
            return
        if words[0] == 'ae_object':
            if self.ae_object is None:
                raise KeyError('There is no ae_object in the object')
            if not 'ae_sets' in self:
                raise Exception('Object is not complete')
            if len(words)!=2:
                raise KeyError('That key does not exist')
            sub_words = words[1].split('_')
            blade_num = int(sub_words[0][5:])
            ae_set_key = 'ae_sets.%d'%(blade_num)
            ae_object_key = '%d.%s'%(self[ae_set_key], '_'.join(sub_words[1:]))
            self.ae_object[ae_object_key]=value
        else:
            if self.pc_object is None:
                raise KeyError('There is no ae_object in the object')
            if len(words)!=4:
                raise KeyError('That key does not exist')
            pc_object_key = '.'.join(words[1:])
            self.pc_object[pc_object_key] = value

    def __getitem__(self, key):
        words = key.split('.')
        if not words[0] in ('ae_object', 'pc_object'):
            return HTCSection.__getitem__(self, key)
        if len(words)==1:
            if words[0] == 'ae_object':
                return self.ae_object
            else:
                return self.pc_object
        if words[0] == 'ae_object':
            if self.ae_object is None:
                raise KeyError('There is no ae_object in the object')
            if not 'ae_sets' in self:
                raise Exception('Object is not complete')
            if len(words)!=2:
                raise KeyError('That key does not exist')
            sub_words = words[1].split('_')
            blade_num = int(sub_words[0][5:])
            ae_set_key = 'ae_sets.%d'%(blade_num-1)
            ae_object_key = '%d.%s'%(self[ae_set_key], '_'.join(sub_words[1:]))
            return self.ae_object[ae_object_key]
        else:
            if self.pc_object is None:
                raise KeyError('There is no ae_object in the object')
            if len(words)!=4:
                raise KeyError('That key does not exist')
            pc_object_key = '.'.join(words[1:])
            return self.pc_object[pc_object_key]

    def keys(self):
        retval = HTCSection.keys(self)
        retval.append('ae_object')
        retval.append('pc_object')
        return retval

    def all_keys(self):
        retval = HTCSection.all_keys(self)
        if not self.ae_object is None:
            if not 'ae_sets' in self:
                raise Exception('Object is not complete')
            ae_sets_obj = self['ae_sets']
            ae_object_keys = self.ae_object.keys()
            blade_num = 1
            # loop over the blades
            for ae_set_key in ae_sets_obj.keys():
                if ae_set_key != '__on__':
                    ae_set_val = ae_sets_obj[ae_set_key]
                    ae_set_str = str(ae_set_val)+'.'
                    blade_key = 'blade%d'%(blade_num)
                    # loop over the keys in ae object
                    for ae_object_key in ae_object_keys:
                        if ae_object_key.startswith(ae_set_str):
                            ae_object_sub_key = ae_object_key[len(ae_set_str):]
                            retval.append('ae_object.'+blade_key+'_'+ae_object_sub_key)
                    blade_num+=1
        if not self.pc_object is None:
            pc_object_keys = self.pc_object.keys()
            for pc_key in pc_object_keys:
                retval.append('pc_object.%s'%pc_key)
        return retval

    def leaf_keys(self):
        retval = HTCSection.leaf_keys(self)
        if not self.ae_object is None:
            if not 'ae_sets' in self:
                raise Exception('Object is not complete')
            ae_sets_obj = self['ae_sets']
            ae_object_keys = self.ae_object.keys()
            blade_num = 1
            # loop over the blades
            for ae_set_key in ae_sets_obj.keys():
                if ae_set_key != '__on__':
                    ae_set_val = ae_sets_obj[ae_set_key]
                    ae_set_str = str(ae_set_val)+'.'
                    blade_key = 'blade%d'%(blade_num)
                    # loop over the keys in ae object
                    for ae_object_key in ae_object_keys:
                        if ae_object_key.startswith(ae_set_str):
                            ae_object_sub_key = ae_object_key[len(ae_set_str):]
                            retval.append('ae_object.'+blade_key+'_'+ae_object_sub_key)
                    blade_num+=1
        if not self.pc_object is None:
            pc_object_keys = self.pc_object.keys()
            for pc_key in pc_object_keys:
                retval.append('pc_object.%s'%pc_key)
        return retval

class HTCLine(HTCContents):
    values = None
    comments = ""
    def __init__(self, name, values, comments):
        if "__" in name:
            name = name[:name.index("__")]
        self.name_ = name
        self.values = list(values)
        self.comments = comments.strip(" \t")
        self.on = True

    def __repr__(self):
        return str(self)

    def __str__(self, level=0):
        if self.name_ == "" or not self.on:
            return ""
        return "%s%s%s;%s\n" % ("  "*(level), self.name_,
                                  ("", "\t" + self.str_values())[bool(self.values)],
                                  ("", "\t" + self.comments)[bool(self.comments.strip())])

    def str_values(self):
        return " ".join([str(v).lower() for v in self.values])

    def __getitem__(self, key):
        if key == "__on__":
            return self.on
        if isinstance(key, str):
            if not is_int(key):
                raise KeyError('Cannot convert to an integer')
            key = int(key)
        return self.values[key]

    def __setitem__(self, key, value):
        if key == "__on__":
            self.on = bool(value)
            return
        if isinstance(key, str):
            if not is_int(key):
                raise KeyError('Cannot convert to an integer')
            key = int(key)
        self.values[key] = value

    def keys(self):
        retval = ['__on__']
        retval.extend(range(0,len(self.values)))
        return retval

    def all_keys(self):
        return self.keys()

    def leaf_keys(self):
        return self.keys()

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


class HTCC2DefSection(HTCSection):

    nsec = None
    x = None
    y = None
    z = None
    theta = None
    data = None
    on = None

    def __init__(self, name, begin_comments="", end_comments=""):
        HTCSection.__init__(self, name, begin_comments=begin_comments, end_comments=end_comments)
        self.nsec = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.theta = 0
        self.on = True

    def from_lines(self, lines):
        at_sec = 0
        while lines:
            if lines[0].lower().startswith("end"):
                line, self.end_comments = parse_next_line(lines)
                break
            else:
                line = lines.pop(0)
                line_val, comments, = line.strip().split(";")
                words = line_val.strip().split()
                if words[0].lower().startswith("nsec"):
                    self['nsec'] = int(words[1])
                else:
                    self.x[at_sec]=float(words[2])
                    self.y[at_sec]=float(words[3])
                    self.z[at_sec]=float(words[4])
                    self.theta[at_sec]=float(words[5])
                    at_sec+=1
        if at_sec != self.nsec:
            raise Exception('The number of lines in the input is not consistent with nsec')

    def __setitem__(self, key, value):

        if key == '__on__':
            self.on = bool(value)
            return

        if key == 'nsec':

            # do nothing if there is nothing to do
            if self.nsec == int(value):
                return

            # save the old value
            old_nsec = self.nsec
            self.nsec = int(value)

            # get the common size
            asgn = old_nsec
            if asgn > self.nsec:
                asgn = self.nsec

            # Resize all our data
            old_val = self.x
            self.x = np.zeros(self.nsec)
            if asgn > 0:
                self.x[:asgn] = old_val[:asgn]
            old_val = self.y
            self.y = np.zeros(self.nsec)
            if asgn > 0:
                self.y[:asgn] = old_val[:asgn]
            old_val = self.z
            self.z = np.zeros(self.nsec)
            if asgn > 0:
                self.z[:asgn] = old_val[:asgn]
            old_val = self.theta
            self.theta = np.zeros(self.nsec)
            if asgn > 0:
                self.theta[:asgn] = old_val[:asgn]
            return

        if value.size != self.nsec:
            ValueError('The value size is not consistent with the object size')

        if key == 'x':
            self.x = value

        if key == 'y':
            self.y = value

        if key == 'z':
            self.z = value

        if key == 'theta':
            self.theta = value

    def __getitem__(self, key):
        if key == '__on__':
            return self.on
        if key == 'nsec':
            return self.nsec
        if key == 'x':
            return self.x
        if key == 'y':
            return self.y
        if key == 'z':
            return self.z
        if key == 'theta':
            return self.theta

    def __iter__(self):
        return iter([self.nsec, self.x, self.y, self.z, self.theta])

    def __contains__(self, key):
        if key in self.keys():
            return True
        else:
            return False

    def __str__(self, level=0):
        if not self.on:
            return ""
        s = "%sbegin %s;%s\n"%("  "*level, self.name_, ("", "\t" + self.begin_comments)[bool(self.begin_comments.strip())])
        s += "%s  nsec %d\n"%("  "*level, self.nsec)
        for I in range(0,self.nsec):
            s += "%s  sec %d %0.17e %0.17e %0.17e %0.17e;\n"%("  "*level, I+1, self.x[I], self.y[I], self.z[I], self.theta[I])
        s += "%send %s;%s\n" % ("  "*level, self.name_, ("", "\t" + self.end_comments)[self.end_comments.strip() != ""])
        return s

    def keys(self):
        return ['__on__', 'nsec', 'x', 'y', 'z', 'theta']

    def all_keys(self):
        return self.keys()

    def leaf_keys(self):
        return self.keys()

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

    def add_sensor_from_line(self, line):
        self._add_sensor(HTCSensor.from_lines([line]))

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
        if not self.on:
            return ""
        s = "%sbegin %s;%s\n" % ("  "*level, self.name_, ("", "\t" + self.begin_comments)[len(self.begin_comments.strip()) > 0])
        s += "".join([c.__str__(level + 1) for c in self])
        s += "".join([s.__str__(level + 1) for s in self.sensors])
        s += "%send %s;%s\n" % ("  "*level, self.name_, ("", "\t" + self.end_comments)[self.end_comments.strip() != ""])
        return s

    def keys(self):
        retval = HTCSection.keys(self)
        for I, sensor in enumerate(self.sensors):
            sensor_key = 'sensor_%d_%s_%s'%(I+1, sensor.sns_type, sensor.sensor)
            retval.append(sensor_key)
        return retval

    def __getitem__(self, key):
        if key.startswith("sensor_"):
            keys = key.split('.')
            sensor_words = keys[0].split('_')
            idx = int(sensor_words[1])-1
            if len(keys)>1:
                new_key = '.'.join(keys[1:])
                return self.sensors[idx][new_key]
            else:
                return self.sensors[idx]
        else:
            return HTCSection.__getitem__(self,key)

    def __setitem__(self, key, value):
        if key.startswith("sensor_"):
            keys = key.split('.')
            sensor_words = keys[0].split('_')
            idx = int(sensor_words[1])-1
            if len(keys)>1:
                new_key = '.'.join(keys[1:])
                self.sensors[idx][new_key] = value
            else:
                self.sensors[idx] = value
        else:
            HTCSection.__setitem__(self,key,value)

class HTCOutputAtTimeSection(HTCOutputSection):
    sns_type = None
    time = None
    def __init__(self, name, begin_comments="", end_comments=""):
        if len(name.split()) < 3:
            raise ValueError('"keyword" and "time" arguments required for output_at_time command:\n%s' % name)
        name, self.sns_type, time = name.split()
        self.time = float(time)
        HTCOutputSection.__init__(self, name, begin_comments=begin_comments, end_comments=end_comments)

    def __str__(self, level=0):
        if not self.on:
            return ""
        s = "%sbegin %s %s %s;%s\n" % ("  "*level, self.name_, self.sns_type, self.time, ("", "\t" + self.begin_comments)[len(self.begin_comments.strip())])
        s += "".join([c.__str__(level + 1) for c in self])
        s += "".join([s.__str__(level + 1) for s in self.sensors])
        s += "%send %s;%s\n" % ("  "*level, self.name_, ("", "\t" + self.end_comments)[self.end_comments.strip() != ""])
        return s

    def line_from_line(self, lines):
        while lines[0].strip() == "":
            lines.pop(0)
        name = lines[0].split()[0].strip()
        if name in ['filename', 'data_format', 'buffer', 'time']:
            return HTCLine.from_lines(lines)
        else:
            return HTCSensorAtTime.from_lines(lines, self.sns_type)

    def keys(self):
        retval=['sensor_type']
        retval.extend(HTCOutputSection.keys(self))
        return retval

    def __getitem__(self, key):
        if key == 'sensor_type':
            return self.sns_type
        else:
            return HTCOutputSection.__getitem__(self,key)

    def __setitem__(self, key, value):
        if key == 'sensor_type':
            self.sns_type = value
        else:
            HTCOutputSection.__setitem__(self,key,value)

def get_sensor_scale_parameter(sensor_type, sensor):

    # 1 induc_sector_ct
    # 1 induc_sector_cq
    # 1 induc_sector_a
    # 1 induc_sector_am
    # 1 induc_am_norm
    # 2 vrel
    # 2 alfa
    # 2 alfadot
    # 2 sideslip
    # 2 cl
    # 2 cd
    # 2 cm
    # 2 lift
    # 2 drag
    # 2 moment
    # 2 tors_e
    # 2 inflow_angle
    # 2 dcldalfa
    # 2 dcddalfa
    # 2 gamma
    # 2 windspeed_boom
    # 2 actuatordiskload
    # 3 rotation
    # 3 secforce
    # 3 secmoment
    # 4 int_force
    # 4 int_moment
    # 4 position
    # 4 velocity
    # 4 acceleration
    # 4 windspeed
    # 4 induc
    
    if sensor_type == 'aero':
        if sensor in ('induc_sector_ct', 'induc_sector_cq', 'induc_sector_a', 'induc_sector_am', 'induc_am_norm'):
            return 1
        elif sensor in ('vrel', 'alfa', 'alfadot', 'sideslip', 'cl', 'cd', 'cm', 'lift', 'drag', 'moment', 'tors_e', 'inflow_angle', 'dcldalfa', 'dcddalfa', 'gamma', 'windspeed_boom', 'actuatordiskload'):
            return 2
        elif sensor in ('rotation', 'secforce', 'secmoment'):
            return 3
        elif sensor in ('int_force', 'int_moment', 'position', 'velocity', 'acceleration', 'windspeed', 'induc'):
            return 4
    return None

class HTCSensor(HTCLine):
    sns_type = ""
    sensor = ""
    values = []
    is_normalized = False
    raw_line = None

    def __init__(self, sns_type, sensor, values, comments):
        self.sns_type = sns_type
        self.sensor = sensor
        self.values = values
        self.comments = comments.strip(" \t")
        self.is_normalized = False

    # This will take the radius an normalize it
    def normalize_sensor(self, scale=None):
        if self.is_normalized:
            return
        self.is_normalized = True
        if scale is None:
            param = get_sensor_scale_parameter(self.sns_type, self.sensor)
            if param is None:
                return
            self.values[param]/=scale

    # This will take the radius an normalize it
    def scale_sensor(self, scale=None):
        if self.is_normalized:
            return
        self.is_normalized = True
        if scale is None:
            param = get_sensor_scale_parameter(self.sns_type, self.sensor)
            if param is None:
                return
            self.values[param]*=scale

    @staticmethod
    def from_lines(lines):
        line, comments, raw_line = parse_next_line_and_raw_line(lines)
        raw_line = raw_line.strip()
        if len(line.split()) > 2:
            _3to2list5 = list(line.split())
            sns_type, sensor, values, = _3to2list5[:2] + [_3to2list5[2:]]
        elif len(line.split()) == 2:
            sns_type, sensor = line.split()
            values = []
        else:
            sns_type, sensor, values = "", "", []
        def fmt(v):
            try:
                if int(float(v)) == float(v):
                    return int(float(v))
                return float(v)
            except ValueError:
                return v
        values = [fmt(v) for v in values]
        retval = HTCSensor(sns_type, sensor, values, comments)
        retval.raw_line = raw_line
        return retval

    def __str__(self, level=0):
        if not self.on:
            return ""
        return "%s%s %s%s;%s\n" % ("  "*(level),
                                self.sns_type,
                                self.sensor,
                                ("", "\t" + self.str_values())[bool(self.values)],
                                ("", "\t" + self.comments)[bool(self.comments.strip())])

    def keys(self):
        retval = ['__on__', 'sensor_type', 'sensor']
        retval.extend(range(0,len(self.values)))
        return retval

    def __getitem__(self, key):
        if key.startswith("sensor"):
            if key=='sensor_type':
                return self.sns_type
            elif key=='sensor':
                return self.sensor
            else:
                raise KeyError('That key does not exist')
        else:
            return HTCLine.__getitem__(self,key)

    def __setitem__(self, key, value):
        if key.startswith("sensor"):
            if key=='sensor_type':
                self.sns_type = value
            elif key=='sensor':
                self.sensor = value
            else:
                raise KeyError('That key does not exist')
        else:
            return HTCLine.__setitem__(self,key,value)

class HTCSensorAtTime(HTCSensor):

    def __init__(self, sns_type, sensor, values, comments):
        HTCSensor.__init__(self, sns_type, sensor, values, comments)

    def __str__(self, level=0):
        if not self.on:
            return ""
        return "%s%s%s;%s\n" % ("  "*(level),
                                self.sensor,
                                ("", "\t" + self.str_values())[bool(self.values)],
                                ("", "\t" + self.comments)[bool(self.comments.strip())])

    def keys(self):
        retval = ['__on__', 'sensor']
        retval.extend(range(0,len(self.values)))
        return retval

    @staticmethod
    def from_lines(lines, sns_type):
        line, comments, raw_line = parse_next_line_and_raw_line(lines)
        if len(line.split()) > 1:
            _3to2list5 = list(line.split())
            sensor, values, = _3to2list5[:1] + [_3to2list5[1:]]
        elif len(line.split()) == 1:
            sensor = line.strip()
            values = []
        else:
            sensor, values = "", "", []
        def fmt(v):
            try:
                if int(float(v)) == float(v):
                    return int(float(v))
                return float(v)
            except ValueError:
                return v
        values = [fmt(v) for v in values]
        retval = HTCSensorAtTime(sns_type, sensor, values, comments)
        retval.raw_line = raw_line
        return retval

    def __getitem__(self, key):
        if key.startswith("sensor"):
            if key=='sensor':
                return self.sensor
            else:
                raise KeyError('That key does not exist')
        else:
            return HTCLine.__getitem__(self,key)

    def __setitem__(self, key, value):
        if key.startswith("sensor"):
            if key=='sensor':
                self.sensor = value
            else:
                raise KeyError('That key does not exist')
        else:
            return HTCLine.__setitem__(self,key,value)

