'''
Created on 20/01/2014

@author: MMPE

See documentation of HTCFile below

'''
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from io import open
from builtins import str
from future import standard_library
from wetb.utils.process_exec import pexec
from wetb.utils.cluster_tools.cluster_resource import unix_path_old
standard_library.install_aliases()
from collections import OrderedDict

from wetb.hawc2.htc_contents import HTCContents, HTCSection, HTCLine
from wetb.hawc2.htc_extensions import HTCDefaults, HTCExtensions
import os
from copy import copy

class Input_File_Data(object):

    def __init__(self, key_in=None, original_file_in=None, file_object_in=None, written_file_in=None):

        self.key=key_in
        self.original_file=original_file_in
        self.file_object=file_object_in
        self.written_file=written_file_in

    def __getitem__(self, item_key):

        if isinstance(item_key, str):
            if item_key.isdigit() or (item_key[0] in ('-', '+') and item_key[1:].isdigit()):
                item_key = int(item_key)

        if not isinstance(item_key, int):
            if item_key == 'key':
                item_key = 0
            elif item_key == 'original_file':
                item_key = 1
            elif item_key == 'file_object':
                item_key = 2
            elif item_key == 'written_file':
                item_key = 3

        if item_key == 0:
            return self.key
        elif item_key == 1:
            return self.original_file
        elif item_key == 2:
            return self.file_object
        elif item_key == 3:
            return self.written_file
        else:
            raise KeyError('That key does not exist')

    def __setitem__(self, item_key, item_val):

        if isinstance(item_key, str):
            if item_key.isdigit() or (item_key[0] in ('-', '+') and item_key[1:].isdigit()):
                item_key = int(item_key)

        if not isinstance(item_key, int):
            if item_key == 'key':
                item_key = 0
            elif item_key == 'original_file':
                item_key = 1
            elif item_key == 'file_object':
                item_key = 2
            elif item_key == 'written_file':
                item_key = 3

        if item_key == 0:
            self.key = item_val
        elif item_key == 1:
            self.original_file = item_val
        elif item_key == 2:
            self.file_object = item_val
        elif item_key == 3:
            self.written_file = item_val
        else:
            raise KeyError('That key does not exist')

    def __str__(self):
        if self.key is None and self.original_file is None and self.file_object is None and self.written_file is None:
            return 'Input_File_Data()'
        retval = 'Input_File_Data('
        has_arg = False
        if not self.key is None:
            retval += 'key_in = '+str(self.key)
            has_arg = True
        if not self.original_file is None:
            if has_arg:
                retval += ', '
            retval += 'original_file_in = '+str(self.original_file)
            has_arg = True
        if not self.file_object is None:
            if has_arg:
                retval += ', '
            retval += 'file_object_in = '+str(self.file_object)
            has_arg = True
        if not self.written_file is None:
            if has_arg:
                retval += ', '
            retval += 'written_file_in = '+str(self.written_file)
            has_arg = True
        retval += ')'

        return retval

class HAWC2_Input_Files(dict):

    def __init__(self):
        super(HAWC2_Input_Files, self).__init__()

    def _get_leaves(self, container):

        retval = []
        if isinstance(container,Input_File_Data):
            return [container]
        elif isinstance(container, list):
            for obj in container:
                retval.extend(self._get_leaves(obj))
        elif isinstance(container, dict):
            for obj in container.values():
                retval.extend(self._get_leaves(obj))
        else:
            raise Exception('That container type has not been implemented')

        return retval

    def get_all_extra_inputs(self):

        retval = []
        for key, container in self.items():
            if key != 'htc':
                retval.extend(self._get_leaves(container))
        return retval

class HTCFile(HTCContents, HTCDefaults, HTCExtensions):
    """Wrapper for HTC files

    Examples:
    ---------
    >>> htcfile = HTCFile('htc/test.htc')
    >>> htcfile.wind.wsp = 10
    >>> htcfile.save()
    
    #---------------------------------------------
    >>> htc = HTCFile(filename=None, modelpath=None) # create minimal htcfile
        
    #Add section
    >>> htc.add_section('hydro')
    
    #Add subsection
    >>> htc.hydro.add_section("hydro_element")
    
    #Set values
    >>> htc.hydro.hydro_element.wave_breaking = [2, 6.28, 1] # or
    >>> htc.hydro.hydro_element.wave_breaking = 2, 6.28, 1
    
    #Set comments
    >>> htc.hydro.hydro_element.wave_breaking.comments = "This is a comment"
    
    #Access section
    >>> hydro_element = htc.hydro.hydro_element #or
    >>> hydro_element = htc['hydro.hydro_element'] # or
    >>> hydro_element = htc['hydro/hydro_element'] # or
    >>> print (hydro_element.wave_breaking) #string represenation
    wave_breaking    2 6.28 1;    This is a comment
    >>> print (hydro_element.wave_breaking.name_) # command
    wave_breaking
    >>> print (hydro_element.wave_breaking.values) # values
    [2, 6.28, 1
    >>> print (hydro_element.wave_breaking.comments) # comments
    This is a comment
    >>> print (hydro_element.wave_breaking[0]) # first value
    2
    
    #Delete element
    del htc.simulation.logfile #Delete logfile line. Raise keyerror if not exists
    """

    filename = None
    htc_inputfiles = []
    level = 0
    modelpath = "../"
    initial_comments = None
    _contents = None

    def __init__(self, filename=None, modelpath=None):
        """        
        Parameters
        ---------
        filename : str
            Absolute filename of htc file
        modelpath : str
            Model path relative to htc file 
        """
        
        if filename is not None:
            self.filename = filename
        self.modelpath = modelpath or self.auto_detect_modelpath()
        if filename and self.modelpath!="unknown" and not os.path.isabs(self.modelpath):
            self.modelpath = os.path.realpath(os.path.join(os.path.dirname(self.filename), self.modelpath))
        
    
                #assert 'simulation' in self.contents, "%s could not be loaded. 'simulation' section missing" % filename

    def auto_detect_modelpath(self):
        if self.filename is None:
            return "../"
        
        #print (["../"*i for i in range(3)])
        import numpy as np
        input_files = HTCFile(self.filename, 'unknown').input_files()
        rel_input_files = [f for f in input_files if not os.path.isabs(f)] 
        found = ([np.sum([os.path.isfile(os.path.join(os.path.dirname(self.filename), "../"*i, f)) for f in rel_input_files]) for i in range(4)])
        #for f in self.input_files():
        #    print (os.path.isfile(os.path.join(os.path.dirname(self.filename), "../",f)), f)
        if max(found)>0:
            relpath = "../"* np.argmax(found)
            return os.path.abspath(os.path.join(os.path.dirname(self.filename), relpath))
        else:
            raise ValueError("Modelpath cannot be autodetected for '%s'.\nInput files not found near htc file"%self.filename)
        
    def _load(self):
        self.reset()
        self.initial_comments = []
        self.htc_inputfiles = []
        self.contents = OrderedDict()
        if self.filename is None:
            lines = self.empty_htc.split("\n")
        else:
            lines = self.readlines(self.filename)

        lines = [l.strip() for l in lines]

        #lines = copy(self.lines)
        while lines:
            if lines[0].startswith(";"):
                self.initial_comments.append(lines.pop(0).strip() + "\n")
            elif lines[0].lower().startswith("begin"):
                self._add_contents(HTCSection.from_lines(lines))
            else:
                line = HTCLine.from_lines(lines)
                if line.name_ == "exit":
                    break
                self._add_contents(line)

    def __delitem__(self, item):
        HTCContents.__delitem__(self, item)

    def reset(self):
        self._contents = None

        
    @property
    def contents(self):
        if self._contents is None:
            self._load()
        return self._contents
    
    @contents.setter
    def contents(self, value):
        self._contents = value
        
            
    def readfilelines(self, filename):
        with open(unix_path_old(filename), encoding='cp1252') as fid:
            lines = list(fid.readlines())
        if lines[0].encode().startswith(b'\xc3\xaf\xc2\xbb\xc2\xbf'):
            lines[0] = lines[0][3:]
        return lines

    def readlines(self, filename):
        self.htc_inputfiles.append(filename)
        htc_lines = []
        lines = self.readfilelines(filename)
        for l in lines:
            if l.lower().lstrip().startswith('continue_in_file'):
                filename = l.lstrip().split(";")[0][len("continue_in_file"):].strip().lower()
                
                if self.modelpath=='unknown':
                    self.htc_inputfiles.append(filename)
                else:
                    filename = os.path.join(self.modelpath, filename)
                    for line in self.readlines(filename):
                        if line.lstrip().lower().startswith('exit'):
                            break
                        htc_lines.append(line)
            else:
                htc_lines.append(l)
        return htc_lines


    def __setitem__(self, key, value):
        self.contents # load if not loaded
        HTCContents.__setitem__(self, key, value)
        #self.contents[key] = value

    def __str__(self):
        self.contents #load
        retval = ""
        for comment in self.initial_comments:
            retval+=comment
        #import pdb; pdb.set_trace()
        for c in self:
            retval+=c.__str__(0)
        retval+="exit;"
        return retval

    def save(self, filename=None):
        self.contents #load if not loaded
        if filename is None:
            filename = self.filename
        else:
            self.filename = filename
        # exist_ok does not exist in Python27
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))  #, exist_ok=True)
        with open(filename, 'w', encoding='cp1252') as fid:
            fid.write(str(self))

    def set_name(self, name, subfolder=''):
        #if os.path.isabs(folder) is False and os.path.relpath(folder).startswith("htc" + os.path.sep):
        self.contents #load if not loaded
        fmt_folder = lambda folder, subfolder : "./" + os.path.relpath(os.path.join(folder, subfolder)).replace("\\", "/")
        
        self.filename = os.path.abspath(os.path.join(self.modelpath, fmt_folder('htc', subfolder), "%s.htc" % name)).replace("\\", "/")
        if 'simulation' in self and 'logfile' in self.simulation:
            self.simulation.logfile = os.path.join(fmt_folder('log', subfolder), "%s.log" % name).replace("\\", "/")
            if 'animation' in self.simulation:
                self.simulation.animation = os.path.join(fmt_folder('animation', subfolder), "%s.dat" % name).replace("\\", "/")
            if 'visualization' in self.simulation:
                self.simulation.visualization = os.path.join(fmt_folder('visualization', subfolder), "%s.hdf5" % name).replace("\\", "/")
        elif 'test_structure' in self and 'logfile' in self.test_structure:  # hawc2aero
            self.test_structure.logfile = os.path.join(fmt_folder('log', subfolder), "%s.log" % name).replace("\\", "/")
        self.output.filename = os.path.join(fmt_folder('res', subfolder), "%s" % name).replace("\\", "/")

    def set_time(self, start=None, stop=None, step=None):
        self.contents # load if not loaded
        if stop is not None:
            self.simulation.time_stop = stop
        else:
            stop = self.simulation.time_stop[0]
        if step is not None:
            self.simulation.newmark.deltat = step
        if start is not None:
            self.output.time = start, stop
            if "wind" in self:# and self.wind.turb_format[0] > 0:
                self.wind.scale_time_start = start

    def get_input_files_and_keys(self):

        retval = HAWC2_Input_Files()

        # Add the htc files
        retval['htc'] = []
        for htc in self.htc_inputfiles:
            retval['htc'].append(Input_File_Data(key_in = None, original_file_in = os.path.abspath(os.path.realpath(htc)), file_object_in = self, written_file_in = None))

        # now lets collect all the st files
        if 'new_htc_structure' in self:
            for mb_key in self.new_htc_structure.keys():
                full_mb_key = 'new_htc_structure.'+mb_key
                if mb_key.startswith('main_body'):
                    if 'timoschenko_input' in self[full_mb_key].keys():
                        if not 'st' in retval:
                            retval['st'] = {}
                        st_key = 'new_htc_structure.'+mb_key+'.timoschenko_input.filename.0'
                        retval['st'][mb_key]=Input_File_Data(key_in=st_key, original_file_in=os.path.abspath(os.path.realpath(self[st_key])), file_object_in = None, written_file_in = None)
                    if 'external_bladedata_dll' in self[full_mb_key].keys():
                        if not 'st_external_bladedata_dll' in retval:
                            retval['st_external_bladedata_dll'] = {}
                        stdll_key = 'new_htc_structure.'+mb_key+'.external_bladedata_dll.2'
                        retval['st_external_bladedata_dll'][mb_key]=Input_File_Data(key_in=stdll_key, original_file_in=os.path.abspath(os.path.realpath(self[stdll_key])), file_object_in = None, written_file_in = None)
        if 'aero' in self:
            ae_key = 'aero.ae_filename.0'
            pc_key = 'aero.pc_filename.0'
            retval['ae']=Input_File_Data(key_in=ae_key, original_file_in=os.path.abspath(os.path.realpath(self[ae_key])), file_object_in = None, written_file_in = None)
            retval['pc']=Input_File_Data(key_in=pc_key, original_file_in=os.path.abspath(os.path.realpath(self[pc_key])), file_object_in = None, written_file_in = None)
            if 'external_bladedata_dll' in self['aero'].keys():
                file_key = 'aero.external_bladedata_dll.2'
                retval['ae_pc_external_bladedata_dll']=Input_File_Data(key_in=file_key, original_file_in=os.path.abspath(os.path.realpath(self[file_key])), file_object_in = None, written_file_in = None)
            if 'output_profile_coef_filename' in self['aero'].keys():
                file_key = 'aero.output_profile_coef_filename.0'
                retval['output_profile_coef_filename']=Input_File_Data(key_in=file_key, original_file_in=os.path.abspath(os.path.realpath(self[file_key])), file_object_in = None, written_file_in = None)
            if 'dynstall_ateflap' in self.aero:
                file_key = 'aero.dynstall_ateflap.flap.2'
                retval['dynstall_ateflap']=Input_File_Data(key_in=file_key, original_file_in=os.path.abspath(os.path.realpath(self[file_key])), file_object_in = None, written_file_in = None)
            if 'bemwake_method' in self.aero:
                file_key = 'aero.bemwake_method.a-ct-filename.0'
                retval['a-ct-filename']=Input_File_Data(key_in=file_key, original_file_in=os.path.abspath(os.path.realpath(self[file_key])), file_object_in = None, written_file_in = None)
        if 'dll' in self:
            for dll_key in self['dll'].keys():
                if dll_key != '__on__':
                    if 'dll' not in retval:
                        retval['dll']={}
                    dll_file_key = 'dll.'+dll_key+'.filename.0'
                    retval['dll'][dll_key]=Input_File_Data(key_in = dll_file_key, original_file_in = os.path.abspath(os.path.realpath(self[dll_file_key])), file_object_in = None, written_file_in = None)
        if 'wind' in self:
            if 'user_defined_shear' in self['wind']:
                file_key = 'wind.user_defined_shear.0'
                retval['user_defined_shear']=Input_File_Data(key_in = file_key, original_file_in = os.path.abspath(os.path.realpath(self[file_key])), file_object_in = None, written_file_in = None)
            if 'user_defined_shear_turbulence' in self['wind']:
                file_key = 'wind.user_defined_shear_turbulence.0'
                retval['user_defined_shear_turbulence']=Input_File_Data(key_in = file_key, original_file_in = os.path.abspath(os.path.realpath(self[file_key])), file_object_in = None, written_file_in = None)
        if 'wakes' in self:
            if 'use_specific_deficit_file' in self['wakes']:
                file_key = 'wakes.use_specific_deficit_file.0'
                retval['use_specific_deficit_file']=Input_File_Data(key_in = file_key, original_file_in = os.path.abspath(os.path.realpath(self[file_key])), file_object_in = None, written_file_in = None)
            if 'write_ct_cq_file' in self['wakes']:
                file_key = 'wakes.write_ct_cq_file.0'
                retval['write_ct_cq_file']=Input_File_Data(key_in = file_key, original_file_in = os.path.abspath(os.path.realpath(self[file_key])), file_object_in = None, written_file_in = None)
            if 'write_final_deficits' in self['wakes']:
                file_key = 'wakes.write_final_deficits.0'
                retval['write_final_deficits']=Input_File_Data(key_in = file_key, original_file_in = os.path.abspath(os.path.realpath(self[file_key])), file_object_in = None, written_file_in = None)
        if 'hydro' in self:
            if 'water_properties' in self['hydro'].keys():
                file_key = 'hydro.water_properties.0'
                retval['water_properties']=Input_File_Data(key_in = file_key, original_file_in = os.path.abspath(os.path.realpath(self[file_key])), file_object_in = None, written_file_in = None)
        if 'soil' in self:
            if 'soil_element' in self['soil'].keys():
                file_key = 'soil.soil_element.datafile.0'
                retval['soil_element_datafile']=Input_File_Data(key_in = file_key, original_file_in = os.path.abspath(os.path.realpath(self[file_key])), file_object_in = None, written_file_in = None)
        if 'force' in self:
            if 'dll' in self['force'].keys():
                if 'dll' in self['force.dll'].keys():
                    file_key = 'force.dll.dll.0'
                    retval['soil_element_datafile']=Input_File_Data(key_in = file_key, original_file_in = os.path.abspath(os.path.realpath(self[file_key])), file_object_in = None, written_file_in = None)

        return retval

    def input_files(self):
        self.contents # load if not loaded
        if self.modelpath=="unknown":
            files = [f.replace("\\","/") for f in self.htc_inputfiles]
        else:
            files = [os.path.abspath(f).replace("\\","/") for f in self.htc_inputfiles]
        if 'new_htc_structure' in self:
            for mb in [self.new_htc_structure[mb] for mb in self.new_htc_structure.keys() if mb.startswith('main_body')]:
                if "timoschenko_input" in mb:
                    files.append(mb.timoschenko_input.filename[0])
                files.append(mb.get('external_bladedata_dll', [None, None, None])[2])
        if 'aero' in self:
            files.append(self.aero.ae_filename[0])
            files.append(self.aero.pc_filename[0])
            files.append(self.aero.get('external_bladedata_dll', [None, None, None])[2])
            files.append(self.aero.get('output_profile_coef_filename', [None])[0])
            if 'dynstall_ateflap' in self.aero:
                files.append(self.aero.dynstall_ateflap.get('flap', [None] * 3)[2])
            if 'bemwake_method' in self.aero:
                files.append(self.aero.bemwake_method.get('a-ct-filename', [None] * 3)[0])
        for dll in [self.dll[dll] for dll in self.get('dll', {}).keys() if 'filename' in self.dll[dll]]:
            files.append(dll.filename[0])
        if 'wind' in self:
            files.append(self.wind.get('user_defined_shear', [None])[0])
            files.append(self.wind.get('user_defined_shear_turbulence', [None])[0])
        if 'wakes' in self:
            files.append(self.wind.get('use_specific_deficit_file', [None])[0])
            files.append(self.wind.get('write_ct_cq_file', [None])[0])
            files.append(self.wind.get('write_final_deficits', [None])[0])
        if 'hydro' in self:
            if 'water_properties' in self.hydro:
                files.append(self.hydro.water_properties.get('water_kinematics_dll', [None])[0])
                files.append(self.hydro.water_properties.get('water_kinematics_dll', [None, None])[1])
        if 'soil' in self:
            if 'soil_element' in self.soil:
                files.append(self.soil.soil_element.get('datafile', [None])[0])
        try:
            dtu_we_controller = self.dll.get_subsection_by_name('dtu_we_controller')
            theta_min = dtu_we_controller.init.constant__5[1]
            files.append(os.path.join(os.path.dirname(dtu_we_controller.filename[0]), "wpdata.%d"%theta_min).replace("\\","/"))
        except:
            pass
            
    
        try:
            files.append(self.force.dll.dll[0])
        except:
            pass

        return [f for f in set(files) if f]

    def output_files(self):
        self.contents # load if not loaded
        files = []
        for k, index in [('simulation/logfile', 0),
                         ('simulation/animation', 0),
                         ('simulation/visualization', 0),
                         ('new_htc_structure/beam_output_file_name', 0),
                         ('new_htc_structure/body_output_file_name', 0),
                         ('new_htc_structure/struct_inertia_output_file_name', 0),
                         ('new_htc_structure/body_eigenanalysis_file_name', 0),
                         ('new_htc_structure/constraint_output_file_name', 0),
                         ('wind/turb_export/filename_u', 0),
                         ('wind/turb_export/filename_v', 0),
                         ('wind/turb_export/filename_w', 0)]:
            line = self.get(k)
            if line:
                files.append(line[index])
        if 'new_htc_structure' in self:
            if 'system_eigenanalysis' in self.new_htc_structure:
                f = self.new_htc_structure.system_eigenanalysis[0]
                files.append(f)
                files.append(os.path.join(os.path.dirname(f), 'mode*.dat').replace("\\", "/"))
            if 'structure_eigenanalysis_file_name' in self.new_htc_structure:
                f = self.new_htc_structure.structure_eigenanalysis_file_name[0]
                files.append(f)
                files.append(os.path.join(os.path.dirname(f), 'mode*.dat').replace("\\", "/"))
        files.extend(self.res_file_lst())

        for key in [k for k in self.contents.keys() if k.startswith("output_at_time")]:
            files.append(self[key]['filename'][0] + ".dat")
        return [f for f in files if f]

    def turbulence_files(self):
        self.contents # load if not loaded
        if 'wind' not in self.contents.keys() or self.wind.turb_format[0] == 0:
            return []
        elif self.wind.turb_format[0] == 1:
            files = [self.get('wind.mann.filename_%s' % comp, [None])[0] for comp in ['u', 'v', 'w']]
        elif self.wind.turb_format[0] == 2:
            files = [self.get('wind.flex.filename_%s' % comp, [None])[0] for comp in ['u', 'v', 'w']]
        return [f for f in files if f]


    def res_file_lst(self):
        self.contents # load if not loaded
        res = []
        for output in [self[k] for k in self.keys() if self[k].name_=="output"]:
            dataformat = output.get('data_format', 'hawc_ascii')
            res_filename = output.filename[0]
            if dataformat[0] == "gtsdf" or dataformat[0] == "gtsdf64":
                res.append(res_filename + ".hdf5")
            elif dataformat[0] == "flex_int":
                res.extend([res_filename + ".int", os.path.join(os.path.dirname(res_filename), 'sensor')])
            else:
                res.extend([res_filename + ".sel", res_filename + ".dat"])
        return res


    def simulate(self, exe, skip_if_up_to_date=False):
        self.contents # load if not loaded
        if skip_if_up_to_date:
            from os.path import isfile, getmtime, isabs
            res_file = os.path.join(self.modelpath, self.res_file_lst()[0])
            htc_file = os.path.join(self.modelpath, self.filename)
            if isabs(exe):
                exe_file = exe
            else:
                exe_file = os.path.join(self.modelpath, exe)
            #print (from_unix(getmtime(res_file)), from_unix(getmtime(htc_file)))
            if (isfile(htc_file) and isfile(res_file) and isfile(exe_file) and
                str(HTCFile(htc_file))==str(self) and
                getmtime(res_file) > getmtime(htc_file) and getmtime(res_file) > getmtime(exe_file)):
                if "".join(self.readfilelines(htc_file)) == str(self):
                        return

        self.save()
        htcfile = os.path.relpath(self.filename, self.modelpath)
        hawc2exe = exe
        errorcode, stdout, stderr, cmd = pexec([hawc2exe, htcfile], self.modelpath)

        if "logfile" in self.simulation:
            with open(os.path.join(self.modelpath, self.simulation.logfile[0])) as fid:
                log = fid.read()
        else:
            log = stderr


        if errorcode or 'Elapsed time' not in log:
            raise Exception (str(stdout) + str(stderr))
        return str(stdout) + str(stderr), log
        
    def deltat(self):
        return self.simulation.newmark.deltat[0]
    

#     
#     def get_body(self, name):
#         lst = [b for b in self.new_htc_structure if b.name_=="main_body" and b.name[0]==name]
#         if len(lst)==1:
#             return lst[0]
#         else:
#             if len(lst)==0:
#                 raise ValueError("Body '%s' not found"%name)
#             else:
#                 raise NotImplementedError()
#         

class H2aeroHTCFile(HTCFile):
    def __init__(self, filename=None, modelpath=None):
        HTCFile.__init__(self, filename=filename, modelpath=modelpath)

    @property
    def simulation(self):
        return self.test_structure

    def set_time(self, start=None, stop=None, step=None):
        if stop is not None:
            self.test_structure.time_stop = stop
        else:
            stop = self.simulation.time_stop[0]
        if step is not None:
            self.test_structure.deltat = step
        if start is not None:
            self.output.time = start, stop
            if "wind" in self and self.wind.turb_format[0] > 0:
                self.wind.scale_time_start = start


if "__main__" == __name__:
    f = HTCFile(r"C:\mmpe\HAWC2\models\DTU10MWRef6.0\htc\DTU_10MW_RWT_power_curve.htc", "../")
    f.save(r"C:\mmpe\HAWC2\models\DTU10MWRef6.0\htc\DTU_10MW_RWT_power_curve.htc")

    f = HTCFile(r"C:\mmpe\HAWC2\models\DTU10MWRef6.0\htc\DTU_10MW_RWT.htc", "../")
    f.save(r"C:\mmpe\HAWC2\models\DTU10MWRef6.0\htc\DTU_10MW_RWT.htc")

