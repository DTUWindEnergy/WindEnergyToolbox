'''
Created on 20/01/2014

@author: MMPE

See documentation of HTCFile below

'''
from collections import OrderedDict

from wetb.hawc2.htc_contents import HTCContents, HTCSection, HTCLine, \
    HTCDefaults
import os


class HTCFile(HTCContents, HTCDefaults):

    filename = None
    htc_inputfiles = []
    level = 0
    modelpath = "../"
    initial_comments = None
    def __init__(self, filename=None, modelpath="../"):
        self.modelpath = modelpath
        self.contents = OrderedDict()
        self.initial_comments = []
        self.htc_inputfiles = []
        if filename is None:
            self.filename = 'empty.htc'
            self.lines = self.empty_htc.split("\n")
        else:
            self.filename = filename
            self.lines = self.readlines(filename)
#            with open(filename) as fid:
#                self.lines = fid.readlines()
        self.lines = [l.strip() for l in self.lines]

        lines = self.lines.copy()
        while lines:
            if lines[0].startswith(";"):
                self.initial_comments.append(lines.pop(0).strip() + "\n")
            elif lines[0].lower().startswith("begin"):
                self._add_contents(HTCSection.from_lines(lines))
            else:
                line = HTCLine.from_lines(lines)
                self._add_contents(line)
                if line.name_ == "exit":
                    break
        assert 'simulation' in self.contents, "%s could not be loaded. 'simulation' section missing" % filename

    def readlines(self, filename):
        self.htc_inputfiles.append(filename)
        htc_lines = []
        with open(filename) as fid:
            lines = fid.readlines()
        for l in lines:
            if l.lower().lstrip().startswith('continue_in_file'):
                filename = l.lstrip().split(";")[0][len("continue_in_file"):].strip()
                filename = os.path.join(os.path.dirname(self.filename), self.modelpath, filename)
                htc_lines.extend(self.readlines(filename))
            else:
                htc_lines.append(l)
        return htc_lines


    def __setitem__(self, key, value):
        self.contents[key] = value

    def __str__(self):
        return "".join(self.initial_comments + [c.__str__(1) for c in self])

    def save(self, filename=None):
        if filename is None:
            filename = self.filename
        else:
            self.filename = filename
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as fid:
            fid.write(str(self))

    def set_name(self, name, folder="htc"):
        self.filename = os.path.join(self.modelpath, folder, "%s.htc" % name).replace("\\", "/")
        self.simulation.logfile = "./log/%s.log" % name
        self.output.filename = "./res/%s" % name

    def input_files(self):
        files = self.htc_inputfiles
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
        for dll in [self.dll[dll] for dll in self.get('dll', {}).keys()]:
            files.append(dll.filename[0])
        if 'wind' in self:
            files.append(self.wind.get('user_defined_shear', [None])[0])
            files.append(self.wind.get('wind.user_defined_shear_turbulence', [None])[0])
        if 'wakes' in self:
            files.append(self.wind.get('use_specific_deficit_file', [None])[0])
            files.append(self.wind.get('write_ct_cq_file', [None])[0])
            files.append(self.wind.get('write_final_deficits', [None])[0])
        if 'hydro' in self:
            if 'water_properties' in self.hydro:
                files.append(self.hydro.water_properties.get('water_kinematics_dll', [None])[0])
        if 'soil' in self:
            if 'soil_element' in self.soil:
                files.append(self.soil.soil_element.get('datafile', [None])[0])
        if 'force' in self:
            files.append(self.force.get('dll', [None])[0])

        return [f for f in set(files) if f]

    def output_files(self):
        files = []
        for k, index in [('simulation/logfile', 0),
                         ('simulation/animation', 0),
                         ('simulation/visualization', 0),
                         ('new_htc_structure/beam_output_file_name', 0),
                         ('new_htc_structure/body_output_file_name', 0),
                         ('new_htc_structure/struct_inertia_output_file_name', 0),
                         ('new_htc_structure/body_eigenanalysis_file_name', 0),
                         ('new_htc_structure/constraint_output_file_name', 0),
                         ('new_htc_structure/structure_eigenanalysis_file_name', 0),
                         ('turb_export/filename_u', 0),
                         ('turb_export/filename_v', 0),
                         ('turb_export/filename_w', 0)]:
            line = self.get(k)
            if line:
                files.append(line[index])

        if 'system_eigenanalysis' in self.new_htc_structure:
            f = self.new_htc_structure.system_eigenanalysis[0]
            files.append(f)
            files.append(os.path.join(os.path.dirname(f), 'mode*.dat'))
        files.extend(self.res_file_lst())

        for key in [k for k in self.contents.keys() if k.startswith("output_at_time")]:
            files.append(self[key]['filename'][0] + ".dat")
        return [f for f in files if f]

    def turbulence_files(self):
        files = [self.get('wind.%s.filename_%s' % (type, comp), [None])[0] for type in ['mann', 'flex'] for comp in ['u', 'v', 'w']]
        return [f for f in files if f]


    def res_file_lst(self):
        if 'output' not in self:
            return []
        dataformat = self.output.get('data_format', 'hawc_ascii')
        res_filename = self.output.filename[0]
        if dataformat == "gtsdf" or dataformat == "gtsdf64":
            return [res_filename + ".hdf5"]
        elif dataformat == "flex_int":
            return [res_filename + ".int", os.path.join(os.path.dirname(res_filename), 'sensor')]
        else:
            return [res_filename + ".sel", res_filename + ".dat"]



if "__main__" == __name__:
    f = HTCFile(r"C:\mmpe\HAWC2\Hawc2_model\htc\NREL_5MW_reference_wind_turbine_launcher_test.htc")
    print ("\n".join(f.output_files()))


