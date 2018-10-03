
import os
import shutil
import time
import math
from wetb.hawc2.htc_file import HTCFile
from wetb.hawc2.ae_file import AEFile
from wetb.hawc2.pc_file import PCData
from wetb.hawc2.st_file import StOrigData, StFPMData
from wetb.hawc2.Hawc2io import ReadHawc2
try:
    import subprocess32 as subprocess
    _with_timeout = True
except:
    import subprocess
    _with_timeout = False

def mkdir_for_file_in_dir(filename):
    # prepare the search
    mkdir_list = []
    dirname = os.path.dirname(filename)
    # detect at what level we have a directory
    while not os.path.exists(dirname):
        dirname, mkdir_file = os.path.split(dirname)
        mkdir_list.insert(0, mkdir_file)
    # Make all the directories
    for mkdir_file in mkdir_list:
        dirname = os.path.join(dirname, mkdir_file)
        os.mkdir(dirname)

def open_file_in_dir(filename, mode):
    # mkdir the path
    mkdir_for_file_in_dir(filename)
    # return an open file descriptor in that folder
    return open(filename, mode)

class Hawc2_Simulation(object):

    def __init__(self, source_file_in = None):
        super(Hawc2_Simulation, self).__init__()

        # The htc file data structure
        self.htcf = None
        # this data structure contains all the input files that need to be maintained
        self.input_files = None
        # The file for the source
        self.source_file = source_file_in
        self.source_name = None
        # The directories for the source and the destination
        self.source_directory = None
        self.write_directory = None
        # indicates that the sensors in the source htc file are normalized
        self.source_sensors_normalized = False
        # indicates that the object should keep the sensors normalized
        self.object_sensors_normalized = False
        # The results of the simulation
        self.results = None

        # These are the execution settings
        # This is the command that launches hawc2
        self.exec_command = 'wine hawc2mb.exe'
        # how many times should we re-try
        self.exec_retry_count = 3
        # how long should we wait to start the first re-try
        self.exec_sleep_time = 10
        # Is this just a dry run
        self.exec_dry_run = False
        # Indicates that the simulation is a success
        self.exec_success = False
        # Run the simulation within try-except (hides error messages)
        self.exec_use_try_except = True
        # Run the simulation with a time-out ... useful for detecting errors
        self.exec_use_time_out = _with_timeout
        # The time-out used in conjunction with using a time-out
        self.exec_time_out = 1200
        # Send messages
        self.exec_verbose = True

        # This is the reference curve
        self.blade_length_x_key = 'new_htc_structure.main_body_blade1.c2_def.x'
        self.blade_length_y_key = 'new_htc_structure.main_body_blade1.c2_def.x'
        self.blade_length_z_key = 'new_htc_structure.main_body_blade1.c2_def.x'

        if not self.source_file is None:
            self.load_simulation()

    def calculate_blade_scale(self):
        retval = 1.0
        if not self.htcf is None:
            all_keys = self.htcf.all_keys()
            if self.blade_length_x_key in all_keys and self.blade_length_y_key in all_keys and self.blade_length_z_key in all_keys:
                x = self.htcf[self.blade_length_x_key]
                y = self.htcf[self.blade_length_y_key]
                z = self.htcf[self.blade_length_z_key]
                if x.size!=y.size or y.size!=z.size:
                    raise Exception('The size of the blade axis data is not consistent')
                retval = 0.0
                for I in range(1,x.size):
                    retval += math.sqrt( (x[I]-x[I-1])**2.0 + (y[I]-y[I-1])**2.0 + (z[I]-z[I-1])**2.0)
        return retval

    # Specifies that all sensors normalization settings
    def set_sensor_normalization_scheme(self, source_sensors_normalized, object_sensors_normalized):
        self.source_sensors_normalized = source_sensors_normalized
        self.object_sensors_normalized = object_sensors_normalized
        # Need to make sure that all the sensors are set in a normalized state
        if self.source_sensors_normalized:
            self.normalize_sensors(1.0)
        # Check if we need to normalize
        if self.source_sensors_normalized != self.object_sensors_normalized:
            # source is scaled, but we need to normalize
            if self.object_sensors_normalized:
                self.normalize_sensors()
            # source is normalizaed by we need to scale
            else:
                self.scale_sensors()

    def normalize_sensors(self, blade_scale = None):
        if not self.htcf is None:
            if blade_scale is None:
                blade_scale = self.calculate_blade_scale()
            for key in self.htcf.all_keys():
                obj = self.htcf[key]
                if isinstance(obj, HTCSensor) and not isinstance(obj, HTCSensorAtTime):
                    obj.normalize_sensor(blade_scale)

    def scale_sensors(self, blade_scale = None):
        if not self.htcf is None:
            if blade_scale is None:
                blade_scale = self.calculate_blade_scale()
            for key in self.htcf.all_keys():
                obj = self.htcf[key]
                if isinstance(obj, HTCSensor) and not isinstance(obj, HTCSensorAtTime):
                    obj.scale_sensors(blade_scale)

    # This is meant to delete the ersults from an old out-of-date calculation
    def delete_output_files(self):
        print('delete_output_files')

    # This is meant to write the input files
    def write_input_files(self, write_directory = None, force_write = False):

        if not write_directory is None:
            self.write_directory = os.path.abspath(os.path.realpath(write_directory))

        if self.write_directory is None:
            raise Exception('The write directory has not been specified')

        if self.write_directory == self.source_directory and not force_write:
            raise Exception('Cannot write the contents into the source directory without the write being forced')

        # Get the directory
        old_home = os.getcwd()

        # Change to the htc directory
        os.chdir(self.write_directory)

        # scale sensors for writing
        if self.object_sensors_normalized:
            self.scale_sensors()

        # Start to write every thing
        # First we need to write and rename the extra files
        extra_files = self.input_files.get_all_extra_inputs()
        for file_obj in extra_files:
            if file_obj.original_file.startswith(self.source_directory):
                if self.source_directory != self.write_directory:
                    file_obj.written_file = file_obj.original_file.replace(self.source_directory, self.write_directory)
                    self.htcf[file_obj.key] = file_obj.written_file
                else:
                    file_obj.written_file = file_obj.original_file
                if not file_obj.file_object is None:
                    fout = open_file_in_dir(file_obj.written_file, 'w')
                    fout.write(str(file_obj.file_object))
                    fout.close()
                elif file_obj.original_file != file_obj.written_file and not os.path.exists(file_obj.written_file):
                    is_good = False
                    try:
                        mkdir_for_file_in_dir(file_obj.written_file)
                        os.symlink(file_obj.original_file, file_obj.written_file)
                        is_good = True
                    except:
                        pass
                    if not is_good:
                        mkdir_for_file_in_dir(file_obj.written_file)
                        shutil.copyfile(file_obj.original_file, file_obj.written_file)
                        is_good = True
            else:
                file_obj.written_file = file_obj.original_file
        # Now write the HTC file
        self.input_files['htc'][0].written_file = os.path.join(self.write_directory, self.source_name)
        fout = open_file_in_dir(self.input_files['htc'][0].written_file, 'w')
        fout.write(str(self.htcf))
        fout.close()

        # normalize sensors
        if self.object_sensors_normalized:
            self.normalize_sensor()

        # return to the old directory
        os.chdir(old_home)

    # This will set the write directory
    def set_write_directory(self, write_directory):
        self.write_directory = os.path.abspath(os.path.realpath(write_directory))

    # Tests whether the input for the current state has already been written
    def does_input_exist(self):
        print('does_input_exist called')
        return False

    # This will execute the simulation
    def run_simulation(self):

        # Get the directory
        old_home = os.getcwd()

        # Change to the htc directory
        os.chdir(self.write_directory)

        # This is the point where you would specify that a file is running
        print('Now the code would execute the model')
        # These values allow for some second attempts at running HAWC2 (occasionally it crashed right away with wine)
        exec_str = self.exec_command+' '+self.input_files['htc'][0].written_file
        exec_str = exec_str.split()
        if not self.exec_dry_run:
            self.exec_success = False
            try_count = 0
            if self.exec_use_try_except:
                while not self.exec_success and try_count<self.exec_retry_count:
                    try_count+=1
                    try:
                        if self.exec_use_time_out:
                            proc = subprocess.check_output(exec_str, timeout=self.exec_time_out)
                        else:
                            proc = subprocess.check_output(exec_str)
                        self.exec_success = True
                        if self.exec_verbose:
                            print(exec_str, 'output:')
                            print(proc)
                    except:
                        # wait a little and try again
                        if try_count<self.exec_retry_count:
                            print(exec_str, ' crashed for case __ ... About to execute another attempt')
                            time.sleep(self.exec_sleep_time)
                        else:
                            print(exec_str, ' crashed for case __ for the final time, it appears something fundamental is wrong')
            else:
                if self.exec_use_time_out:
                    proc = subprocess.check_output(exec_str, timeout=self.exec_time_out)
                else:
                    proc = subprocess.check_output(exec_str)
                self.exec_success = True
                if self.exec_verbose:
                    print(exec_str, 'output:')
                    print(proc)
        else:
            print(exec_str + ' dry run...')
            self.exec_success = True

        # return to the old directory
        os.chdir(old_home)

    # This will load a simulation from source
    def load_simulation(self, source_file_in=None):

        if not source_file_in is None:
            self.source_file=source_file_in

        # Get the directory
        old_home = os.getcwd()

        # Get the source directory
        dirname = os.path.dirname(self.source_file)
        self.source_directory = os.path.abspath(os.path.realpath(dirname))

        # Change to the htc directory
        os.chdir(self.source_directory)

        # load in the contents
        # Load in the HTC file
        self.source_name = os.path.basename(self.source_file)
        self.htcf = HTCFile(self.source_name)
        self.set_sensor_normalization_scheme(self.source_sensors_normalized, self.object_sensors_normalized)

        # load in the other files linked in the HTC file
        self.input_files = self.htcf.get_input_files_and_keys()
        if 'st' in self.input_files:
            for body_key, input_data in self.input_files['st'].items():
                timo_input = self.htcf[input_data.key[:-11]]
                if 'FPM' in timo_input and timo_input['FPM.0']==1:
                    input_data.file_object = StFPMData(input_data.original_file)
                else:
                    input_data.file_object = StOrigData(input_data.original_file)
                timo_input['st_object'] = input_data.file_object
        if 'ae' in self.input_files:
            self.input_files['ae'].file_object = AEFile(self.input_files['ae'].original_file)
            self.htcf['aero.ae_object'] = self.input_files['ae'].file_object
        if 'pc' in self.input_files:
            self.input_files['pc'].file_object = PCData(self.input_files['pc'].original_file)
            self.htcf['aero.pc_object'] = self.input_files['pc'].file_object

        # return to the old directory
        os.chdir(old_home)
    
    def load_results(self):

        # Get the directory
        old_home = os.getcwd()

        if self.htcf is None:
            raise Exception('There is no simulation to load results from')

        # Change to the htc directory
        os.chdir(self.write_directory)

        # load the results
        file_name = self.htcf['output.filename.0']
        self.results = ReadHawc2(file_name)
        self.results.load_data()
        # Add the keys
        sensor_list = self.htcf['output'].sensors
        key_list = []
        chcnt = 0
        for sensor in sensor_list:
            if sensor['__on__']:
                sensor_size = sensor.get_sensor_size()
                chcnt+=sensor_size
                if sensor_size==1:
                    key_list.append(sensor.raw_line)
                else:
                    for I in range(0,sensor_size):
                        key_list.append(sensor.raw_line+'['+str(I)+']')
        if chcnt !=self.results.NrCh:
            print('An error has been detected that corresponds to inconsistencies in the size of channels. The following is printed for debugging purposes. Compare this with the sel file (or equivalent) to determine what sensor is in error. The data in htc_contents.py needs to be updated to reflect the correct value.')
            print('sensor_size', 'starts_at_channel', 'sensor_raw_line')
            at_ch=0
            for sensor in sensor_list:
                if sensor['__on__']:
                    sensor_size = sensor.get_sensor_size()
                    print(sensor_size, at_ch, sensor.raw_line)
                    at_ch+=sensor_size
            raise Exception('It appears the number of channels in the results is different than the number as described by the channels')
        self.results.set_keys(key_list)

        # return to the old directory
        os.chdir(old_home)
    
    def add_sensor_from_line(self, line):
        if self.htcf is None:
            raise Exception('There is no simulation to add sensor to')
        self.htcf['output'].add_sensor_from_line(line)

    def keys(self):
        retval = []
        if not self.htcf is None:
            htc_keys = self.htcf.all_keys()
            for key in htc_keys:
                retval.append('hawc2_input.'+str(key))
        if not self.results is None:
            result_keys = self.results.keys()
            for key in result_keys:
                retval.append('hawc2_output.'+str(key))
        return retval

    # Retrieve some information from the object
    def __getitem__(self,key):
        if key == 'hawc2_input':
            return self.htcf
        elif key == 'hawc2_output':
            return self.results
        elif key.startswith('hawc2_input.'):
            if self.htcf is None:
                raise Exception('The HAWC2 input structure has not been created')
            key = key[len('hawc2_input.'):]
            return self.htcf[key]
        elif key.startswith('hawc2_output.'):
            if self.results is None:
                raise Exception('The HAWC2 output structure has not been created')
            key = key[len('hawc2_output.'):]
            return self.results[key]
        else:
            raise KeyError('That key does not exist')

    # Set some information from on the object
    def __setitem__(self,key,value):
        if key == 'hawc2_input':
            self.htcf = value
        elif key == 'hawc2_output':
            self.results = value
        elif key.startswith('hawc2_input.'):
            if self.htcf is None:
                raise Exception('The HAWC2 input structure has not been created')
            key = key[len('hawc2_input.'):]
            self.htcf[key] = value
        elif key.startswith('hawc2_output.'):
            if self.results is None:
                raise Exception('The HAWC2 output structure has not been created')
            key = key[len('hawc2_output.'):]
            self.results[key] = value
        else:
            raise KeyError('That key does not exist')

