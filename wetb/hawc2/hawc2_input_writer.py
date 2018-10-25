import pandas as pd
from wetb.hawc2.htc_file import HTCFile
from wetb.hawc2.tests import test_files
import os
import itertools
import importlib
from pandas.core.base import PandasObject
import numpy as np


class HAWC2InputWriter():

    def __init__(self, base_htc_file, **kwargs):
        self.base_htc_file = base_htc_file
        for k, v in kwargs.items():
            setattr(self, k, v)

    def from_pandas(self, dataFrame, write_input_files=True):
        if not isinstance(dataFrame, PandasObject) and hasattr(dataFrame, 'to_pandas'):
            dataFrame = dataFrame.to_pandas()
        if isinstance(dataFrame, pd.Panel):
            return {n: self.from_pandas(df.dropna(how='all'), write_input_files)for n, df in dataFrame.iteritems()}

        else:
            assert 'Name' in dataFrame
            if write_input_files:
                for _, row in dataFrame.iterrows():
                    self.write_input_files(**row.to_dict())
        return dataFrame

    def from_excel(self, excel_file, write_input_files=True):
        return self.from_pandas(pd.read_excel(excel_file), write_input_files)

    def from_CVF(self, constants, variables, functions, write_input_files=True):
        attributes = list(constants.keys()) + list(variables.keys()) + list(functions.keys())
        tags = []
        for var in itertools.product(*list(variables.values())):
            this_dict = dict(constants)
            var_dict = dict(zip(variables.keys(), var))
            this_dict.update(var_dict)
            for key, func in functions.items():
                this_dict[key] = func(this_dict)

            tags.append(list(this_dict.values()))

        df = pd.DataFrame(tags, columns=attributes)
        return self.from_pandas(df, write_input_files)

    def from_definition(self, definition_file, write_input_files=True):
        file = os.path.splitext(definition_file)[0]
        module = os.path.relpath(file, os.getcwd()).replace(os.path.sep, ".")
        def_mod = importlib.import_module(module)
        return self.from_CVF(def_mod.constants, def_mod.variables, def_mod.functions, write_input_files)

    def __call__(self, name, folder='', **kwargs):
        return self.write_input_files(name, folder, **kwargs)

    def write_input_files(self, Name, Folder='', **kwargs):
        htc = HTCFile(self.base_htc_file)
        for k, v in kwargs.items():
            k = k.replace('/', '.')
            if '.' in k:
                line = htc[k]
                v = str(v).strip().replace(",", " ")
                line.values = v.split()
            else:
                getattr(self, 'set_%s' % k)(htc, **kwargs)
        htc.set_name(Name, Folder)
        htc.save()
        args = {'Name': Name, 'Folder': Folder}
        args.update(kwargs)

        return pd.Series(args)


def main():
    if __name__ == '__main__':

        path = os.path.dirname(test_files.__file__) + '/simulation_setup/DTU10MWRef6.0/'
        base_htc_file = path + 'htc/DTU_10MW_RWT.htc'
        h2writer = HAWC2InputWriter(base_htc_file)

        # pandas2htc
        df = pd.DataFrame({'wind.wsp': [4, 6, 8], 'Name': ['c1', 'c2', 'c3']})
        h2writer.from_pandas(df)

        # HAWC2InputWriter
        class MyWriter(HAWC2InputWriter):
            def set_time(self, htc, time, **_):
                htc.set_time(self.time_start, self.time_start + time)

        myWriter = MyWriter(base_htc_file, time_start=100)
        myWriter("t1", 'tmp', time=600, **{"wind.wsp": 10})

        # from_CVF (constants, variables and functions)
        constants = {'simulation.time_stop': 100}
        variables = {'wind.wsp': [4, 6, 8],
                     'wind.tint': [0.1, 0.15, 0.2]}
        functions = {'Name': lambda x: 'sim_wsp' + str(x['wind.wsp']) + '_ti' + str(x['wind.tint'])}
        df = h2writer.from_CVF(constants, variables, functions, write_input_files=False)
        print(df)


main()
