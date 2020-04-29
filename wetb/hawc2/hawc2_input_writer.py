import itertools
import jinja2
import pandas as pd
import click
import os
from pathlib import Path
from pandas.core.base import PandasObject
from wetb.hawc2.htc_file import HTCFile


class HAWC2InputWriter(object):
    """
    Base HAWC2InputWriter object class, using the tagless system. Subclasses are:
     - JinjaWriter
    """

    def __init__(self, base_htc_file, **kwargs):
        self.base_htc_file = base_htc_file
        self.contents = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, out_fn, **kwargs):
        return self.write(out_fn, **kwargs)

    def from_pandas(self, dataFrame):
        """
        Loads a DataFrame of contents from a PandasObject

        Args:
            dataFrame (Pandas.DataFrame of PandasObject): Dataframe of contents

        Returns:
            self (HAWC2InputWriter): Instance of self to enable chaining.
        """

        if not isinstance(dataFrame, PandasObject) and hasattr(dataFrame, 'to_pandas'):
            dataFrame = dataFrame.to_pandas()

        if isinstance(dataFrame, dict):
            return {n: self.from_pandas(df.dropna(how='all')) for n, df in dataFrame.iteritems()}

        self.contents = dataFrame

        return self

    def from_excel(self, excel_file):
        """
        Loads a DataFrame of contents from an excel file

        Args:
            excel_file (str, Pathlib.Path): path to excel file.

        Returns:
            self (HAWC2InputWriter): Instance of self to enable chaining.
        """

        self.contents = pd.read_excel(excel_file, keep_default_na=False)
        if 'Unnamed: 0' in self.contents:
            self.contents = self.contents.drop(columns=['Unnamed: 0'])
        return self

    def from_CVF(self, constants, variables, functionals):
        """
        Produces a DataFrame of contents given dictionaries of constants, variables
        and functionals.

        Args:
            constants (dict): content, value pairs for which the value remains
                constant for all generated htc files.
            variables (dict of list): content, list pairs for which all combinations
                of the listed values are generated in the htc files.
            functionals (dict of functions): content, function pairs for which the
                content is dependent on the constants and variables contents.

        Returns: self (HAWC2InputWriter): Instance of self to enable chaining.
        """

        attributes = list(constants.keys()) + list(variables.keys()) + list(functionals.keys())
        contents = []

        for var in itertools.product(*list(variables.values())):
            this_dict = dict(constants)
            var_dict = dict(zip(variables.keys(), var))
            this_dict.update(var_dict)

            for key, func in functionals.items():
                this_dict[key] = func(this_dict)

            contents.append(list(this_dict.values()))

        self.contents = pd.DataFrame(contents, columns=attributes)

        return self

    def write(self, out_fn, **kwargs):
        '''
        Renders a single htc file for a given set of 'tagless' contents.
        args:
        out_fn (str or pathlib.Path): The output filename where to render
        the jinja template.
        params (pd.DataFrame or pd.Series) : The input contents.
        '''
        # if isinstance(params, PandasObject):
        #     params = params.to_dict()

        htc = HTCFile(self.base_htc_file, jinja_tags=kwargs)
        for k, v in kwargs.items():
            k = k.replace('/', '.')
            if '.' in k:
                line = htc[k]
                v = str(v).strip().replace(",", " ")
                line.values = v.split()
            else:
                if hasattr(self, 'set_%s' % k):
                    getattr(self, 'set_%s' % k)(htc, **kwargs)

            htc.save(out_fn)

    def write_all(self, out_dir):
        '''
        Renders all htc files for the set of contents.
        args:
            out_dir (str or pathlib.Path): The directory where the htc files are generated.
        '''
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        N = len(self.contents)

        print(f'Generating {N} htc files in directory: {out_dir}')

        with click.progressbar(self.contents.iterrows(), length=N) as bar:
            for _, row in bar:
                if 'Folder' in row and row.Folder:
                    path = out_dir / row.Folder
                else:
                    path = out_dir

                self.write(path / (row.Name + '.htc'), **row.to_dict())


class JinjaWriter(HAWC2InputWriter):
    """
    !!! DEPRECATED
    !!! HAWC2InputWriter accepts jinja tags
    !!!
    Subclass of the HAWC2InputWriter object. Generates htc files using contents compatible with jinja2.
    """

    def write(self, out_fn, **kwargs):
        params = pd.Series(kwargs)
        with open(self.base_htc_file) as f:
            template = jinja2.Template(f.read())

            out = template.render(params)

            with open(out_fn, 'w') as f:
                f.write(out)


def main():
    if __name__ == '__main__':
        from wetb.hawc2.tests import test_files

        # HAWC2InputWriter example
        path = os.path.dirname(test_files.__file__) + '/simulation_setup/DTU10MWRef6.0/'
        base_htc_file = path + 'htc/DTU_10MW_RWT.htc'

        class MyWriter(HAWC2InputWriter):
            def set_time(self, htc, time, **_):
                htc.set_time(self.time_start, self.time_start + time)

        myWriter = MyWriter(base_htc_file, time_start=100)
        myWriter('tmp/t1.htc', Name="t1", time=600, **{"wind.wsp": 10})

        # JinjaWriter example with constants, variables, and functionals
        Constants = {
            'time_stop': 100,
        }

        Variables = {
            'wsp': [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
            'tint': [0.05, 0.10, 0.15],

        }

        Functionals = {
            'Name': lambda x: 'dtu10mw_wsp' + str(x['wsp']) + '_ti' + str(x['tint']),
        }

        htc_template_fn = os.path.dirname(test_files.__file__) + \
            '/simulation_setup/DTU10MWRef6.0/htc/DTU_10MW_RWT.htc.j2'
        writer = JinjaWriter(htc_template_fn)
        writer.from_CVF(Constants, Variables, Functionals)
        print(writer.contents)

        # write all htc files to folder 'htc'
        writer.write_all('htc')


main()
