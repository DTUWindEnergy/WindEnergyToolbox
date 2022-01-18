import itertools
import os
from pathlib import Path
import warnings

import click
import jinja2
import pandas as pd
from pandas.core.base import PandasObject

from wetb.hawc2.htc_file import HTCFile


class HAWC2InputWriter(object):
    """Load a dlc and write the corresponding HAWC2 input files.

    Parameters
    -----------
    base_htc_file : str or Pathlib.path
        Path to base htc file from which the other htc files are created.

    Notes
    ------
    You can load the dlc keywords and parameters using the `from_` methods presented
    below. Any methods of the form `set_<tag>`(htc, **kwargs), where `<tag>` is one
    of one of the keywords in the dlc, will be called when writing htc files.

    You can write the htc files using the `write` or `write_all` methods presented below.
    The `Name` tag must be defined when writing htc files.

    Necessary/protected tags
    -------------------------
    Name : str
        The name of the htc file (without extension) for each simulation. This will
        be used in a `set_Name` call to the htc file, which will overwrite the
        `simulation.logfile` and `output.filename` lines.
    """

    def __init__(self, base_htc_file, **kwargs):
        self.base_htc_file = base_htc_file
        self.contents = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, path, **kwargs):
        """Write single htc file if writer is called directly"""
        return self.write(path, **kwargs)

    def from_pandas(self, dataFrame):
        """Load dlc contents from a PandasObject.

        Parameters
        ----------
        dataFrame : Pandas.DataFrame or PandasObject
            Dataframe of contents.

        Returns
        -------
        self : HAWC2InputWriter
            Instance of self to enable chaining.
        """
        if not isinstance(dataFrame, PandasObject) and hasattr(dataFrame, 'to_pandas'):
            dataFrame = dataFrame.to_pandas()
        self.contents = dataFrame
        return self

    def from_excel(self, excel_file):
        """Load dlc contents from an excel file.

        Parameters
        ----------
        excel_file : str or Pathlib.Path
            Path to excel file.

        Returns
        -------
        self : HAWC2InputWriter
            Instance of self to enable chaining.
        """

        self.contents = pd.read_excel(excel_file, keep_default_na=False)
        if 'Unnamed: 0' in self.contents:
            self.contents = self.contents.drop(columns=['Unnamed: 0'])
        return self

    def from_CVF(self, constants, variables, functionals):
        """Load dlc contents from dictionaries of constants, variables and functionals.

        Parameters
        ----------
        constants : dict
            (content, value) pairs for which the value remains constant for all
            generated htc files. E.g.: {'time_stop': 700}.
        variables : dict of lists
            (content, list) pairs for which all combinations of the listed values are
            generated in the htc files. E.g., {'wsp': [4, 6, 8]}.
        functionals : dict of functions
            (content, function) pairs for which the content is dependent on the
            constants and variables contents. E.g., {'Name': lambda x: 'wsp' + str(x['wsp'])}.

        Returns
        -------
        self : HAWC2InputWriter
            Instance of self to enable chaining.
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

    def set_Name(self, htc, **kwargs):
        """Update the filename in simulation and output blocks of an htc file.

        Notes
        -----
        This function is called during the `write` and `write_all` methods, thereby
        overwriting the logfile and output filename. If you would like to specify a
        subfolder within the `htc` and `res` directories where your simulations should
        be placed, pass in the 'Folder' keyword into `kwargs`.

        Parameters
        ----------
        htc : wetb.hawc2.htc_file.HTCFile
            The htc file object to be modified.
        name : str
            The filename to be used in the logfile and simulation file (without
            extension).
        **kwargs
            Keyword arguments. Must include keyword 'Name'. May optionally include
            'Folder' to define a subfolder in your `htc` and `res` directories where
            files will be stored.
        """
        htc.set_name(kwargs['Name'], subfolder=kwargs.get('Folder', ''))

    def write(self, path, **kwargs):
        """Write a single htc file.

        Notes
        -----
        This function works both with the tagless and jinja file-writing systems. Any
        methods of the form `set_<tag>` are called during file writing.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to save the htc file to.
        **kwargs : pd.DataFrame, pd.Series or dict
            Keyword arguments. The tags to update/replace in the file.
        """
        htc = HTCFile(self.base_htc_file, jinja_tags=kwargs)
        for k, v in kwargs.items():
            k = k.replace('/', '.')
            if '.' in k:  # directly replace tags like "wind.wsp"
                line = htc[k]
                v = str(v).strip().replace(",", " ")
                line.values = v.split()
            else:  # otherwise, use the "set_" attribute
                if hasattr(self, 'set_%s' % k):
                    getattr(self, 'set_%s' % k)(htc, **kwargs)
        htc.save(path)

    def write_all(self, out_dir):
        """Write all htc files for the dlc.

        Notes
        -----
        This function works both with the tagless and jinja file-writing systems. Any
        methods of the form `set_<tag>` are called during each htc write. DLC contents
        must contain a 'Name' column.

        Parameters
        ----------
        out_dir : str or pathlib.Path
            The directory in which the htc files should be written.
        """
        try:
            self.contents['Name']
        except KeyError:
            raise KeyError('"Name" not given in dlc contents! Cannot write files.')

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
    def __init__(self, base_htc_file, **kwargs):
        """DEPRECATED!!!
        Subclass of the HAWC2InputWriter object. Generates htc files using contents compatible with jinja2.
        """
        warnings.warn('The JinjaWriter is deprecated! Please switch to the HAWC2InputWriter',
                      DeprecationWarning)
        HAWC2InputWriter.__init__(self, base_htc_file, **kwargs)

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

        # example of subclassing with custom set_<tag> method
        path = os.path.dirname(test_files.__file__) + '/simulation_setup/DTU10MWRef6.0/'
        base_htc_file = path + 'htc/DTU_10MW_RWT.htc'

        class MyWriter(HAWC2InputWriter):
            def set_time(self, htc, **kwargs):
                htc.set_time(self.time_start, self.time_start + kwargs['time'])

        myWriter = MyWriter(base_htc_file, time_start=100)
        myWriter('tmp/t1.htc', Name='t1', **{'time': 600, 'wind.wsp': 10})

        # example with constants, variables, and functionals
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
        writer = HAWC2InputWriter(htc_template_fn)
        writer.from_CVF(Constants, Variables, Functionals)
        print(writer.contents)

        # write all htc files to folder 'tmp'
        writer.write_all('tmp')


main()
