from wetb.hawc2.hawc2_input_writer import HAWC2InputWriter
from wetb.hawc2.tests import test_files
import os


def main():
    if __name__ == '__main__':

        # ======================================================================
        # load master htc file path
        # ======================================================================
        path = os.path.dirname(test_files.__file__) + "/simulation_setup/DTU10MWRef6.0/"
        htc_base_file = path + 'htc/DTU_10MW_RWT.htc'
        
        h2writer = HAWC2InputWriter(htc_base_file)

        # ======================================================================
        # Set DLC definition module filename
        # ======================================================================
        definition_file = 'dlc_definition.py'

        #=======================================================================
        # Generate htc files from definition file, and print list of simulations
        #=======================================================================
        df = h2writer.from_definition(definition_file)
        print(df)

main()