from wetb.hawc2.htc_file import HTCFile
from wetb.hawc2.tests import test_files
import os


def main():
    if __name__ == '__main__':

        # ======================================================================
        # load existing htc file
        # ======================================================================
        path = os.path.dirname(test_files.__file__) + "/simulation_setup/DTU10MWRef6.0/"
        htc = HTCFile(path + "htc/DTU_10MW_RWT.htc")

        # ======================================================================
        # modify wind speed and turbulence intensity
        # ======================================================================

        htc.wind.wsp = 10

        # access wind section and change turbulence intensity
        wind = htc.wind
        wind.tint = .1

        #=======================================================================
        # print contents
        #=======================================================================
        print(htc)  # print htc file
        print(htc.keys())  # print htc sections
        print(wind)  # print wind section

        #=======================================================================
        # change tilt angle
        #=======================================================================
        orientation = htc.new_htc_structure.orientation

        # Two ways to access the relative orientation between towertop and shaft
        # 1) Knowning that it is the second relative section:
        rel_tt_shaft = orientation.relative__2
        # 2) Knowning that the section contains a field "body1" with value "topertop"
        rel_tt_shaft = orientation.get_subsection_by_name(name='towertop', field='body1')

        rel_tt_shaft.body2_eulerang__2 = 6, 0, 0
        print(rel_tt_shaft.body2_eulerang__2)

        # ======================================================================
        # set time, name and save
        # ======================================================================
        # set time of simulation, first output section and wind.scale_time_start
        htc.set_time(start=5, stop=10, step=0.1)

        # change name of logfile, animation, visualization and first output section
        htc.set_name("tmp_wsp10_tint0.1_tilt6")

        htc.save()  # Save htc modified htcfile as "tmp_wsp10_tint0.1_tilt6"

        # ======================================================================
        # run simulation
        # ======================================================================
        # htc.simulate("<path2hawc2>/hawc2mb.exe")


main()
