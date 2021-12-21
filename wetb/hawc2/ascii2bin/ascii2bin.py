import sys
import warnings

warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from wetb.hawc2.ascii2bin.pandas_dat_ascii2bin import pandas_dat_ascii2bin

class TextUI(object):
    def show_message(self, m):
        print (m)

    def exec_long_task(self, text, allow_cancel, task, *args, **kwargs):
        print (text)
        return task(*args, **kwargs)

sys.path.append(".")

def size_from_file(selfilename):
    with open(selfilename, encoding='utf-8') as f:
        info = f.readlines()[8].split()
        scans = int(info[0])
        no_sensors = int(info[1])
    return (scans, no_sensors)

def ascii2bin(ascii_selfilename, bin_selfilename=None, ui=TextUI()):

    # Convert dat file
    ascii_datfilename = ascii_selfilename.replace(".sel", '.dat')
    if bin_selfilename is None:
        bin_selfilename = ascii_selfilename[:-4] + "_bin.sel"


    # Read, convert and write sel file
    with open(ascii_selfilename, encoding='utf-8') as f:
        lines = f.readlines()

    if "BINARY" in lines[8]:
        ui.show_message("%s is already binary" % ascii_selfilename)
    else:
        #lines[1] = "  Version ID : Pydap %d.%d\n" % (version.__version__[:2])
        lines[5] = "  Result file : %s.dat\n" % bin_selfilename[:-4]
        lines[8] = lines[8].replace("ASCII", 'BINARY')
        lines[-1] = lines[-1].strip() + "\n"
        lines.append("Scale factors:\n")

        scale_factors = pandas_dat_ascii2bin(ascii_datfilename, bin_selfilename.replace('.sel', '.dat'), ui)
        for sf in scale_factors:
            lines.append("  %.5E\n" % sf)
        with open(bin_selfilename, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        if ui is not None:
            ui.show_message("Finish converting %s to %s" % (ascii_selfilename, bin_selfilename))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print ("syntax: ascii2bin ascii_sel_filename [bin_sel_filename]")
    elif len(sys.argv) == 2:
        ascii2bin(sys.argv[1])
    else:
        ascii2bin(sys.argv[1], sys.argv[2])
