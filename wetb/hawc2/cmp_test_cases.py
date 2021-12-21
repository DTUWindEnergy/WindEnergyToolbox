import string
import shutil
import numpy as np
import os
import filecmp

import matplotlib.pyplot as plt






import unittest
import numpy as np
import re
import sys
from wetb.hawc2.Hawc2io import ReadHawc2


class CompareTestCases(unittest.TestCase):


    def compare_lines(self, ref_file, test_file, skip_first=0):
        with open(ref_file, encoding='utf-8') as ref:
            ref_lines = ref.readlines()
        with open(test_file, encoding='utf-8') as test:
            test_lines = test.readlines()
        self.assertEqual(len(ref_lines), len(test_lines), "\nNumber of lines differs in: '%s' and '%s'" % (ref_file, test_file))
        for i, (ref_l, test_l) in enumerate(zip(ref_lines[skip_first:], test_lines[skip_first:])):
            if ref_l.lower() != test_l.lower():
                diff = "".join([[" ", "^"][a != b] for a, b in zip(ref_l, test_l)])
                err_str = "%s%s%s\n\n" % (ref_l, test_l, diff)
                raise AssertionError("Difference in line %d of %s\n%s" % (i, ref_file, err_str))

    def compare_sel(self, ref_file, test_file):
        self.compare_lines(ref_file, test_file, 8)


    def compare_dat_contents(self, ref_file, test_file):
        if filecmp.cmp(ref_file, test_file, shallow=False) is False:
            self.compare_lines(ref_file, test_file)

    def min_tol(self, ref_data, test_data):
        def error(x, a, b):
            atol, rtol = x
            if rtol > 0 and atol > 0 and np.allclose(b, a, rtol, atol):
                return rtol + atol
            else:
                return 10 ** 99
        from scipy.optimize import fmin
        atol, rtol = fmin(error, (1, 1), (ref_data, test_data), disp=False)
        return atol, rtol




    def compare_dat_plot(self, ref_file, test_file, show_plot=False, rtol=1.e-5, atol=1.e-8):

        ref = ReadHawc2(os.path.splitext(ref_file)[0])
        test = ReadHawc2(os.path.splitext(test_file)[0])
        ref_data = ref()
        test_data = test()
        if not np.allclose(ref_data, test_data, rtol=rtol, atol=atol):
            different_sensors = []
            for i in range(ref.NrCh):
                if not np.allclose(ref_data[:, i], test_data[:, i], rtol=rtol, atol=atol):
                    different_sensors.append(i)
            path = os.path.join(os.path.dirname(test_file), "Compare", os.path.splitext(os.path.basename(ref_file))[0])
            shutil.rmtree(path, ignore_errors=True)
            try:
                os.mkdir(path)
            except:
                try:
                    os.mkdir(os.path.join(os.path.dirname(test_file), "Compare"))
                    os.mkdir(path)
                except:
                    pass
            valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)

            sys.stderr.write("%d sensors are different in datafiles\n" % len(different_sensors))
            sys.stderr.write("All close with\nAbsolute tolerance: %.6f\nRelative tolerance: %.6f\n\n" % self.min_tol(ref_data, test_data))

            sys.stderr.write("  ".join("%-20s" % s for s in ["Mean abs error", "Mean rel error (%)", "Max abs error", "Max rel error(%)", "Sensor"]) + "\n")
            abs_err = np.abs(ref_data - test_data)
            mask = (np.abs(ref_data) != 0)
            rel_err = abs_err[mask] / np.abs(ref_data[mask]) * 100
            err_str = ["%.6f" % e for e in [np.mean(abs_err), np.mean(rel_err), np.max(abs_err), np.max(rel_err)]]
            err_str = "  ".join(["%-20s" % e  for e in err_str])
            sys.stderr.write("%s  All data values\n" % (err_str))

            for i in different_sensors:
                abs_err = np.abs(ref_data[:, i] - test_data[:, i])
                mask = (np.abs(ref_data[:, i]) != 0)
                rel_err = abs_err[mask] / np.abs(ref_data[mask, i]) * 100
                err_str = ["%.6f" % e for e in [np.mean(abs_err), np.mean(rel_err), np.max(abs_err), np.max(rel_err)]]
                err_str = "  ".join(["%-20s" % e  for e in err_str])
                sys.stderr.write("%s  %d %s [%s] %s\n" % (err_str, (i + 1), ref.ChInfo[0][i], ref.ChInfo[1][i], ref.ChInfo[2][i]))
                sys.stderr.flush()
                plt.cla()
                plt.plot(ref_data[:, i], 'g', lw=3, label="Ref: %s [%s] %s" % (ref.ChInfo[0][i], ref.ChInfo[1][i], ref.ChInfo[2][i]))
                plt.plot(test_data[:, i], 'r', lw=1, label="test: %s [%s] %s" % (test.ChInfo[0][i], test.ChInfo[1][i], test.ChInfo[2][i]))
                from matplotlib.font_manager import FontProperties
                fontP = FontProperties()
                fontP.set_size('small')
                plt.legend(loc='best', prop=fontP)
                plt.axes().set_title(os.path.basename(ref_file))
                if show_plot:
                    plt.show()
                else:

                    plot_file = os.path.join(path, ("%03d_" % (i + 1)) + "".join([c for c in  ref.ChInfo[0][i] if c in valid_chars]) + ".png")
                    plt.savefig(plot_file)
            #raise AssertionError("Difference in the the values of:\n%s" % "\n".join(["%d %s" % (i + 1, ref.ChInfo[0][i]) for i in different_sensors]))



    def version_tag(self, filename):
        re_version = re.compile(r".*_(\d*\.\d*)\.sel")
        match = re.match(re_version, filename)
        if match and len(match.groups()) == 1:
            return match.group(1)
        return ""

    def common_path(self, path1, path2):
        cp = []
        for f1, f2 in zip(os.path.realpath(path1).split(os.path.sep), os.path.realpath(path2).split(os.path.sep)):
            if f1 == f2:
                cp.append(f1)
            else:
                break
        return os.path.sep.join(cp)


    def compare_file(self, ref_file, test_file, show_plot=False, rtol=1.e-5, atol=1.e-8):
        try:
            assert os.path.isfile(test_file), "File '%s' not found" % test_file
            try:
                self.compare_sel(ref_file, test_file)
                self.compare_dat_plot(ref_file.replace(".sel", ".dat"), test_file.replace(".sel", ".dat"), show_plot=show_plot, rtol=rtol, atol=atol)
                print ("ok\n\n\n")
            except AssertionError as e:
                sys.stderr.write(str(e) + "\n")
                self.compare_dat_plot(ref_file.replace(".sel", ".dat"), test_file.replace(".sel", ".dat"), show_plot=show_plot, rtol=rtol, atol=atol)
                print ("Data file ok\n\n\n")

        except AssertionError as e:
            sys.stderr.write (str(e) + "\n\n")


    def compare_folder(self, ref_res_path, test_res_path, ref_version_tag, test_version_tag, show_plot=False, rtol=1.e-5, atol=1.e-8):

        files = [f for f in os.listdir(ref_res_path) if f.endswith(".sel")]
        common_path = self.common_path(ref_res_path, test_res_path)
        for filename in files:

            self.version_tag(filename)
            ref_file = os.path.join(ref_res_path, filename)
            ref_version_tag = self.version_tag(filename)
            test_version_tag = self.version_tag([f for f in os.listdir(test_res_path) if f.endswith(".sel")][0])




            print ("-"*50)

            try:
                prefix = filename[:filename.index(ref_version_tag)]
                postfix = filename[filename.index(ref_version_tag) + len(ref_version_tag):]
                test_filename = [f for f in os.listdir(test_res_path) if f.startswith(prefix) and f.endswith(postfix)][0]
                test_file = os.path.join(test_res_path, test_filename)
                print ("Comparing %s and %s\n" % tuple(f.replace(common_path, "") for f in (ref_file, test_file)))
            except IndexError:
                sys.stdout.flush()
                sys.stderr.write ("\nNo matching test file found for %s\n\n" % ref_file.replace(common_path, ""))
                sys.stderr.flush()
                continue

            self.compare_file(ref_file, test_file, show_plot, rtol, atol)

    def runTest(self):
        pass


if __name__ == "__main__":

    ref_path = r'S:\AED\HAWC2\HAWC2_release_test_cases\version_11.4\output\res/'
    test_path = r'S:\AED\HAWC2\HAWC2_release_test_cases\version_11.8w\output\res/'
    # rtol: relative tolerance
    # atol: absolute tolerance
    # absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))
    CompareTestCases().compare_folder(ref_path, test_path, "11.4", "11.8w", show_plot=False, rtol=1.e-5, atol=1.e-5)
