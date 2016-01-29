import glob
import unittest
import os

def module_strings():
    test_file_paths = glob.glob('test_*.py')
    #test_file_strings.extend(glob.glob('../wetb/**/test_*.py'))
    #for root,_,_ in os.walk("../wetb/"):

    for folder, _, _ in os.walk('../wetb'):
        test_file_paths.extend(glob.glob(os.path.join(folder, "tests/test_*.py")))
    return [s[3:len(s) - 3].replace(os.path.sep, ".") for s in test_file_paths]


def suite():
    try:
        suites = []
        for s in module_strings():
            suites.append(unittest.defaultTestLoader.loadTestsFromName(s))
    except:
        print ("Failed to import '%s'" % s)

    return unittest.TestSuite(suites)

#no_tests = 0
#all = set()
#all_modules = set()
#for s in module_strings():
#    if s.split('.')[-1] in all_modules:
#        print ("!!!!!%s already loaded" % s)
#    all_modules.add(s.split('.')[-1])
#    m = __import__(s, {}, {}, "*")
#    cls = [t for t in m.__dict__.keys() if t.lower().startswith('test') and t != 'TestCaseAppFunc' and t != "TestCase" and t.strip()[0] != "#"]
#
#    test_funcs = [t for t in dir(m.__dict__[cls[0]]) if t.lower().startswith('test') and t != 'testfilepath']
#    for t in test_funcs:
#        if t in all:
#            print ("!!!!!!!!! %s already present" % t)
#            pass
#        else:
#            all.add(t)
#    no_tests += len(test_funcs)
#    print ("%-40s" % s, len(test_funcs), "\t\t", test_funcs)
#print ("Number of tests: ", no_tests, len(all), len(module_strings()))




if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    test_suite = suite()
    runner.run(test_suite)
else:
    # for run as pydev unit-test
    try:
        for mstr in module_strings():
            __import__(mstr, {}, {}, "*")
            exec("from %s import *" % mstr)
    except Exception as e:
        for mstr in module_strings():
            print (mstr)
            __import__(mstr, {}, {}, "*")
            exec("from %s import *" % mstr)
