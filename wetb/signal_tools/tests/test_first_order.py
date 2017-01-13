'''
Created on 13. jan. 2017

@author: mmpe
'''
import unittest
from wetb.signal_tools.filters import first_order
import numpy as np

class Test_first_order_filters(unittest.TestCase):


    def test_low_pass(self):
        a = np.random.randint(0,100,100).astype(np.float)
        b = first_order.low_pass(a, 1, 1)
        self.assertLess(b.std(), a.std())
        if 0:
            import matplotlib.pyplot as plt
            plt.plot(a)
            plt.plot(b)
            plt.show()

    def test_high_pass(self):
        a = np.random.randint(0,100,100).astype(np.float)
        b = first_order.high_pass(a, 1, 1)
        self.assertLess(b.mean(), a.mean())
        if 0:
            import matplotlib.pyplot as plt
            plt.plot(a)
            plt.plot(b)
            plt.show()
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()