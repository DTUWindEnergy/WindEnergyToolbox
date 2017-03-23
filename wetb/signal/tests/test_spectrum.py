'''
Created on 12/11/2015

@author: MMPE
'''
import unittest
import numpy as np
from wetb.signal.spectrum import psd

class TestSpectrum(unittest.TestCase):




    def test_psd(self):
        """check peak at correct frequency"""
        fs = 1024
        t = np.arange(0, 100, 1.0 / fs)
        sig = np.sin(2 * np.pi * t * 200) + np.sin(2 * np.pi * t * 100)
        f, spectrum = psd(sig, fs)

        #import matplotlib.pyplot as plt
        #plt.plot(f, spectrum)
        #plt.show()

        self.assertAlmostEqual(f[spectrum > 0.0001][0], 100, 2)
        self.assertAlmostEqual(f[spectrum > 0.0001][1], 200, 2)



    def test_map_to_frq2(self):
        """check integral==variance"""
        sig = np.random.randint(-3, 3, 100000).astype(np.float64)
        f, y = psd(sig, 10)
        self.assertAlmostEqual(np.var(sig), np.trapz(y, f), 3)


    def test_map_to_frq_smoothing(self):
        """check integral==variance"""
        sig = np.random.randint(-3, 3, 10000).astype(np.float64)
        sig += np.sin(np.arange(len(sig)) * 2 * np.pi / 10)

        import matplotlib.pyplot as plt
#        plt.plot(sig)
#        plt.figure()
        for i, k in enumerate([1, 4], 1):
            #plt.subplot(1, 2, i)
            for n in [1, 4]:
                f, y = psd(sig, 10, k, n)
#                plt.semilogy(f, y, label='k=%d, n=%d' % (k, n))
                if k == 1:
                    self.assertAlmostEqual(np.var(sig), np.trapz(y, f), 1)
                self.assertAlmostEqual(f[np.argmax(y)], 1, 1)

#        plt.legend()
#        plt.show()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
