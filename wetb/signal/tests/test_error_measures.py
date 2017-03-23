'''
Created on 20/07/2016

@author: MMPE
'''
import os
import unittest

import numpy as np
from wetb.signal.error_measures import rms2fit_old, rms2fit
from wetb.signal.fit import bin_fit


tfp = os.path.join(os.path.dirname(__file__), 'test_files/')
class Test(unittest.TestCase):


#    def test_rms2mean(self):
#        data = np.load(tfp + "wsp_power.npy")
#        print (data.shape)
#        wsp = data[:, 1].flatten()
#        power = data[:, 0].flatten()
#
#        import matplotlib.pyplot as plt
#        plt.plot(wsp, power, '.')
#        x = np.linspace(wsp.min(), wsp.max(), 100)
#        err, f = rms2mean(wsp, power)
#        plt.plot(x, f(x), label='rms2mean, err=%.1f' % err)
#        err, f = rms2fit(wsp, power, bins=20, kind=3, fit_func=np.median)
#        plt.plot(x, f(x), label='rms2median, err=%.1f' % err)
#        print (list(x))
#        print (list(f(x)))
#        plt.legend()
#        plt.show()

    def test_rms2fit(self):
        x = np.array([10.234302313156817, 13.98517783627376, 7.7902362498947921, 11.08597865379001, 8.430623529700588, 12.279982848438033, 33.89151260027775, 12.095047111211629, 13.731371675689642, 14.858309846006723, 15.185588405617654])
        y = np.array([28.515665187174477, 46.285328159179684, 17.763652093098958, 32.949007991536462, 20.788106673177083, 38.819226477864589, 96.53278479817709, 38.479684539388025, 46.072654127604167, 51.875484233398439, 53.379342967122398])
        err, fit = rms2fit_old(x, y, kind=1, bins=15)
        err2, fit2 = rms2fit(x, y, fit_func=lambda x,y: bin_fit(x,y, kind=1, bins=15, bin_min_count=1, lower_upper='extrapolate'))
        self.assertAlmostEqual(err, 0.306,2)
        if 0:
            import matplotlib.pyplot as plt
            plt.plot(x,y, '.')
            x_ = np.linspace(x.min(), x.max(), 100)
            plt.plot(x_, fit(x_), label='rms2fit, err=%.5f' % err)
            plt.plot(x_, fit2(x_), label='rms2fit, err=%.5f' % err2)
            plt.legend()
            plt.show()
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_rms2mean']
    unittest.main()
