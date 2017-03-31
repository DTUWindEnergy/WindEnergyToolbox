'''
Created on 27. mar. 2017

@author: mmpe
'''
import unittest

import numpy as np
from wetb.signal.filters.frq_filters import sine_generator, low_pass, \
    frequency_response, high_pass


class Test(unittest.TestCase):


    def test_sine_generator(self):
        t,y = sine_generator(10,1,2)
        self.assertEqual(np.diff(t).mean(),.1)
        self.assertAlmostEqual(t.max(),1.9)
        if 0:
            import matplotlib.pyplot as plt
            plt.plot(*sine_generator(10,1,2), label="1Hz sine")
            plt.plot(*sine_generator(100,2,2), label="2Hz sine")
            plt.legend()
            plt.show()
    
    def test_low_pass(self):
        sf = 100 # sample frequency
        t,y1 = sine_generator(sf,1,5) # 1Hz sine
        t,y10 = sine_generator(sf,10,5) # 10Hz sine
        sig = y1+y10
        y_lp = low_pass(sig, sf, 1, order=1)
        
        # 1 order: 
        # cut off frq: -3db
        # Above cut off: 20db/decade 
        np.testing.assert_almost_equal(y_lp[100:400], y1[100:400]*.501, 1)
        np.testing.assert_almost_equal((y_lp-y1*.501)[100:400], y10[100:400]*.01, 1)
        
        if 0:
            import matplotlib.pyplot as plt
            plt.plot(t,sig, label='Input signal')
            plt.plot(t, y1*0.501, label='1Hz sine (-3db)')
            plt.plot(t,y_lp, label='Output signal')
            plt.plot(t,y_lp - y1*0.501, label="Output - 1Hz sine(-3db)")
            plt.plot(t, y10*0.01, label='10Hz sine (-20db)')
            plt.plot(t, y_lp - y1*0.501 -  y10*.01, label="(Output - 1Hz sine(-3db)) - 10Hz sine (-20db)")
            plt.legend()
            plt.show()
            
    def test_frq_response(self):
        w,h = frequency_response(100,10,'low',1)
        self.assertAlmostEqual(np.interp(10, w, h), -3.01,2) # cut off frq -3.01 db
        self.assertAlmostEqual(np.interp(100, w, h), -20,1) # -20 db per decade
        if 0:
            import matplotlib.pyplot as plt
            frequency_response(100,10,'low',1, plt=plt)
            frequency_response(100,10,'low',2, plt=plt)
            frequency_response(100,10,'low',5, plt=plt)
            frequency_response(100,10,'low',8, plt=plt)
            frequency_response(100,10,'low',10, plt=plt)
            frequency_response(100,10,'high',1, plt=plt)
            frequency_response(100,10,'high',2, plt=plt)
            plt.show()
        
    def test_high_pass(self):
        sf = 100 # sample frequency
        t,y1 = sine_generator(sf,1,5) # 1Hz sine
        t,y10 = sine_generator(sf,10,5) # 10Hz sine
        sig = y1+y10
        y_lp = high_pass(sig, sf, 10, order=1)
        
        # 1 order: 
        # cut off frq: -3db
        # Below cut off: 20db/decade 
        np.testing.assert_almost_equal(y_lp[100:400], y10[100:400]*.501, 1)
        np.testing.assert_almost_equal((y_lp-y10*.501)[100:400], y1[100:400]*.01, 1)
        
        if 0:
            import matplotlib.pyplot as plt
            plt.plot(t,sig, label='Input signal')
            plt.plot(t, y10*0.501, label='10Hz sine (-3db)')
            plt.plot(t,y_lp, label='Output signal')
            plt.plot(t,y_lp - y10*0.501, label="Output - 10Hz sine(-3db)")
            plt.plot(t, y1*0.01, label='1Hz sine (-20db)')
            plt.plot(t, y_lp - y10*0.501 -  y1*.01, label="(Output - 10Hz sine(-3db)) - 1Hz sine (-20db)")
            plt.legend()
            plt.show()
         
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_sine_generator']
    unittest.main()