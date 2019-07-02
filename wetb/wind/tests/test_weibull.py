'''
Created on 04/12/2015

@author: mmpe
'''
import unittest
import matplotlib.pyplot as plt
import numpy as np
from wetb.wind import weibull


class TestWeibull(unittest.TestCase):

    def test_weibull(self):
        wb = weibull.pdf(4, 2)
        x = np.arange(0, 20, .1)
        self.assertAlmostEqual(wb(x).max(), 0.21441923796454562)
        if 0:
            plt.plot(x, wb(x))
            plt.show()

    def test_random_weibull(self):
        y = weibull.random(4, 2, 1000000)
        pdf, x = np.histogram(y, 100, normed=True)
        x = (x[1:] + x[:-1]) / 2
        self.assertLess(sum((pdf - weibull.pdf(4, 2)(x)) ** 2), 0.0001)
        if 0:
            plt.plot(x, pdf)
            plt.plot(x, weibull.pdf(4, 2)(x))
            plt.show()

    def test_fit_weibull(self):
        np.random.seed(1)
        y = weibull.random(4, 2, 1000000)
        A, k = weibull.fit(y)
        self.assertAlmostEqual(A, 4, delta=0.01)
        self.assertAlmostEqual(k, 2, delta=0.01)
        if 0:
            plt.hist(y, 100, normed=True)
            x = np.arange(0, 20, .1)
            plt.plot(x, weibull.pdf(4, 2)(x))
            plt.show()

    def test_weibull_cdf(self):
        wb = weibull.pdf(4, 2)
        x = np.arange(0, 20, .01)
        cdf = weibull.cdf(4, 2)
        self.assertEqual(cdf(999), 1)
        np.testing.assert_array_almost_equal(np.cumsum(wb(x)) / 100, cdf(x), 3)

        if 0:
            plt.plot(x, np.cumsum(wb(x)) * .01)
            plt.plot(x, cdf(x))
            plt.show()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testweibull']
    unittest.main()
