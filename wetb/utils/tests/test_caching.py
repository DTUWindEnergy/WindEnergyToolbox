'''
Created on 08/11/2013

@author: mmpe
'''
import multiprocessing
import time
import unittest


from wetb.utils.timing import get_time
from wetb.utils.caching import cache_function, set_cache_property


class Example(object):
    def __init__(self, *args, **kwargs):
        object.__init__(self, *args, **kwargs)

        set_cache_property(self, "test", self.slow_function)
        set_cache_property(self, 'pool', lambda : multiprocessing.Pool(20))

    def slow_function(self):
        time.sleep(1)
        return 1

    @cache_function
    def test_cache_function(self):
        return self.slow_function()

    @get_time
    def prop(self, prop):
        return getattr(self, prop)




def f(x):
    return x ** 2

class TestCacheProperty(unittest.TestCase):
    def setUp(self):
        pass

    def testcache_property_test(self):
        e = Example()
        self.assertAlmostEqual(e.prop("test")[1], 1, 1)
        self.assertAlmostEqual(e.prop("test")[1], 0, 2)

    def testcache_property_pool(self):
        e = Example()
        e.prop("pool")  #load pool
        self.assertAlmostEqual(e.prop("pool")[1], 0, places=4)
        #print (get_time(e.pool.map)(f, range(10)))


    def test_cache_function(self):
        e = Example()
        self.assertAlmostEqual(get_time(e.test_cache_function)()[1], 1, places=2)
        self.assertAlmostEqual(get_time(e.test_cache_function)()[1], 0, places=2)
        self.assertAlmostEqual(get_time(e.test_cache_function)(reload=True)[1], 1, places=2)
        self.assertAlmostEqual(get_time(e.test_cache_function)()[1], 0, places=2)
        e.clear_cache()
        self.assertAlmostEqual(get_time(e.test_cache_function)()[1], 1, places=2)





if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
