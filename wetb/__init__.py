from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
test = "TEST"
try:
    import pkg_resources
    __version__ = pkg_resources.safe_version(pkg_resources.get_distribution(__name__).version)
except:
    __version__ = 'unknown'
