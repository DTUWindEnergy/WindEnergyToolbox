import pkg_resources
test = "TEST"
try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'
