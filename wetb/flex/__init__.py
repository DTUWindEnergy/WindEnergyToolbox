from ._io import load, read_sensor_info
from wetb import gtsdf

class Dataset(gtsdf.Dataset):
    def __init__(self, filename):
        self.time, self.data, self.info = load(filename)