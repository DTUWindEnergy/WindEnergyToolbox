from ._io import load, read_sensor_info
from wetb import gtsdf
import numpy as np


class Dataset(gtsdf.Dataset):
    def __init__(self, filename, name_stop=8, dtype=float):
        self.time, self.data, self.info = load(filename, name_stop, dtype)
