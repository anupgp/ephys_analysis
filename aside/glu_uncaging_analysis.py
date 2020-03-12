
import numpy as np
from neo.io import AxonIO
import matplotlib.pyplot as plt

import ephys_analysis as ephys

fname = "/Volumes/Anup_2TB/raw_data/beiquelab/zen/data_anup/20190627/C2/2019_06_27_0056.abf"

print(ephys.get_protocol_name(fname))
