from ephys_class import EphysClass
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import re

datapath="/Volumes/GoogleDrive/Shared drives/Beique Lab DATA/Ephys Data/Olympus 2P/Anup/"
expdate = "20200720"
cellid = "C1"
channels =  ['ImLEFT', 'IN1', 'IN7']
print(os.listdir(datapath))
print(os.listdir(os.path.join(datapath,expdate)))
files = os.listdir(os.path.join(datapath,expdate,cellid))
abffiles = [f for f in files if re.search(".*.abf",f)]
abffiles.sort()
print(abffiles)
for abffile in abffiles:
    print("Opening %s"%abffile)
    ephys = EphysClass(os.path.join(datapath,expdate,cellid,abffile),loaddata=True)
    ephys.info()
    ephys.show([0,1,3],[])           # [channels],[sweeps]
    # input()


