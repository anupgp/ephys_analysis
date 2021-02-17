from ephys_class import EphysClass
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import re

# datapath="/home/anup/gdrive-beiquelabdata/Ephys Data/Olympus 2P/Anup/"
datapath="/home/anup/gdrive-beiquelabdata/Imaging Data/LSM880 2P/Kirk/Ephys/"
# expdate = "20201218"
expdate = "2021-02-10"
cellid = "C1"
# channels =  ['ImLEFT', 'IN1', 'IN7'] # olympus Rig
channels = ['ImRightP', 'VmRightS', 'IN6'] # LSM880
reschannel = [channels[0]]
clampchannel = [channels[1]]
print(os.listdir(datapath))
print(os.listdir(os.path.join(datapath,expdate)))
files = os.listdir(os.path.join(datapath,expdate))
abffiles = [f for f in files if re.search(".*.abf",f)]
abffiles.sort()
print(abffiles)
for abffile in abffiles:
    print("Opening %s"%abffile)
    ephys = EphysClass(os.path.join(datapath,expdate,abffile),loaddata=True)
    # EphysClass.seriesres_voltageclamp(ephys,'ImRightP',clampchannel)
    # ephys.info()
    ephys.show([0,2],[])           # [channels],[sweeps]
    # input()
