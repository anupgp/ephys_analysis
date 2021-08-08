from ephys_class import EphysClass
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import re

datapath="/home/anup/gdrive-beiquelabdata/Ephys Data/Olympus 2P/Anup/"
expdate = "20210716"
# expdate = "2021-02-10"
cellid = "c2"
# channels =  ['ImLEFT', 'IN1', 'IN7'] # olympus Rig
channels = ['ImRightP', 'VmRightS', 'IN6'] # LSM880
reschannel = channels[0]
clampchannel =  channels[1]
stimchannel = channels[2]
print(os.listdir(datapath))
print(os.listdir(os.path.join(datapath,expdate,cellid)))
files = os.listdir(os.path.join(datapath,expdate,cellid))
abffiles = [f for f in files if re.search(".*.abf",f)]
abffiles.sort()
print(abffiles)
for abffile in abffiles:
    print("Opening %s"%abffile)
    ephys = EphysClass(os.path.join(datapath,expdate,cellid,abffile),loaddata=True,clampch=clampchannel,resch=reschannel,stimch=stimchannel)
    ephys.extract_indices_holdingsteps(NstepsPos=1,NstepsNeg=1)
    ephys.extract_stim_props()
    # EphysClass.seriesres_voltageclamp(ephys,'ImRightP',clampchannel)
    ephys.info()
    ephys.show([0,1,2],[])           # [channels],[sweeps]
#     # input()
