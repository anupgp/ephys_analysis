from ephys_class import EphysClass
import pandas as pd
import re
import os
import numpy as np
from itertools import chain
import datetime
from matplotlib import pyplot as plt

datapath = '/Volumes/GoogleDrive/Shared drives/Beique Lab/CURRENT LAB MEMBERS/Anup Pillai/ephys_mininum_stim/raw_data/'
abfname = "17511028.abf" 
reschannel = "ImRight"
trgchannel = "IN9"
vclampchannel = "VmRight"
ephys1 = EphysClass(datapath+abfname,loaddata=True)                                                    
ephys1.info()
print(ephys1.data.shape)
ephys1.extract_stim_props(trgchannel)
ephys1.extract_res_props(reschannel,trgchannel)
ephys1.extract_indices_holdingsteps(vclampchannel,1,1)
ephys1.show([0],np.arange(49,50))           # [channels],[sweeps]
# [tlags,ypeaks,fh,ah] = ephys1.find_peaks(reschannel,trgchannel,plotdata=False,peakdir="-")
# [tlagst,ypeakst,fht,aht] = ephys1.template_match(reschannel,trgchannel,peakdir="+")
# plt.show()
