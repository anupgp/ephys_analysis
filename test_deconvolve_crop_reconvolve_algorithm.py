from ephys_class import EphysClass
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import re

ephysdatapath="/Volumes/GoogleDrive/Shared drives/Beique Lab DATA/Ephys Data/Olympus 2P/Anup/"
ophysdatapath="/Volumes/GoogleDrive/Shared drives/Beique Lab DATA/Imaging Data/Olympus 2P/Anup/"
masterdfpath = "/Users/macbookair/goofy/data/beiquelab/pfc_clusterd_inputs/pfc_clustered_glu_uncaging.xlsx"
masterdf = pd.read_excel(masterdfpath,header=0,use_cols=10)
masterdf["datefolder"] = masterdf["datefolder"].astype(int)
masterdf["date"] = pd.to_datetime(masterdf["datefolder"],format="%Y%m%d")
masterdf["dob"] = pd.to_datetime(masterdf["dob"],format="%Y%m%d")
masterdf["age"] = (masterdf["date"]-masterdf["dob"]).dt.days
print(masterdf)
channels =  ['ImLEFT', 'IN1', 'IN7']
resch = "ImLEFT"
clampch = "IN1"
trgch = "IN7"
for row in range(0,masterdf.shape[0]):
# for row in range(20,30):
    datefolder = str(masterdf.loc[row,"datefolder"])
    cellfolder = str(masterdf.loc[row,"cellfolder"])
    ephysfile = os.path.join(ephysdatapath,datefolder,cellfolder,str(masterdf.loc[row,"ephysfile"]))
    ophysfile = os.path.join(ophysdatapath,datefolder,cellfolder,str(masterdf.loc[row,"imagefile"]))
    print(ephysfile,"\n",ophysfile)
    neuronid = masterdf.loc[row,"neuronid"]
    branchid = masterdf.loc[row,"neuronid"]
    spineid = masterdf.loc[row,"neuronid"]
    clampmode = masterdf.loc[row,"clampmode"]
    stimmode = masterdf.loc[row,"stim-mode"]
    ephys = EphysClass(ephysfile,loaddata=True)
    # ephys.info()
    ephys.extract_stim_props(trgch)
    ephys.extract_res_props(resch,trgch)
    if(clampmode == "cc"):
        # ephys.estimate_tau_access_res_from_epsp(resch,clampch)
        ephys.deconvolve_crop_reconvolve(resch,clampch,trgch)
        input()
