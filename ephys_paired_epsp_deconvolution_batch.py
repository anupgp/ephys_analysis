from ephys_class import EphysClass
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import re

ephysdatapath="/Volumes/GoogleDrive/Shared drives/Beique Lab DATA/Ephys Data/Olympus 2P/Anup/"
ophysdatapath="/Volumes/GoogleDrive/Shared drives/Beique Lab DATA/Imaging Data/Olympus 2P/Anup/"
masterdfpath = "/Users/macbookair/goofy/data/beiquelab/pfc_clusterd_inputs"
masterdf = "pfc_clustered_glu_uncaging.xlsx"

masterdf = pd.read_excel(os.path.join(masterdfpath,masterdf),header=0,use_cols=10)
masterdf["datefolder"] = masterdf["datefolder"].astype(int)
masterdf["date"] = pd.to_datetime(masterdf["datefolder"],format="%Y%m%d")
masterdf["dob"] = pd.to_datetime(masterdf["dob"],format="%Y%m%d")
masterdf["age"] = (masterdf["date"]-masterdf["dob"]).dt.days
print(masterdf)
channels =  ['ImLEFT', 'IN1', 'IN7']
resch = "ImLEFT"
clampch = "IN1"
trgch = "IN7"
# open pandas dataframe to populate data
roidf = pd.DataFrame()

for row in range(0,masterdf.shape[0]):
    selectfile = int(masterdf.loc[row,"selectfile"])
    datefolder = str(masterdf.loc[row,"datefolder"])
    cellfolder = str(masterdf.loc[row,"cellfolder"])
    ephysfile = str(masterdf.loc[row,"ephysfile"])
    abffile = os.path.join(ephysdatapath,datefolder,cellfolder,ephysfile)
    # ophysfile = os.path.join(ophysdatapath,datefolder,cellfolder,str(masterdf.loc[row,"imagefile"]))
    print(ephysfile)
    neuronid = int(masterdf.loc[row,"neuronid"])
    dendriteid = int(masterdf.loc[row,"dendriteid"])
    spineid = masterdf.loc[row,"spineid"]
    clampmode = str(masterdf.loc[row,"clampmode"])
    stimpower = masterdf.loc[row,"laserstimpower"]
    dob = masterdf.loc[row,"dob"]
    stimmode = str(masterdf.loc[row,"stimmode"])
    ephys = EphysClass(abffile,loaddata=True)
    # ephys.info()
    ephys.extract_stim_props(trgch)
    ephys.extract_res_props(resch,trgch)
    if((clampmode == "vc") or (selectfile ==0)):
        continue;
    if(clampmode == "cc"):
        tau,accessres = ephys.estimate_tau_access_res_from_epsp(resch,clampch)
        peaks,fh,ah = ephys.deconvolve_crop_reconvolve(resch,clampch,trgch)
        isi = ephys.isi
        nstims = int(ephys.nstim)
        nsweeps = int(ephys.nsweeps)
        for isweep in np.arange(0,peaks.shape[0]):
            for istim in np.arange(0,peaks.shape[1]):
                record = {"ephysfile":ephysfile,"neuronid":neuronid}
                record.update({"expdate":datefolder,"cellfolder":cellfolder,"dendrite":dendriteid})
                record.update({"spineid":spineid,"nstims":nstims,"nsweeps":nsweeps})
                record.update({"isweep": isweep+1, "isi":isi,"istim":istim+1,"clampmode":clampmode})
                record.update({"peak":peaks[isweep,istim]})
                record.update({"accessres":accessres/1E6,"tau":tau*1E3,"dob":dob,"stimpower":stimpower})
                roidf = roidf.append(record,ignore_index = True) # append a record per stim
        # plt.show()
        plt.close()
            # ----------
    # print(roidf)
    # input()
# -----------------
# save the dataframe
dfname = "ephys_analysis_epsp_clustered_input.csv"
roidf.to_csv(os.path.join(masterdfpath,dfname),index=False)
