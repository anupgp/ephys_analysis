from ephys_class import EphysClass
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import re
import math

# analysis of clustered glutamate uncaging at proximal spines on PFC neurons
# ephysdatapath="/Volumes/GoogleDrive/Shared drives/Beique Lab DATA/Ephys Data/Olympus 2P/Anup/"
# ophysdatapath="/Volumes/GoogleDrive/Shared drives/Beique Lab DATA/Imaging Data/Olympus 2P/Anup/"
# masterdfpath = "/Users/macbookair/goofy/data/beiquelab/pfc_clusterd_inputs"
# masterdf = "pfc_clustered_glu_uncaging.xlsx"
# dfname = "ephys_analysis_epsp_clustered_input_pfcl5.csv"

 # analysis of clustred glutamate uncaging at proximal spines on CA1 neurons
ephysdatapath="/home/anup/gdrive-beiquelabdata/Ephys Data/Olympus 2P/Anup/"
ophysdatapath="/home/anup/gdrive-beiquelabdata/Imaging Data/Olympus 2P/Anup/"
masterdfpath = "/home/anup/gdrive-beiquelab/CURRENT LAB MEMBERS/Anup Pillai/ca1_uncaging_cluster/"
masterdf = "ca1_glu_uncaging_clustered.xlsx"
dfname = "ephys_analysis_epsp_clustered_input_hcca1.csv"
loadcols = ["selectfile","badsweeps","datefolder","cellfolder","neuronid","dendriteid","spineid","ephysfile","clampmode","nstims","isi","laserstimpower","accessres","cellfolder_ophys","imagefile","dob","zstack","zstackfolder"]
figdirname = "deconvolved_traces"

# masterdf = pd.read_excel(os.path.join(masterdfpath,masterdf),header=0,usecols=loadcols)
masterdf = pd.read_excel(os.path.join(masterdfpath,masterdf),header=0)
masterdf["datefolder"] = masterdf["datefolder"].astype(int)
masterdf["date"] = pd.to_datetime(masterdf["datefolder"],format="%Y%m%d")
masterdf["dob"] = pd.to_datetime(masterdf["dob"],format="%Y%m%d")
masterdf["age"] = (masterdf["date"]-masterdf["dob"]).dt.days
print(masterdf)
input()
channels =  ['ImLEFT', 'IN1', 'IN7']
resch = "ImLEFT"
clampch = "IN1"
trgch = "IN7"
# open pandas dataframe to populate data
roidf = pd.DataFrame()

for row in range(0,masterdf.shape[0]):
# for row in range(142,masterdf.shape[0]):
    selectfile = int(masterdf.loc[row,"selectfile"])
    datefolder = str(masterdf.loc[row,"datefolder"])
    cellfolder = str(masterdf.loc[row,"cellfolder"])
    ephysfile = str(masterdf.loc[row,"ephysfile"])
    abffile = os.path.join(ephysdatapath,datefolder,cellfolder,ephysfile)
    # ophysfile = os.path.join(ophysdatapath,datefolder,cellfolder,str(masterdf.loc[row,"imagefile"]))
    print(ephysfile)
    neuronid = int(masterdf.loc[row,"neuronid"])
    dendriteid = int(masterdf.loc[row,"dendriteid"])
    badsweeps = masterdf.loc[row,"badsweeps"]
    if (math.isnan(badsweeps)):
        badsweeps = []
    print("badsweeps: ",badsweeps)
    spineid = masterdf.loc[row,"spineid"]
    clampmode = str(masterdf.loc[row,"clampmode"])
    stimpower = masterdf.loc[row,"laserstimpower"]
    dob = masterdf.loc[row,"dob"]
    ephys = EphysClass(abffile,loaddata=True,badsweeps=badsweeps)
    ephys.info()
    
    ephys.extract_stim_props(trgch)
    ephys.extract_res_props(resch,trgch)
    # input()
    if((clampmode == "vc") or (selectfile ==0)):
        continue;
    if(clampmode == "cc"):
        ephys.get_clampprop(resch,clampch)
        tau,accessres = ephys.estimate_tau_access_res_from_epsp(resch,clampch)
        peaks,fh,ah = ephys.deconvolve_crop_reconvolve(resch,clampch,trgch)
        isi = ephys.isi
        nstims = int(ephys.nstim)
        nsweeps = int(ephys.nsweeps)
        for isweep in np.arange(0,peaks.shape[0]):
            # {
            for istim in np.arange(0,peaks.shape[1]):
                # {
                record = {"ephysfile":ephysfile,"neuronid":neuronid}
                record.update({"expdate":datefolder,"cellfolder":cellfolder,"dendriteid":dendriteid})
                record.update({"spineid":spineid,"nstims":nstims,"nsweeps":nsweeps})
                record.update({"isweep": isweep+1, "isi":isi,"istim":istim+1,"clampmode":clampmode})
                record.update({"peak":peaks[isweep,istim]})
                record.update({"accessres":accessres/1E6,"tau":tau*1E3,"dob":dob,"stimpower":stimpower})
                roidf = roidf.append(record,ignore_index = True) # append a record per stim
                # }
            # }
        figname = "".join((datefolder,"_","neuron",str(neuronid),"_","dend",str(dendriteid),"_","spine-",spineid,("_"),str(int(isi*1000)),"ms",".png"))
        figpath = os.path.join(masterdfpath,figdirname,figname)
        print(figpath)
        # input()
        fh.savefig(figpath) 
        # input()
        # plt.show()
        # plt.close()
            # ----------
    print(roidf)
    # input()
# -----------------
# save the dataframe
roidf.to_csv(os.path.join(masterdfpath,dfname),index=False)
