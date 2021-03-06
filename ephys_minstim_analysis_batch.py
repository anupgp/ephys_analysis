import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import os
from ephys_class import EphysClass
import plotting_functions

mainpath = '/Volumes/GoogleDrive/Shared drives/Beique Lab/CURRENT LAB MEMBERS/Anup Pillai/ephys_mininum_stim/'
datafolder = "raw_data"
masterfilename = "minstim_summary_anup.xlsx"
reschannel = "ImRight"
trgchannel = "IN9"
vclampchannel = "VmRight"

masterdf = pd.read_excel(os.path.join(mainpath+masterfilename),sheet_name=1,header=0,use_cols=4)
print(masterdf)
# open pandas dataframe to populate data
roidf = pd.DataFrame()

for row in range(0,masterdf.shape[0]):
# for row in range(1,2):
    abfname = str(masterdf.loc[row,"abffile"])
    abffullpath = os.path.join(mainpath,datafolder,abfname)
    # skip if the abffile name is not proper
    if(re.search(r'^.*nan.*$',abfname) is not None):
        print('!!!! Warning: empty cell !!!!')
        print('Skipping to the next row')
        continue
    # skip if the file doesnot exist
    if(not os.path.exists(abffullpath)):
        print('abffile file does not exist!')
        print('Skipping to the next file')
        continue
    vclamp = str(masterdf.loc[row,"vclamp"])
    neuron = str(masterdf.loc[row,"neuron"])
    exp = str(masterdf.loc[row,"exp"])
    badsweeps = str(masterdf.loc[row,"badsweeps"]).split(",")
    fileid = re.search(".*[^.abf]",abfname)[0]+'_'+"neuron"+neuron+'_'+"vc"+vclamp+'_'+"exp"+exp
    print("Loading ABF file:\t",abfname)
    # read the csv file with labels
    labelfname = fileid+"_withlabel.csv"
    if (os.path.exists(mainpath+labelfname)):
        print("Loading LABEL file:\t",labelfname)
        labelfile = pd.read_csv(mainpath+labelfname,header=0)
        # crop column names to start from the first sweep
        rawlabels = [colname for colname in list(labelfile.columns) if re.search("^[\d]{8,8}.*neuron.*",colname)]
        print('Raw labels:\t',rawlabels)
        # extract labels from each colname
        labels = np.array([re.search("(?:.*sweep[\d]+_)([01b]+)",rawlabel)[1] for rawlabel in rawlabels])
        print("sweep labels: ",labels)
        # extract successes alone (can contain '1','0','b' for success, failure and bad
        isuccess = [label == '1' for label in labels]
        # get success sweeps
        success_sweeps = np.arange(1,len(labels)+1)[isuccess]
        print("success sweeps: ",success_sweeps)
    else:
        print("!!!! Label file not found !!!!!!")
        break
    try:
        badsweeps = np.array([float(elm)-1 for elm in badsweeps],dtype=np.int)
    except:
        badsweeps = None
    print("Opening: ",abffullpath)
    print(badsweeps)
    ephys = EphysClass(abffullpath,loaddata=True,badsweeps=badsweeps,vclamp = int(vclamp), cclamp = None,fid=fileid)
    ephys.extract_stim_props(trgchannel)
    ephys.extract_res_props(reschannel,trgchannel)
    ephys.extract_indices_holdingsteps(vclampchannel,NstepsNeg=1,NstepsPos=1)
    ephys.info()    
    # ephys.show([0],[])           # [channels],[sweeps]
    # create folder for each cell
    figpath = mainpath + os.path.splitext(os.path.basename(abfname))[0] +'/'
    print("Figure path: ",figpath)
    if(not os.path.exists(figpath)):
        os.mkdir(figpath)
    # ----------------    
    lags,betas,peaks = ephys.findpeaks_template_match(reschannel,trgchannel,success_sweeps,figpath)
    nsweeps = len(ephys.sweeps)
    isi = ephys.isi
    for isweep in np.arange(0,betas.shape[0]):
        for istim in np.arange(0,len(betas[isweep])):
            record = {"ephysfile":abfname,"cellid":os.path.splitext(os.path.basename(abfname))[0],"fileid":fileid}
            record.update({"nsweeps":nsweeps,"isweep": isweep+1, "isi":isi,"istim":istim,"vclamp":vclamp})
            record.update({"tlag":lags[isweep,istim],"beta":betas[isweep,istim],"peak":peaks[isweep,istim]})
            roidf = roidf.append(record,ignore_index = True) # append a record per stim
            # ----------
    print(roidf)
# -----------------
# save the dataframe
dfname = "ephys_analysis_minstim_V2.csv"
roidf.to_csv(os.path.join(mainpath,dfname),index=False)
