# Save all responses starting a predefined time before the stimulus to a fixed time iterval into a csv file
# the header will have information on the name of the file and voltage clamp

from ephys_class import EphysClass
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import re


# get all abffile names from the drive

mainpath = '/Volumes/GoogleDrive/Shared drives/Beique Lab/CURRENT LAB MEMBERS/Anup Pillai/ephys_mininum_stim/'
rawdatafolder = "raw_data/"
masterfilename = "minstim_summary_anup.xlsx"
if(not os.path.exists(mainpath)):
    print("Data path not found!")
    exit
# -------------------
masterdf = pd.read_excel(os.path.join(mainpath+masterfilename),sheet_name=1,header=0,use_cols=4)
colnames = list(masterdf.columns)
reschannel = "ImRight"
trgchannel = "IN9"
vclampchannel = "VmRight"
fcount=0
for row in range(0,masterdf.shape[0]):
    colnames = []
    abfname = str(masterdf.loc[row,"abffile"])
    abffullpath = os.path.join(mainpath,rawdatafolder,abfname)
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
    print('fileid: ',fileid)
    try:
        badsweeps = np.array([float(elm)-1 for elm in badsweeps],dtype=np.int)
    except:
        badsweeps = None
    # ------------------
    print("Opening: ",abffullpath)
    ephys = EphysClass(abffullpath,loaddata=True,badsweeps=badsweeps)
    ephys.extract_stim_props(trgchannel)
    ephys.extract_res_props(reschannel,trgchannel)
    ephys.extract_indices_holdingsteps(vclampchannel,NstepsNeg=1,NstepsPos=1)
    ephys.extract_indices_stims(trgchannel,Nstims=1)
    # ephys.info()    
    # ephys.show([],[1,2,3,4,5])           # [channels],[sweeps]
    t,y = ephys.extract_response(reschannel,trgchannel,-0.01,0.1,min_isi=0)
    colnames.append("t")
    # fh = plt.figure()
    # ah = fh.add_subplot(111)
    # ah.plot(t,y)
    # plt.show()
    # -----------
    [colnames.append(fileid+"_sweep"+str(sweep)) for sweep in ephys.sweeps]
    # print(colnames)
    # open pandas dataframe to populate data
    roidf = pd.DataFrame(columns=colnames,data=np.concatenate((t,y),axis=1))
    roidf.to_csv(os.path.join(mainpath,fileid)+".csv")
    # print(roidf)
    # input('key')

    
    

