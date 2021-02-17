import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ephys_class import EphysClass
import os
import re
from scipy.signal import find_peaks

rawdatapath = '/Volumes/GoogleDrive/Shared drives/Beique Lab DATA/Imaging Data/LSM880 2P/Anup/'
analysisdatapath = '/Users/macbookair/goofy/data/beiquelab/pfc_project'
masterfile = 'lsm880_holo_uncaging_good_cells.xlsx'
reschannel = "ImLeftP"
trgchannel = "trigger"
vclampchannel = "VmLeftS"
# abffile = "2019_08_08_0037.abf"
masterdf = pd.read_excel(os.path.join(analysisdatapath,masterfile),sheet_name=0,header=0,use_cols=4)
print(masterdf)
yy = []
for row in range(0,masterdf.shape[0]):
    abfname = str(masterdf.loc[row,"abffile"])
    cell = str(masterdf.loc[row,"cell"])
    date = str(masterdf.loc[row,"date"])
    nsweeps = int(masterdf.loc[row,"nsweeps"])
    goodsweeps = [int(item) for item in str(masterdf.loc[row,"sweeps"]).split(',')]
    badsweeps = [elm for elm in np.arange(0,nsweeps) if np.isin(elm,goodsweeps,invert=True)]
    print("goodsweeps: {}".format(goodsweeps))
    print("badsweeps: {}".format(badsweeps))
    abfnamefull = os.path.join(rawdatapath,date,cell,abfname)
    print(abfnamefull)
    ephys = EphysClass(abfnamefull,loaddata=True,badsweeps=badsweeps)
    #     ephys.show([],[])           # [channels],[sweeps]
    ephys.extract_stim_props(trgchannel)
    ephys.extract_res_props(reschannel,trgchannel)
    ephys.extract_indices_holdingsteps(vclampchannel,NstepsNeg=1,NstepsPos=1)
    ephys.info()
    t,y = ephys.extract_response(reschannel,trgchannel,tpre=-0.0,tpost=0.15)
    yy.append(y)
    # ---------------
yy = np.concatenate(yy,axis=1)
print(yy.shape)
yya = []
# align traces to peak
ipeaks = np.zeros(yy.shape[1],dtype=np.int)
for icol in np.arange(0,yy.shape[1]):
    ipeaksall,_ = find_peaks(-yy[:,icol],prominence=(0.5,None),width=(100,None),height=(15,None))
    ipeaks[icol] = ipeaksall[0]
    
    # ---------
print(ipeaks)
for icol in np.arange(0,yy.shape[1]):
    ileft = int(ipeaks[icol]-400)
    iright = int(ipeaks[icol]+1500)
    print(ileft,iright)
    yya.append(yy[ileft:iright,icol][np.newaxis].T)
    
yya = np.concatenate(yya,axis=1)
ta = np.arange(0,yya.shape[0]*ephys.si,ephys.si)
# ------------------------

fh = plt.figure()
ah = fh.add_subplot(111)
ah.axis('off')
ah.plot(ta,yya,color="gray")
# ah.plot(t[ipeaks],yy[ipeaks,0],'o')
ah.plot(ta,np.mean(yya,axis=1),color='k',linewidth=2)
# plot scale bars
ah.vlines(0.08,-30,-40,linewidth=2,color='k')
ah.plot(0.007,5,marker=7,markersize=18,color='k')
ah.text(0.075,-40,"10 pA", rotation=90,fontsize=14)
ah.hlines(-40,0.08,0.09,linewidth=2,color='k')
ah.text(0.077,-45,"10 ms", rotation=0,fontsize=14)
# save figure
figtitle = "lsm880_rapp_opto_holo_uncaging_epscs.eps"
plt.savefig(os.path.join(analysisdatapath,figtitle),dpi=600,transparent=True,format="eps")
plt.show()


