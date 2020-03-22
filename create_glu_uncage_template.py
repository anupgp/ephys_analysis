# select all epsp's from all the files and create an averaged trace for template matching

from ephys_class import EphysClass
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import re
import os

# read masterfile in Microsoft_excel
masterfile_path = '/Users/macbookair/goofy/data/beiquelab/glu_uncage_ca1/'
# datapath = '/Volumes/Anup_2TB/raw_data/beiquelab/o2p/'
datapath = '/Users/macbookair/goofy/data/beiquelab/rawdata/'
masterfname = masterfile_path + 'glu_uncage_ca1_1spine_mastersheet.xlsx'
masterdf = pd.read_excel(masterfname,header=0,use_cols=10)
colnames = list(masterdf.columns)
reschannel = "ImLEFT"
trgchannel = "IN7"
fcount=0
yy = []
timewindow = 0.2
min_isi = 0.2
fh = plt.figure()
ah1 = plt.subplot(111)
for row in range(0,masterdf.shape[0]):
    abffile = str(masterdf.loc[row,"ephys_file"])
    if(re.search(r'^.*nan.*$',abffile) is not None):
        print('!!!! Warning: empty cell !!!!')
        print('Skipping to the next row')
        continue
    power = round(masterdf.loc[row,"power"])
    clamp = str(masterdf.loc[row,"clamp"])
    gender = str(masterdf.loc[row,"gender"])
    spinecount = int(round(masterdf.loc[row,"spine"]))
    badsweeps = str(masterdf.loc[row,"badsweeps"]).split(",")
    print("badsweeps as string:\t",badsweeps)
    try:
        badsweeps = np.array([float(elm)-1 for elm in badsweeps],dtype=np.int)
        print("badsweeps as int:\t",badsweeps)
    except:
        badsweeps = -1
    print("badsweep:\t",badsweeps)
    # add path to filename
    abffile = datapath+abffile
    # check is linescan_timeseries file exists
    if(not os.path.exists(abffile)):
        print('abffile file does not exist!')
        print('Skipping to the next file')
        continue
    # extract date
    expdate = re.search('[0-9]{8,8}?',abffile)[0]
    # extract spineid
    spineid = re.sub("\/","_",re.search('\/[0-9]{8,8}?\/.*?\/',abffile)[0][1:-1])+"S"+str(spinecount)
    # filename
    # abffname = re.search('[^/]+.\.*$',abffile)[0] # filename without extension
    # abffname = re.sub(' ','_',abffname)
    print('ABF filename:\t',abffile)
    print('Expdate:\t',expdate)
    print('Spineid:\t',spineid)
    print('Clamp:\t',clamp)
    if (not re.search(".*CC.*",clamp)):
        continue
    ephys1 = EphysClass(abffile,loaddata=True)
    ephys1.get_stimprops(trgchannel)
    ephys1.get_signal_props(reschannel,trgchannel)
    sweeps = np.setdiff1d(np.arange(0,ephys1.nsweeps),badsweeps)
    print("Bad sweeps:\t",badsweeps)
    print("Sweeps:\t",sweeps)
    t = np.arange(0,timewindow,ephys1.si)
    y = ephys1.extract_response(reschannel,trgchannel,timewindow,min_isi)
    plotcolors = ['red','green','blue','gray']
    if((len(y)>0) and len(sweeps)>0):
        y = y-y[0,:]
        print(y.shape)
        yy.append(y[:,sweeps])
        # fh = plt.figure()
        # ah1 = plt.subplot(111)
        # ah1.set_title(re.search("[^/]+.abf",abffile)[0])
        # for sweep in sweeps:
        #     ah1.plot(t,y[:,sweep],color=plotcolors[sweep])
        # plt.show()
    # -----------------------
yy = np.concatenate(yy,axis=1)
print(yy.shape)
print(t.shape,yy.shape)
ah1.plot(t,yy)
ah1.plot(t,yy.mean(axis=1),color='k')
plt.show()
# save all the traces and the averaged response
glu_uncage_avg_response_fname = masterfile_path + 'glu_uncage_avg_response.csv'
np.savetxt(glu_uncage_avg_response_fname,np.concatenate((t[:,np.newaxis],yy.mean(axis=1)[:,np.newaxis]),axis=1),delimiter=',')
glu_uncage_all_responses_fname = masterfile_path + 'glu_uncage_all_responses.csv'
np.savetxt(glu_uncage_all_responses_fname,np.concatenate((t[:,np.newaxis],yy),axis=1),delimiter=',')
