from ephys_class import EphysClass
import pandas as pd
import re
import os
import numpy as np

# read masterfile in Microsoft_excel
masterfile_path = '/Users/macbookair/goofy/data/beiquelab/glu_uncage_ca1/'
# datapath = '/Volumes/Anup_2TB/raw_data/beiquelab/o2p/'
datapath = '/Users/macbookair/goofy/data/beiquelab/rawdata/'
templatefilepath = '/Users/macbookair/goofy/data/beiquelab/glu_uncage_ca1/glu_uncage_avg_response.csv'
masterfname = masterfile_path + 'glu_uncage_ca1_1spine_mastersheet.xlsx'
masterdf = pd.read_excel(masterfname,header=0,use_cols=10)
colnames = list(masterdf.columns)
reschannel = "ImLEFT"
clampchannel = "IN1"
trgchannel = "IN7"
istep = 30e-12
fcount=0

# open template file
template = pd.read_csv(templatefilepath,header=None)
template.columns = ["t","y"]
for row in range(0,masterdf.shape[0]):
# for row in range(17,18):
# for row in range(3,4):
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
    try:
        badsweeps = np.array([float(elm)-1 for elm in badsweeps],dtype=np.int)
    except:
        badsweeps = -1
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
    # create ephys object
    ephys1 = EphysClass(abffile,loaddata=True,badsweeps=badsweeps)
    ephys1.info()
    ephys1.get_stimprops(trgchannel)
    ephys1.get_signal_props(reschannel,trgchannel)
    # ephys1.average_trace(reschannel,trgchannel,0.01)
    # ephys1.template_match(reschannel,trgchannel,template["t"],template["y"])
    [tpeaks,tlags,ypeaks] = ephys1.find_peaks(reschannel,trgchannel)
    print(tpeaks,tlags,ypeaks)
    # ephys1.show([0,1,2,3],[1])           # [channels],[sweeps]
    # ephys1.get_stimprops(trgchannel)
    # ephys1.get_signal_props(reschannel,trgchannel)
    # sweeps = np.setdiff1d(np.arange(0,ephys1.nsweeps),badsweeps)
    # ephys1.info()
    # ephys1.seriesres_currentclamp(reschannel,clampchannel)
    # print('Series resistance = ','\t',ephys1.sres)

