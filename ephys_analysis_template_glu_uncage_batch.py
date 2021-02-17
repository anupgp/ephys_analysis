from ephys_class import EphysClass
import pandas as pd
import re
import os
import numpy as np
from itertools import chain
import datetime
from matplotlib import pyplot as plt

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
# open pandas dataframe to populate data
roidf = pd.DataFrame()
fileid=0
# open template file
template = pd.read_csv(templatefilepath,header=None)
template.columns = ["t","y"]
for row in range(0,masterdf.shape[0]):
# for row in range(8,9):
    abffile = str(masterdf.loc[row,"ephys_file"])
    ophysfile = str(masterdf.loc[row,"ophys_file"])
    if(re.search(r'^.*nan.*$',abffile) is not None):
        print('!!!! Warning: empty cell !!!!')
        print('Skipping to the next row')
        continue
    uncagepower = round(masterdf.loc[row,"power"])
    clamp = str(masterdf.loc[row,"clamp"])
    sex = str(masterdf.loc[row,"sex"])
    dob = str(masterdf.loc[row,"dob"])
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
    # expdate = datetime.datetime.strptime(re.search('[0-9]{8,8}?',abffile)[0],"%Y%m%d")
    expdate = re.search('[0-9]{8,8}?',abffile)[0]
    # extract spineid
    spineid = re.sub("\/","_",re.search('\/[0-9]{8,8}?\/.*?\/',abffile)[0][1:-1])+"S"+str(spinecount)
    # abffname = re.search('[^/]+.\.*$',abffile)[0] # filename without extension
    # abffname = re.sub(' ','_',abffname)
    print('ABF filename:\t',abffile)
    print('Expdate:\t',expdate)
    print('Spineid:\t',spineid)
    print('Clamp:\t',clamp)
    # create ephys object
    ephys1 = EphysClass(abffile,loaddata=True,badsweeps=badsweeps)
    ephys1.info()
    ephys1.get_stimprops(trgchannel)
    ephys1.get_signal_props(reschannel,trgchannel)
    fh = None
    fht = None
    # skip if no good sweeps found
    if (len(ephys1.sweeps) == 0):
        print('No good sweeps in the file!')
        continue
    if (re.search(".*CC.*",clamp)):
        ephys1.seriesres_currentclamp(reschannel,clampchannel)
        [tlags,ypeaks,fh,ah] = ephys1.find_peaks(reschannel,trgchannel,plotdata=False,peakdir="+")
        [tlagst,ypeakst,fht,aht] = ephys1.template_match_avg(reschannel,clampchannel,peakdir="+")
    if (re.search(".*VC.*",clamp)):
        [tlags,ypeaks,fh,ah] = ephys1.find_peaks(reschannel,trgchannel,plotdata=False,peakdir="-")
        tlagst = np.zeros(tlags.shape)
        ypeakst = np.zeros(ypeaks.shape)
    # create folder for each spineid
    if(not os.path.exists(masterfile_path+spineid+'/')):
        os.mkdir(masterfile_path+spineid+'/')
    if(fh is not None):
    # save figure find_peaks
        figname = spineid+'_'+clamp+'_'+ephys1.fid+'_find_peaks'+'.png'
        fh.savefig(masterfile_path+spineid+'/'+figname,dpi=300)
    if(fht is not None):
    # save figure template_match
        figname = spineid+'_'+clamp+'_'+ephys1.fid+'_template_match'+'.png'
        fht.savefig(masterfile_path+spineid+'/'+figname,dpi=300)
    # --------------
    isi = ephys1.isi
    sres = ephys1.sres
    sres = sres.mean()
    nsweeps = len(ephys1.sweeps)
    print(tlags,ypeaks)
    # ephys1.show([0,1,2,3],[1])           # [channels],[sweeps]
    # create the record
    for istim in np.arange(0,len(ypeaks)):
        record = {"ephysfile":abffile,"ophysfile":ophysfile,"doe":expdate,"spineid":spineid,"fileid":fileid}
        record.update({"dob":dob,"sex":sex,"clamp":clamp,"uncagepower":uncagepower})
        record.update({"nsweeps":nsweeps,"isi":isi,"istim":istim})
        record.update({"tlag":tlags[istim],"peak":ypeaks[istim],"sres":sres})
        record.update({"tlagt":tlagst[istim],"peakt":ypeakst[istim]})
        print(record)
        roidf = roidf.append(record,ignore_index = True) # append a record per stim
    fileid = fileid+1
    # plt.show()
# -----------------
# save the dataframe
dfname = "ephys_analysis_uncage_pairedpulse_cclampV3.csv"
roidf.to_csv(masterfile_path+dfname,index=False)
