# analysis of single EPSCs from glutamate uncaging at spines on the apical and basal dendrites of hipocampal CA1 and PFC L5 neurons
# AMPAR desensatization with Cyclothiazide

from ephys_class import EphysClass
import ephys_class
import ephys_functions as ephysfuns
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import re
import math


path_ephysdata="/home/anup/gdrive-beiquelabdata/Ephys Data/Olympus 2P/Anup/"
path_ophysdata="/home/anup/gdrive-beiquelabdata/Imaging Data/Olympus 2P/Anup/"
# masterdfpath = "/home/anup/gdrive-beiquelab/CURRENT LAB MEMBERS/Anup Pillai/ca1_uncaging_cluster/"
path_masterdf="/home/anup/goofy/data/beiquelab/amparDesen"
fname_masterdf = "amparDesen_master_file.xlsx"
dfname = "ephys_analysis_epscs_deconvolve_template_match_ampar_desen.csv"
# dfname = "ephys_analysis_epsp_clustered_input_hcca1.csv"
usecols = ["selectfile","badsweeps","blocker","expdatefolder","dob","gender","region","area","compartment","cellfolder","fileid","neuronid","dendriteid","spineid","clampmode","nstims","isi","laserstimpower","accessres"]
masterdf = pd.read_excel(os.path.join(path_masterdf,fname_masterdf),header=0,usecols=usecols,index_col=None,sheet_name=0)
masterdf = masterdf.dropna(how='all') # drop rows with all empty columns
masterdf = masterdf.reset_index(drop=True)
# create a unique spine name
masterdf["spinename"] = join_df_columns_with_separator(masterdf,['expdatefolder','blocker','cellfolder','neuronid','dendriteid','spineid'])
spines = masterdf["spinename"].unique()
# isis_template = [100,200,400,600,800,1000] # Only sweeps with these ISI's are used for creating the template
isis_template = [1000,800,600,400,200,100] # Only sweeps with these ISI's are used for creating the template
reschannel = "ImLEFT"
clampchannel = "IN1"
stimchannel = "IN7"
prestim = 0.00                # trace duration prestim
poststim = 0.07                 # trace duration poststim
stimdur = poststim-prestim      # trace duration
roidf = pd.DataFrame()
# ----------------------
for ispine in range(len(spines)):
# for ispine in range(7,8):
    dfspine = masterdf.loc[masterdf["spinename"] == spines[ispine]]
    dfspine = dfspine.reset_index()
    # generate template trace
    fileidstemp =  list(dfspine[dfspine["isi"].isin(isis_template)]["fileid"]) # fileids of all sweeps with isi in isis_template for each spine
    print("*** starting to create template for spine: {}, from fileids: {}".format(spines[ispine],fileidstemp))
    traces = []
    for itfile in range(len(fileidstemp)):
        templateid = fileidstemp[itfile]
        select = list(dfspine.loc[dfspine["fileid"] == templateid,"selectfile"].astype(int))[0]
        # if (~select):
        #     continue
        # # }
        fname = "".join((list(dfspine.loc[dfspine["fileid"] == templateid,"fileid"].astype(str))[0],".abf"))
        cellfolder = list(dfspine.loc[dfspine["fileid"] == templateid,"cellfolder"].astype(str))[0]
        expdatefolder = list(dfspine.loc[dfspine["fileid"] == templateid,"expdatefolder"].astype(int).astype(str))[0]
        fnamefull = os.path.join(path_ephysdata,expdatefolder,cellfolder,fname)
        # badsweeps = dfspine.loc[dfspine["fileid"] == templateid,"badsweeps"]
        # badsweeps = masterdf.loc[row,"badsweeps"]
        # if (math.isnan(badsweeps)):
        #     badsweeps = []
        # print(badsweeps)
        # input()
        print("fileid: {}, filename: {}, cellfolder: {}, expdate: {}".format(templateid,fname,cellfolder,expdatefolder))
        ephys = EphysClass(fnamefull,loaddata=True)
        # print(ephys.goodsweeps)
        ephys.extract_indices_holdingsteps(clampchannel,1,1)
        ephys.extract_stim_props(stimchannel)
        ephys.extract_res_props(reschannel,stimchannel)
        # print(ephys.stimprop)
        # ephys.show([0,1,3],[])           # [channels],[sweeps]
        t,y = ephys.extract_response(reschannel,stimchannel,prestim,poststim) # get individual response traces from each sweep & stimulation
        traces.append(y)
    # }
    # extract traces into a numpy array
    tracelen = min([trace.shape[0] for trace in traces]) # # find the smallest trace length in the list
    tracenum = np.array([trace.shape[1] for trace in traces]).sum()
    print(tracenum,tracelen)
    traces = [trace[0:tracelen,:] for trace in traces]
    traces = np.concatenate(traces,axis=1)
    temptrace = traces.mean(axis=1).reshape(traces.shape[0],1)
    temptrace = ephys_class.smooth(temptrace[:,0],windowlen=31,window='hanning')[:,np.newaxis]
    print("template trace for spine: {} completed".format(spines[ispine]))
    # ----------------
    # plot all extracted traces from each spine
    # fh = plt.figure()
    # ah = fh.add_subplot(111)
    # figtitle = "".join(("date: ",expdatefolder,": cellfolder: ",cellfolder,", templateid: ",templateid))
    # ah.set_title(figtitle)
    # ah.plot(t[0:tracelen],traces)
    # ah.plot(t[0:tracelen],temptrace,color='k')
    # plt.show()
    # ----------------
    # template matching on epscs
    # get all fileids for each spine
    fileids =  list(dfspine["fileid"]) # fileids of all sweeps for each spine
    print(fileids)
    for ifileid in range(len(fileids)):
        fileid = fileids[ifileid]
        select = list(dfspine.loc[dfspine["fileid"] == fileid,"selectfile"].astype(int))[0]
        if (select == 0):
            print("select: skipping file: {}".format(select,fileid))
            continue
        # }
        print("*** starting template match to trace : {} from spine: {}".format(fileid,spines[ispine]))
        fname = "".join((list(dfspine.loc[dfspine["fileid"] == fileid,"fileid"].astype(str))[0],".abf"))
        cellfolder = list(dfspine.loc[dfspine["fileid"] == fileid,"cellfolder"].astype(str))[0]
        expdatefolder = list(dfspine.loc[dfspine["fileid"] == fileid,"expdatefolder"].astype(int).astype(str))[0]
        fnamefull = os.path.join(path_ephysdata,expdatefolder,cellfolder,fname)
        print("fileid: {}, filename: {}, cellfolder: {}, expdate: {}".format(fileid,fname,cellfolder,expdatefolder))
        print("fnamefull: {}".format(fnamefull))
        ephys = EphysClass(fnamefull,loaddata=True)
        ephys.extract_indices_holdingsteps(clampchannel,1,1)
        ephys.extract_stim_props(stimchannel)
        ephys.extract_res_props(reschannel,stimchannel)
        y = ephys.data[0,0,:]
        y = y.reshape((len(y),1))
        t = np.arange(0,(len(y))*ephys.si,ephys.si).reshape(len(y),1)
        fh,ah,peaks = ephys.deconvolve_epscs_template_match(reschannel,stimchannel,t[0:tracelen],temptrace)
        # ------------------------
        isi = ephys.isi
        nstims = int(ephys.nstim)
        nsweeps = int(ephys.nsweeps)
        neuronid = int(dfspine.loc[0,"neuronid"])
        dendriteid = int(dfspine["dendriteid"][0])
        spineid = int(dfspine["spineid"][0])
        clampmode = dfspine["clampmode"][0]
        gender = dfspine["gender"][0]
        region = dfspine["region"][0]
        area = dfspine["area"][0]
        blocker = dfspine["blocker"][0]
        compartment = dfspine["compartment"][0]
        dob = int(dfspine["dob"][0])
        stimpower = float(dfspine["laserstimpower"][0])
        print(dob,expdatefolder,neuronid,dendriteid,spineid,isi,nsweeps,nstims,stimpower)
        # ----------------------
        # save the figure
        # figsavepath = "/home/anup/gdrive-beiquelab/CURRENT LAB MEMBERS/Anup Pillai/ampar_desen/"
        figsavepath = "/home/anup/goofy/data/beiquelab/amparDesen/trace_figures/"
        figtitle = "".join((ah.get_title(),"_",region,"_",area,"_",compartment,"_",blocker))
        ah.set_title(figtitle)
        figname= "".join((ah.get_title(),".png"))
        print(figname)
        fh.savefig(os.path.join(figsavepath,figname),format="png",dpi=300)
        # plt.show()
        plt.close('all')
        # ------------------
        for isweep in range(nsweeps):
            for istim in range(nstims+1):
                record = {"ephysfile":fname,"neuronid":neuronid}
                record.update({"blocker":blocker,"expdate":expdatefolder,"dob":dob})
                record.update({"gender":gender,"region":region,"area":area,"compartment":compartment})
                record.update({"cellfolder":cellfolder,"ephysfile":fname})
                record.update({"neuronid":neuronid,"dendriteid":dendriteid,"spineid":spineid,"clampmode":clampmode})
                record.update({"nsweeps":nsweeps,"nstims":nstims,"isi":isi})
                record.update({"isweep": isweep+1, "isi":isi,"istim":istim}) # stim0: noise
                record.update({"stimpower":stimpower})
                record.update({"peak":peaks[istim,isweep]})
                # record.update({"accessres":accessres/1E6,"tau":tau*1E3})
                roidf = roidf.append(record,ignore_index = True) # append a record per stim
            # }
        # }
        print(record)
        # input()
    # }
# -----------------
# save the dataframe
roidf.to_csv(os.path.join(path_masterdf,dfname),index=False)
# }
