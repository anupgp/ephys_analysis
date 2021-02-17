# analysis of single EPSCs from glutamate uncaging at spines on the apical and basal dendrites of hipocampal CA1 and PFC L5 neurons
# AMPAR desensatization with Cyclothiazide

from ephys_class import EphysClass
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import re
import math

def join_df_columns_with_separator(df,columns):
    converted = pd.DataFrame()
    for i in range(len(columns)):
        if(df[columns[i]].dtype.kind in 'biufc'):
            print(columns[i],' is ', 'not string')
            converted[columns[i]] = df[columns[i]].astype(int).astype(str)
        else:
            print(columns[i],' is ', 'string')
            converted[columns[i]] = df[columns[i]]
    joined = converted[columns].agg('_'.join,axis=1)
    joined=joined.rename("joined")
    return(joined)

path_ephysdata="/home/anup/gdrive-beiquelabdata/Ephys Data/Olympus 2P/Anup/"
path_ophysdata="/home/anup/gdrive-beiquelabdata/Imaging Data/Olympus 2P/Anup/"
# masterdfpath = "/home/anup/gdrive-beiquelab/CURRENT LAB MEMBERS/Anup Pillai/ca1_uncaging_cluster/"
path_masterdf="/home/anup/goofy/data/beiquelab/amparDesen"
fname_masterdf = "amparDesen_master_file.xlsx"
# dfname = "ephys_analysis_epsp_clustered_input_hcca1.csv"
usecols = ["selectfile","badsweeps","blocker","expdatefolder","dob","gender","region","area","compartment","cellfolder","fileid","neuronid","dendriteid","spineid","clampmode","nstims","isi","laserstimpower","accessres"]
masterdf = pd.read_excel(os.path.join(path_masterdf,fname_masterdf),header=0,usecols=usecols,index_col=None,sheet_name=0)
masterdf = masterdf.dropna(how='all') # drop rows with all empty columns
# create a unique spine name
masterdf["spinename"] = join_df_columns_with_separator(masterdf,['expdatefolder','blocker','cellfolder','neuronid','dendriteid','spineid'])
spines = masterdf["spinename"].unique()
isis_template = [100,200,400,600,800,1000]
reschannel = "ImLEFT"
clampchannel = "IN1"
stimchannel = "IN7"
# ----------------------
for ispine in range(len(spines)):
    print(spines[ispine])
    dfspine = masterdf.loc[masterdf["spinename"] == spines[ispine]]
    print(dfspine)
    # generate template trace
    template_fileids =  list(dfspine[dfspine["isi"].isin(isis_template)]["fileid"])
    template = []
    for itfile in range(len(template_fileids)):
        templateid = template_fileids[itfile]
        fileid = "".join((list(dfspine.loc[dfspine["fileid"] == templateid,"fileid"].astype(str))[0],".abf"))
        cellfolder = list(dfspine.loc[dfspine["fileid"] == templateid,"cellfolder"].astype(str))[0]
        expdatefolder = list(dfspine.loc[dfspine["fileid"] == templateid,"expdatefolder"].astype(int).astype(str))[0]
        print(templateid,fileid,cellfolder,expdatefolder)
        filename = os.path.join(path_ephysdata,expdatefolder,cellfolder,fileid)
        print(filename)
        ephys = EphysClass(filename,loaddata=True)
        print(ephys.goodsweeps)
        ephys.extract_indices_holdingsteps(clampchannel,1,1)
        ephys.extract_stim_props(stimchannel)
        ephys.extract_res_props(reschannel,stimchannel)
        print(ephys.stimprop)
        # input()
        # ephys.show([0,1,3],[])           # [channels],[sweeps]
        ephys.extract_response(reschannel,stimchannel,-0.005,0.06)
        # fh = plt.figure()
        # ah = fh.add_subplot(111)
        # ah.plot(t,y)
        # plt.show()
