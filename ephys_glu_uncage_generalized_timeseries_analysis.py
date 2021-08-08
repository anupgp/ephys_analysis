# A generalized script for ephys batch analysis
# Experiments tested: glutamate uncaging - single spine or cluster of spines
# patch clamp modes: works for both  voltage and current clamp recording
# Stimulus types: single stim or mulitple stims

from ephys_class import EphysClass
import ephys_class
import ephys_functions as ephysfuns
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import re
import math

# paths to ephys data
path_ephysdata="/home/anup/gdrive-beiquelabdata/Ephys Data/Olympus 2P/Anup/" # common to all ephys data
# -------------------------
# Ca1 single spine
# path to ephys data master excelfile
path_masterdf="/home/anup/goofy/data/beiquelab/hc_ca1_glu_uncage_spine_single_epsp/" # local to the ephys project: Ca1 single spine
file_masterdf = "hc_ca1_glu_uncage_spine_single_mastersheet.xlsx" # local to the ephys project
dfname = "hc_ca1_glu_uncage_spine_single_masterdata.csv"           # local to the ephys project
isi_template_vc = [0,500,800,1000] # Only sweeps with these ISI's are used for creating the template
isi_template_cc = [600,800,1000] # Only sweeps with these ISI's are used for creating the template
# --------------------------
# Ca1 cluster spine
# path_masterdf="/home/anup/goofy/data/beiquelab/hc_ca1_glu_uncage_spine_cluster_epsp/" # local to the ephys project:  
# file_masterdf = "hc_ca1_glu_uncage_spine_cluster_mastersheet.xlsx" # local to the ephys project
# dfname = "hc_ca1_glu_uncage_spine_cluster_masterdata.csv"           # local to the ephys project
# isi_template = [200,300,400,500,600,800,1000] # Only sweeps with these ISI's are used for creating the template
# ------------------------
# pfc sluster spine
# path_masterdf="/home/anup/goofy/data/beiquelab/pfc_l5_glu_uncage_spine_cluster_epsp/" # local to the ephys project:  
# file_masterdf = "pfc_l5_glu_uncage_spine_cluster_mastersheet.xlsx" # local to the ephys project
# dfname = "pfc_l5_glu_uncage_spine_cluster_masterdata.csv"           # local to the ephys project
# isi_template = [200,300,400,500,600,800,1000] # Only sweeps with these ISI's are used for creating the template
# ------------------------
path_fig = os.path.join(path_masterdf,"figures")
# -----------------------
# specify columns to load
loadcolumns1 = ["expdate","dob","sex","animal","strain","region","area","side","group","neuronid","dendriteid","spineid"]
loadcolumns2 = ["unselectfile","badsweeps","ephysfile","clamp","stimpower","isi"]
loadcolumns = loadcolumns1 + loadcolumns2
masterdf = pd.read_excel(os.path.join(path_masterdf,file_masterdf),header=0,usecols=loadcolumns,index_col=None,sheet_name=0)
masterdf = masterdf.dropna(how='all') # drop rows with all empty columns
masterdf = masterdf.reset_index(drop=True) # don't increment index for empty rows
# remove files that are marked to unselect from analysis
masterdf = masterdf[masterdf["unselectfile"]!=1]
# create a unique spine name
masterdf["spinename"] = ephysfuns.combine_columns(masterdf,['expdate','group','neuronid','dendriteid','spineid'])
spines = masterdf["spinename"].unique()
print(masterdf)
print(spines)
roidf = pd.DataFrame()
# ----------------------------------
# provide channel names
reschannel = "ImLEFT"
clampchannel = "IN1"
stimchannel = "IN7"
ephysfile_ext = "abf"
prestim_cc = 0.0                # trace duration prestim
poststim_cc = 0.1                 # trace duration poststim
stimdur_cc = poststim_cc-prestim_cc      # trace duration
prestim_vc = 0.0                         # trace duration prestim
poststim_vc = 0.05                 # trace duration poststim
stimdur_vc = poststim_vc-prestim_vc      # trace duration
# processing each spine
# for ispine in range(0,len(spines)):
for ispine in range(0,len(spines)):
    print("spine: {}".format(spines[ispine]))
    dfspine = masterdf.loc[masterdf["spinename"] == spines[ispine]]
    dfspine = dfspine.reset_index()
    print(dfspine)
    # get ephysfiles for generating the template trace from a set of isi defined by isi_template
    templatefiles_cc =  dfspine[(dfspine["clamp"] == "cc") & (dfspine["isi"].isin(isi_template_cc))]["ephysfile"].values
    templatefiles_vc =  dfspine[(dfspine["clamp"] == "vc") & (dfspine["isi"].isin(isi_template_vc))]["ephysfile"].values
    print("Voltage clamp template traces for spine: {} - {}".format(spines[ispine],templatefiles_vc))
    print("Current clamp template traces for spine: {} - {}".format(spines[ispine],templatefiles_cc))
    if (len(templatefiles_cc)>0):
        print("*** starting to create current clamp template for spine: {}\nFileids: {}".format(spines[ispine],templatefiles_cc))
        traces = []
        for itfile in range(len(templatefiles_cc)):
            tfile = templatefiles_cc[itfile]
            neuronid = list(dfspine.loc[dfspine["ephysfile"] == tfile,"neuronid"].astype(int).astype(str))[0]
            expdate = list(dfspine.loc[dfspine["ephysfile"] == tfile,"expdate"].astype(int).astype(str))[0]
            fnamefull = os.path.join(path_ephysdata,expdate,"".join(("C",neuronid)),tfile)
            print(fnamefull)
            badsweeps = list(dfspine.loc[dfspine["ephysfile"] == tfile,"badsweeps"].astype(str))[0].split(',')
            badsweeps = [int(item)-1 for item in badsweeps if item != 'nan'] # trace 1 has index 0
            print("badsweeps: ",badsweeps)
            ephys = EphysClass(fnamefull,loaddata=True,badsweeps=badsweeps,clampch=clampchannel,resch=reschannel,stimch=stimchannel)
            ephys.extract_indices_holdingsteps(1,1)
            ephys.extract_stim_props()
            # ephys.extract_res_props(reschannel,stimchannel)
            ephys.info()
            t,y = ephys.extract_response(reschannel,stimchannel,prestim_cc,poststim_cc) # get individual response traces from each sweep & stimulation
            traces.append(y)
            # ephys.show([0,3],ephys.goodsweeps)
        # }
        # extract traces into a numpy array
        tracelen = min([trace.shape[0] for trace in traces]) # # find the smallest trace length in the list
        tracenum = np.array([trace.shape[1] for trace in traces]).sum()
        print(tracenum,tracelen)
        traces = [trace[0:tracelen,:] for trace in traces]
        traces = np.concatenate(traces,axis=1)
        temptrace_cc = traces.mean(axis=1).reshape(traces.shape[0],1)
        temptrace_cc = ephys_class.smooth(temptrace_cc[:,0],windowlen=31,window='hanning')[:,np.newaxis]
        temptrace_cc = temptrace_cc - temptrace_cc[0] # zero the trace baseline
        ttemp_cc = t[0:tracelen]
        # ---------------
        # plot all extracted traces from each spine
        fh = plt.figure()
        ah = fh.add_subplot(111)
        figtitle = "".join(("spine: ",spines[ispine]))
        ah.set_title(figtitle)
        ah.plot(ttemp_cc,traces)
        ah.plot(ttemp_cc,temptrace_cc,color='k')
        ah.hlines(ttemp_cc[0],ttemp_cc[-1],[0],linestyle='--',color='k')
        fh.savefig(os.path.join(path_fig,"".join(("template_cc",spines[ispine],".png"))),format="png",dpi=300)
        print("template cc trace for spine: {} completed".format(spines[ispine]))
    # }
    # ------------------------------
    if (len(templatefiles_vc)>0):
        print("*** starting to create voltage clamp template for spine: {}\nFileids: {}".format(spines[ispine],templatefiles_vc))
        traces = []
        for itfile in range(len(templatefiles_vc)):
            tfile = templatefiles_vc[itfile]
            neuronid = list(dfspine.loc[dfspine["ephysfile"] == tfile,"neuronid"].astype(int).astype(str))[0]
            expdate = list(dfspine.loc[dfspine["ephysfile"] == tfile,"expdate"].astype(int).astype(str))[0]
            fnamefull = os.path.join(path_ephysdata,expdate,"".join(("C",neuronid)),tfile)
            print(fnamefull)
            badsweeps = list(dfspine.loc[dfspine["ephysfile"] == tfile,"badsweeps"].astype(str))[0].split(',')
            badsweeps = [int(item)-1 for item in badsweeps if item != 'nan'] # trace 1 has index 0
            print("badsweeps: ",badsweeps)
            ephys = EphysClass(fnamefull,loaddata=True,badsweeps=badsweeps,clampch=clampchannel,resch=reschannel,stimch=stimchannel)
            ephys.extract_indices_holdingsteps(clampchannel,1,1)
            ephys.extract_stim_props(stimchannel)
            # ephys.extract_res_props(reschannel,stimchannel)
            ephys.info()
            t,y = ephys.extract_response(reschannel,stimchannel,prestim_vc,poststim_vc) # get individual response traces from each sweep & stimulation
            traces.append(y)
            # ephys.show([0,3],ephys.goodsweeps)
        # }
        # extract traces into a numpy array
        tracelen = min([trace.shape[0] for trace in traces]) # # find the smallest trace length in the list
        tracenum = np.array([trace.shape[1] for trace in traces]).sum()
        print(tracenum,tracelen)
        traces = [trace[0:tracelen,:] for trace in traces]
        traces = np.concatenate(traces,axis=1)
        temptrace_vc = traces.mean(axis=1).reshape(traces.shape[0],1)
        temptrace_vc = ephys_class.smooth(temptrace_vc[:,0],windowlen=31,window='hanning')[:,np.newaxis]
        temptrace_vc = temptrace_vc - temptrace_vc[0] # zero the trace baseline
        ttemp_vc = t[0:tracelen]
        # ----------------
        # plot all extracted traces from each spine
        fh = plt.figure()
        ah = fh.add_subplot(111)
        figtitle = "".join(("spine: ",spines[ispine]))
        ah.set_title(figtitle)
        ah.plot(ttemp_vc,traces)
        ah.plot(ttemp_vc,temptrace_vc,color='k')
        ah.hlines(ttemp_vc[0],ttemp_vc[-1],[0],linestyle='--',color='k')
        fh.savefig(os.path.join(path_fig,"".join(("template_vc",spines[ispine],".png"))),format="png",dpi=300)
        # plt.show()
        print("template vc trace for spine: {} completed".format(spines[ispine]))
    # }
    # ----------------
    # get all voltage/current clamp ephysfiles for each spine
    ephysfiles_vc =  list(dfspine[dfspine["clamp"] == "vc"]["ephysfile"])
    ephysfiles_cc =  list(dfspine[dfspine["clamp"] == "cc"]["ephysfile"])
    ephysfiles = ephysfiles_vc + ephysfiles_cc
    print(ephysfiles)
    # ephysfiles_cc = []
    # extract features using template match method
    for iephysfile in range(len(ephysfiles)):
        spineid = spines[ispine]
        ephysfile = ephysfiles[iephysfile]
        neuronid = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"neuronid"].astype(int).astype(str))[0]
        expdate = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"expdate"].astype(int).astype(str))[0]
        fnamefull = os.path.join(path_ephysdata,expdate,"".join(("C",neuronid)),ephysfile)
        print(fnamefull)
        # -----------------------
        dob = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"dob"].astype(int).astype(str))[0]
        sex = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"sex"].astype(str))[0]
        animal = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"animal"].astype(str))[0]
        strain = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"strain"].astype(str))[0]
        region = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"region"].astype(str))[0]
        area = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"area"].astype(str))[0]
        side = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"side"].astype(str))[0]
        group = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"group"].astype(str))[0]
        neuronid = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"neuronid"].astype("int").astype(str))[0]
        dendriteid = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"dendriteid"].astype("int").astype(str))[0]
        spineid = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"spineid"].astype("int").astype(str))[0]
        clamp = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"clamp"].astype(str))[0]
        stimpower = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"stimpower"].astype("int").astype(str))[0]
        # ----------
        badsweeps = list(dfspine.loc[dfspine["ephysfile"] == ephysfile,"badsweeps"].astype(str))[0].split(',')
        badsweeps = [int(float(item))-1 for item in badsweeps if item != 'nan'] # trace 1 has index 0
        print("badsweeps: ",np.array(badsweeps)+1)
        ephys = EphysClass(fnamefull,loaddata=True,badsweeps=badsweeps,clampch=clampchannel,resch=reschannel,stimch=stimchannel)
        ephys.extract_indices_holdingsteps(clampchannel,1,1)
        ephys.extract_stim_props(stimchannel)
        ephys.extract_res_props(reschannel,stimchannel)
        if (clamp == "cc"):
            # fh,ah,peaks = ephys.deconvolve_template(reschannel,stimchannel,t[0:tracelen],temptrace_cc,"pos",spines[ispine],path_fig)
            # ephys.deconvolve_template(reschannel,stimchannel,ttemp_cc,temptrace_cc,"positive",spines[ispine],path_fig)
            ephys.test()             
        # }                   
        # if (clamp == "vc"):
        #     pass
            # fh,ah,peaks = ephys.deconvolve_template(reschannel,stimchannel,t[0:tracelen],temptrace_vc,"neg",spines[ispine],path_fig)
        # }
#         nsweeps = peaks.shape[1]
#         nstims = peaks.shape[0]
#         isi = int(ephys.isi*1000)
#         print(expdate,dob,sex,animal,strain,region,area,side,group,neuronid,dendriteid,spineid,clamp,stimpower)
#         print(nsweeps,nstims,isi)
#         for isweep in range(nsweeps):
#             for istim in range(nstims):
#                 record = ({"expdate":expdate,"dob":dob,"sex":sex,"animal":animal,"strain":strain,"region":region})
#                 record.update({"area":area,"side":side,"group":group})
#                 record.update({"neuronid":neuronid,"dendriteid":dendriteid,"spineid":spineid,"spine":spines[ispine]})
#                 record.update ({"ephysfile":ephysfile,"clamp":clamp,"stimpower":stimpower})
#                 record.update({"nsweeps":nsweeps,"nstims":nstims,"isi":isi})
#                 record.update({"isweep": isweep+1, "isi":isi,"istim":istim}) # stim0: noise
#                 print(isweep,istim)
#                 record.update({"peak":peaks[istim,isweep]})
#                 roidf = roidf.append(record,ignore_index = True) # append a record per stim
#             # }
#         # }
#         # plt.show()
#     # }
# # -----------------
# # save the dataframe
# roidf.to_csv(os.path.join(path_masterdf,dfname),index=False)
# # }
