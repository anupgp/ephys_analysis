import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

import re
import os
import datetime
from plotting_functions import format_plot,colors_hex

# analysis of clustered glutamate uncaging at proximal spines on PFC neurons
# dfpath = "/Users/macbookair/goofy/data/beiquelab/pfc_clusterd_inputs"
# dfname = "ephys_analysis_epsp_clustered_input.csv"

# analysis of clustred glutamate uncaging at proximal spines on CA1 neurons
dfpath = "/home/anup/gdrive-beiquelab/CURRENT LAB MEMBERS/Anup Pillai/ca1_uncaging_cluster/"
dfname = "ephys_analysis_epsp_clustered_input_hcca1.csv"

# load data
with open(os.path.join(dfpath,dfname),'r') as csvfile:
    df = pd.read_csv(csvfile)

print(df)
input()
# preprocessing
# change column data types
df["expdate"] = pd.to_datetime(df["expdate"],format="%Y%m%d")
df["dob"] = pd.to_datetime(df["dob"])
df["neuronid"] = df["neuronid"].astype("int8")
df["dendriteid"] = df["dendriteid"].astype("int8")
df["isweep"] = df["isweep"].astype("int16")
df["istim"] = df["istim"].astype("int8")
df["nsweeps"] = df["nsweeps"].astype("int16")
df["nstims"] = df["nstims"].astype("int16")
# make spineid categorical
# df["spineid"] = pd.Categorical(df["spineid"],ordered=True,categories=["a","b","c","d","e","abcde","+abcde"])
# simple calculations
df["age"] = df["expdate"]-df["dob"]
df.set_index(["expdate","ephysfile","neuronid","dendriteid","spineid","isweep","istim"],inplace=True)
# midx = df.index
# print(midx.sort_values().get_level_values("spineid").unique())
# print(df.sort_index())
# print(df.reset_index())
      
# df.sort_index()
# normalize peak w.r.t clustered peak for each [expdate,neuronid,dendriteid,isweep,istim]
def normalize(df):
    # calculate avg peak for single stims
    expdate = df.index.unique(level="expdate").values
    neuronid = df.index.unique(level="neuronid").values
    dendriteid = df.index.unique(level="dendriteid").values
    sweeps = df.index.unique(level="isweep").values
    stims = df.index.unique(level="istim").values
    # --------------
    # for single spines take all stimulation responses and average them
    spines = ["a","b","c","d","e"]
    epsptypes = spines
    avgpeaks = np.zeros((len(spines)))
    for i in range(len(spines)): 
        sdf = df.xs([spines[i]],level=["spineid"])[["peak"]]
        avgpeaks[i] = sdf["peak"].values.mean()
        # print(sdf)
    # for clustered spines take the first stimulation and average them
    dfcluster1 = df.xs(["abcde",1],level=["spineid","istim"])[["peak"]]
    clusterpeaks = dfcluster1["peak"].values
    # append the average cluster epsp peak
    avgpeaks = np.append(avgpeaks,clusterpeaks.mean())
    epsptypes = epsptypes + ["abcde"]
    # append ths sum to the array
    avgpeaks = np.append(avgpeaks,np.sum(avgpeaks[:-1]))
    epsptypes = epsptypes + ["sum"]
    # normalize avgpeaks w.r.t sum of individual single epsps
    navgpeaks = np.zeros(avgpeaks.shape)
    navgpeaks = (avgpeaks/avgpeaks[-1])*100
    outdf = pd.DataFrame({
        "expdate":np.repeat(expdate,len(epsptypes)),
        "neuronid":np.repeat(neuronid,len(epsptypes)),
        "dendriteid":np.repeat(dendriteid,len(epsptypes)),
        "epsptype":epsptypes,"peak":avgpeaks,"npeak":navgpeaks})
    # -----------
    outdf.reset_index(inplace=True)
    # create multiindex
    outdf = outdf.set_index(["epsptype"]).sort_index()
    print("outdf",outdf)
    # keep only peak and npeak columns
    outdf = outdf[["peak","npeak"]]
    # print(outdf)
    # ------------
    # print(expdate)
    # print(neuronid)
    # print(dendriteid)
    # print(sweeps)
    # print(stims)
    # print(avgpeaks)
    # print(navgpeaks)
    # print(epsptypes,len(epsptypes))
    # create output dataframe
    # print(outdf)
    return(outdf)
    
    
grouped = df.groupby(level=["expdate","neuronid","dendriteid"])
ndf = grouped.apply(normalize)
# create a new caregorical index for spineid
cidx = pd.CategoricalIndex(ndf.index.get_level_values("epsptype"),categories=["a","b","c","d","e","abcde","sum"],ordered=True)
# map new values for the categorical index
cidx = cidx.map({'a':'spine 1','b':'spine 2','c':'spine 3','d': 'spine 4','e':'spine 5','abcde':'5 spines','sum':'linear sum'})
# reset index
ndf.reset_index(inplace=True)
# generate new multiindex with one ordered categorical index
ndf = ndf.set_index(["expdate","neuronid","dendriteid",cidx]).sort_index()
# drop epsptype column
ndf = ndf.drop(["epsptype"],axis=1)
# compute average per epsptype
grouped = ndf.groupby(["epsptype"])
avgdf = grouped[["peak","npeak"]].apply(lambda x:pd.DataFrame({"mean":np.mean(x),"sem":np.std(x)/np.sqrt(len(x))}))
avgpeakdf = avgdf.xs("peak",level=1)
avgnpeakdf = avgdf.xs("npeak",level=1)
# plotdf = avgnpeakdf
plotdf = avgpeakdf
paramname = "peak"
# plot summation
# colorcycle = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# colorcycle = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e']
colorcycle = colors_hex
fh1 = plt.figure(figsize=(8,6))
ah = fh1.add_subplot(111)
fh1,ah = format_plot(fh1,ah)
fh1.subplots_adjust(
    top=0.9,
    bottom=0.2,
    left=0.15,
    right=0.9,
    hspace=0.2,
    wspace=0.2
)
ah.bar(plotdf.index.values,plotdf["mean"],yerr=plotdf["sem"],color="darkgray")
# plot individual data points
grouped = ndf.groupby(["expdate","neuronid","dendriteid"])
# linecolors = []
count = 0
for name,group in grouped:
    # xvals = group.index.get_level_values("epsptype").values
    xvals = np.array([0,1,2,3,4,5,6])-0.2
    line, = ah.plot(xvals,group[paramname],marker='o',linestyle='--',linewidth=1,markersize=7,color = colors_hex[count])
    # line, = ah.plot(xvals,group["npeak"],marker='o',linestyle='--',linewidth=1,markersize=7)
    count = count + 1
    # linecolors.append(line.get_color())

ah.set_xlabel(None,fontsize=16)
# ah.set_ylabel("EPSP amplitude (%) ",fontsize=18)
ah.set_ylabel("EPSP amplitude (mV)",fontsize=18)
ah.set_xticks(list(plotdf.index))
ah.set_xticklabels(labels = list(plotdf.index),rotation=45)
ah.set_frame_on(True)
# ah.axes.get_xaxis().set_visible(False)
ah.spines["bottom"].set_color("white")
# print("plot colors: ",linecolors)
# save figure
# figname = "pfc_clustered_inputs_epsp_summation_normalized.png"
figname = "pfc_clustered_inputs_epsp_summation_mvolts.png"
fh1.savefig(os.path.join(dfpath,figname),dpi=300)
plt.show()


# function to compute facilitation of paired epsps
def facilitation(df):
    expdate = df.index.unique(level="expdate").values
    neuronid = df.index.unique(level="neuronid").values
    dendriteid = df.index.unique(level="dendriteid").values
    # for each clustered input find the average
    dfcluster1 = df.xs(["abcde",1],level=["spineid","istim"])[["peak"]]
    peaks1 = dfcluster1["peak"].values
    dfcluster = df.xs(["abcde"],level=["spineid"])
    # convert isi to milliseconds
    dfcluster["isi"] = dfcluster["isi"]*1000
    dfcluster["isi"] = dfcluster["isi"].astype("int16")
    # group by isi and istim
    grouper = dfcluster.groupby(["isi"])
    isis = []
    avgpeaks1 = []
    avgpeaks2 = []
    navgpeaks1 = []
    navgpeaks2 = []
    for name,group in grouper:
        peaks1 = group.xs([1],level=["istim"])["peak"].values
        peaks2 = group.xs([2],level=["istim"])["peak"].values
        isis.append(group["isi"].unique()[0])
        avgpeaks1.append(peaks1.mean())
        avgpeaks2.append(peaks2.mean())
        # --------------
    isis = np.array(isis)
    avgpeaks1 = np.array(avgpeaks1)
    avgpeaks2 = np.array(avgpeaks2)
    ratio = (avgpeaks2/avgpeaks1)*100
    
    outdf = pd.DataFrame({
        "expdate":np.repeat(expdate,len(isis)),
        "neuronid":np.repeat(neuronid,len(isis)),
        "dendriteid":np.repeat(dendriteid,len(isis)),
        "isi":isis,"peak1":avgpeaks1,"peak2":avgpeaks2,
        "peak1":avgpeaks1,"peak2":avgpeaks2,"ratio":ratio
    })
    outdf.reset_index(inplace=True)
    # create multiindex
    outdf = outdf.set_index(["isi"]).sort_index()
    # keep only peak1,peak2 and ratio columns
    outdf = outdf[["peak1","peak2","ratio"]]
    return(outdf)
                      

    
grouped = df.groupby(level=["expdate","neuronid","dendriteid"])
fdf = grouped.apply(facilitation)
# reset index
fdf.reset_index(inplace=True)
# generate new multiindex with one ordered categorical index
fdf = fdf.set_index(["expdate","neuronid","dendriteid"]).sort_index()
print(fdf)
# compute mean and sem per isi
grouped = fdf.groupby(["isi"])
avgfdf = grouped[["peak1","peak2","ratio"]].apply(lambda x:pd.DataFrame({"mean":np.mean(x),"sem":np.std(x)/np.sqrt(len(x)),"count":len(x)}))
# drop rows with count <2
avgfdf = avgfdf.drop(avgfdf[avgfdf["count"]<2].index)
print(avgfdf.xs(["ratio"],level=[1]))

# plot facilitation
fh2 = plt.figure(figsize=(8,6))
ah = fh2.add_subplot(111)
fh2.subplots_adjust(
    top=0.9,
    bottom=0.2,
    left=0.15,
    right=0.9,
    hspace=0.2,
    wspace=0.2
)
param = "peak2"
fh2,ah = format_plot(fh2,ah)
ah.errorbar(avgfdf.xs([param],level=[1]).index,avgfdf.xs([param],level=[1])["mean"],
            yerr=avgfdf.xs([param],level=[1])["sem"],
            fmt='o',color="black",elinewidth=2,
            ecolor="darkgray")
ah.plot(avgfdf.xs([param],level=[1]).index,avgfdf.xs([param],level=[1])["mean"])
# plot individual data points
grouped = fdf.groupby(["expdate","neuronid","dendriteid"])
# linecolors = []
count = 0
for name,group in grouped:
    # xvals = group.index.get_level_values("epsptype").values
    xvals = np.array([30,40,50,100,500,1000])
    group = group[group["isi"].isin(xvals)]
    xvals = group["isi"].values
    print(group)
    line, = ah.plot(xvals,group[param],marker='o',linestyle='--',linewidth=1,markersize=7,color = colorcycle[count])
    # line, = ah.plot(xvals,group[param],marker='o',linestyle='--',linewidth=1,markersize=7)
    count = count + 1
    # linecolors.append(line.get_color())

ah.set_xscale('log')
ah.set_xlabel("\nInter-stimulus interval (ms)",fontsize=16)
ah.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ah.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))
ah.set_xticks([30,40,50,100,500,1000])
# ah.set_ylabel("Paired-pulse ratio (EPSP$_2$ / EPSP$_1$) \n",fontsize=16)
ah.set_ylabel("Paired-pulse EPSP$_2$ (mV) \n",fontsize=16)
# save figure
# fig2name = "pfc_clustered_inputs_paired_pulse_ratio.png" 
fig2name = "pfc_clustered_inputs_paired_pulse_epsp2.png"
fh2.savefig(os.path.join(dfpath,fig2name),dpi=300)
plt.show()



