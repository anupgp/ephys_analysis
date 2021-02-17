import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import os
import plotting_functions

def plot_histogram(fh,ah,data,nbins,ltedge,rtedge,color,alpha=1,draw_mean_std=False,setxlims=False):
    # edges start from zero and go both directions (left & right edges)
    binsize =round( (rtedge-ltedge)/nbins,2)
    edges = np.round(np.arange(ltedge,rtedge,binsize),2)
    hist = np.histogram(data,edges,density=True)[0]
    print('hist sum: ',hist.sum())
    # w = round(np.mean(np.diff(edges)),2)
    w = binsize
    print("binsize: {}".format(binsize))
    for i in np.arange(0,len(edges)-1):
        print('{} : {} --> {}'.format(edges[i],edges[i+1],hist[i]))
    # calc bin width for two lists
    ah.bar(edges[:-1]+w/2,hist,width=w,color=color,edgecolor='lightgrey',alpha=alpha)
    # plot mean and std lines
    if (draw_mean_std):
        ah.vlines(data.mean(),min(hist),max(hist),color='k')
        ah.vlines(data.mean()+(2*data.std()),min(hist),max(hist),color='gray')
    if (setxlims):
        xlims = [data.mean()-(4*data.std()),data.mean()+(4*(data.std()))]
        ah.set_xlim(xlims)
    return(fh,ah,edges,hist)

def compute_hist(df,column,nedges):
    edges = []
    hnoise = []
    hres = []
    if(len(df)<=0):
        print("empty")
        return(edges,hoise,hres)
    print(df["fileid"].unique())
    print(df["neuronid"].unique())
    print(df["vclamp"].unique())
    noise = df[df["istim"]==0][column].to_numpy()
    res = df[df["istim"]==1][column].to_numpy()
    y = np.concatenate((noise,res))
    ltedge = round(min(y))
    rtedge = round(max(y))
    fh = plt.figure()
    ah = fh.add_subplot(111)
    # title = "Neuron: "+str(df.loc[0,"neuronid"])+", VClamp: "+str(df.loc[0,"vclamp"])+ ", Fileid: " + str(df.loc[0,"fileid"])
    title = "Neuron: "+str(df.loc[0,"neuronid"])+", VClamp: "+str(df.loc[0,"vclamp"])
    fh,ah = plotting_functions.format_plot(fh,ah,xlab="Peak EPSC (pA)",ylab=" Probability density",title=title)
    fh,ah,edges,hnoise = plot_histogram(fh,ah,noise,nedges,ltedge,rtedge,alpha=1,color='red',draw_mean_std=True)
    fh,ah,edges,hres = plot_histogram(fh,ah,res,nedges,ltedge,rtedge,alpha=0.5,color='blue',draw_mean_std=False,setxlims=False)
    plt.show()
    plt.close('all')
    return([edges,hnoise,hres])


mainpath = '/Volumes/GoogleDrive/Shared drives/Beique Lab/CURRENT LAB MEMBERS/Anup Pillai/ephys_mininum_stim/'
dfname = "ephys_analysis_minstim_V2.csv"
# load data
with open(mainpath+dfname,'r') as csvfile:
    df = pd.read_csv(csvfile)
# make extra columns to contain neuron and exp indices
df["neuronid"] = df["fileid"].str.extract("(?:[0-9]+_neuronn)([\d]+)(?:.*)")
df["expid"] = df["fileid"].str.extract("(?:[0-9]+_neuron)(?:[n\d]+[n\d]_vc[-\d]+_exp)(.*)")
# ----------------------
# rename columns
df.rename(columns={"fileid":"id"},inplace = True)
df.rename(columns={"cellid":"fileid"},inplace = True)
df.rename(columns={"vlamp":"vclamp"},inplace = True)
# change column data types
df["expid"] = df["expid"].astype("int16")
df["neuronid"] = df["neuronid"].astype("int16")
df["istim"] = df["istim"].astype("int16")
df["isweep"] = df["isweep"].astype("int16")
df["nsweeps"] = df["nsweeps"].astype("int16")
# filter bad experiments/cells
badfileids = ["17511028","17511030","17511032","17512000","17512002"]
df = df[~df.fileid.isin(badfileids)]
print(df.columns)
# save data frame
dfname2 = "ephys_analysis_minstim_5badfiles_excluded.csv"    
df.to_csv("".join((mainpath,dfname2)))
"Plot histograms"
# --------------------------
# print(df.columns)
# vclamps = [-70,40]
# print("Number of Neurons:\t{}:\t{}".format(len(df.neuronid.unique()),df.neuronid.unique()))
# for vclamp in vclamps:
#     noisepeaks = df[(df["istim"] == 0) & (df["vclamp"] == vclamp)]["peak"].to_numpy()
#     respeaks = df[(df["istim"] == 1) & (df["vclamp"] == vclamp)]["peak"].to_numpy()
#     nedges = 41
#     allpeaks = np.concatenate((noisepeaks,respeaks))
#     ltedge = round(min(allpeaks))
#     rtedge = round(max(allpeaks))
#     fh = plt.figure()
#     ah = fh.add_subplot(111)
#     fh,ah = plotting_functions.format_plot(fh,ah,xlab="Peak EPSC (pA)",ylab=" Probability density",title="")
#     fh,ah,_,_ = plot_histogram(fh,ah,noisepeaks,nedges,ltedge,rtedge,alpha=1,color='red',draw_mean_std=True)
#     fh,ah,_,_ = plot_histogram(fh,ah,respeaks,nedges,ltedge,rtedge,alpha=0.5,color='blue',setxlims = False)
# plt.show()
# # compute histogram for each neuron at each vclamp
# ncells = len(df["neuronid"].unique())
# nvclamps = len(df["vclamp"].unique())
# nedges = 41
# noisehist = np.zeros((ncells,nvclamps,nedges))
# reshist = np.zeros((ncells,nvclamps,nedges))
# # ---------
# histdf = df.groupby(["neuronid","vclamp"]).apply(compute_hist,column="peak",nedges=51)
# print(histdf)
# df.groupby(["fileid"]).apply(lambda x: print(x.info()))
# print(histdf)
