import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import os

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
# 
fh = plt.figure()
ah = fh.add_subplot(111)
print(df.columns)
noisepeaks = df[(df["istim"] == 0) & (df["vclamp"] == -70)]["peak"].to_numpy()
respeaks = df[(df["istim"] == 1) & (df["vclamp"] == -70)]["peak"].to_numpy()
nedges = 70
allpeaks = np.concatenate((noisepeaks,respeaks))
edges = np.arange(min(allpeaks),max(allpeaks),(max(allpeaks)-min(allpeaks))/nedges)
noisehist = np.histogram(noisepeaks,edges)[0]
reshist = np.histogram(respeaks,edges)[0]
print(len(noisepeaks),edges,noisehist)
ah.bar(edges[:-1],noisehist,np.mean(np.diff(edges)),alpha = 1,color="red")
ah.bar(edges[:-1],reshist,np.mean(np.diff(edges)),alpha = 0.5,color="blue")
plt.show()

# compute histogram for each neuron at each vclamp
ncells = len(df["neuronid"].unique())
nvclamps = len(df["vclamp"].unique())
nedges = 50
noisehist = np.zeros((ncells,nvclamps,nedges))
reshist = np.zeros((ncells,nvclamps,nedges))

def compute_hist(df,column,nedges):
    noise = df[df["istim"]==0][column].to_numpy()
    res = df[df["istim"]==1][column].to_numpy()
    y = np.concatenate((noise,res))
    edges = np.arange(min(y),max(y),(max(y)-min(y))/nedges)
    hnoise = np.histogram(noise,edges)[0]
    hres = np.histogram(res,edges)[0]
    fh = plt.figure()
    ah = fh.add_subplot(111)
    width = np.mean(np.diff(edges))
    ah.bar(edges[:-1],hnoise,width=width,alpha=1,color='grey')
    ah.bar(edges[:-1],hres,width=width,alpha=0.5,color='blue')
    ah.set_title(df.loc[0,"id"])
    plt.show()
    return([edges,hnoise,hres])

histdf = df.groupby(["neuronid","vclamp"]).apply(compute_hist,column="peak",nedges=50)
print(histdf)
