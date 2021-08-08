import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

import re
import os
import datetime
from plotting_functions import format_plot,colors_hex
from ephys_glu_uncage_generalized_analysis_functions import compute_ppf
from ephys_glu_uncage_generalized_analysis_functions import colors_rgb,colors_hex
from scipy import stats as scipystats


# -------------------------
# Ca1 single spine
dfpath="/home/anup/goofy/data/beiquelab/hc_ca1_glu_uncage_spine_single_epsp/" # local to the ephys project: Ca1 single spine
dfname = "hc_ca1_glu_uncage_spine_single_masterdata.csv"           # local to the ephys project
# load data
with open(os.path.join(dfpath,dfname),'r') as csvfile:
    df = pd.read_csv(csvfile)
# }
# preprocessing
# change column data types
df["expdate"] = pd.to_datetime(df["expdate"],format="%Y%m%d")
df["dob"] = pd.to_datetime(df["dob"],format="%Y%m%d")
df["age"] = (df["expdate"]-df["dob"]).dt.days
df["spineid"] = df["spineid"].astype("int8")
df["neuronid"] = df["neuronid"].astype("int8")
df["dendriteid"] = df["dendriteid"].astype("int8")
df["isweep"] = df["isweep"].astype("int16")
df["istim"] = df["istim"].astype("int8")
df["nsweeps"] = df["nsweeps"].astype("int16")
df["nstims"] = df["nstims"].astype("int16")
df["isi"] = (df["isi"]).astype(int)
# df.set_index(["expdate","ephysfile","neuronid","dendriteid","spineid","isweep","istim"],inplace=True)
print(df)
print(df.columns)
print(df["isweep"].unique())
input()
# ------------------------------
df.set_index(["expdate","region","area","side","neuronid","dendriteid","spineid","group","isi","isweep"],inplace=True)
print(df.columns)
grouped = df[df["clamp"]=="cc"].groupby(level=["expdate","region","area","side","neuronid","dendriteid","spineid","group","isi","isweep"])
print(grouped.level)
# -------------------
df = grouped.apply(compute_ppf).reset_index()
spines = df["spine"].unique()
spines = [spine[5:] for spine in spines]
print(spines)
isis = [50,100,200,400,600,800]
param="ppf"
ddf = df[(df["istim"]==1) & (df["isi"].isin(isis))][["spine","isweep","isi",param]].groupby(["spine","isi"]).apply(lambda x: pd.DataFrame({param: [np.mean(x[param])]})).reset_index()
# ddf = ddf.groupby(["spine"]).filter(lambda x: len(x)==len(isis)) # filter spines with missing isi
# ------------------
avgdf = ddf.groupby(["isi"]).apply(lambda x: pd.DataFrame({"mean": [np.nanmean(x[param])]}))
semdf = ddf.groupby(["isi"]).apply(lambda x: pd.DataFrame({"sem": [np.nanstd(x[param])/np.sqrt(len(x))]}))
print(ddf[ddf["spine"]==ddf["spine"].unique()[0]])
print(avgdf)
print(semdf)
# --------------------
# plot ppf
fh1 = plt.figure(figsize=(6,4))
ah1 = fh1.add_subplot(111)
zorder = 0
for name,group in ddf.groupby(["spine"]):
    icolor = [j for j in np.arange(0,len(spines)) if name[5:] == spines[j]][0]
    ah1.plot(group["isi"].to_numpy(),group[param].to_numpy(),color = colors_rgb[icolor,:],label=spines[icolor],marker='o',markersize=5,markerfacecolor=colors_rgb[icolor,:],zorder=zorder,linestyle="-",alpha=0.5)
    zorder = zorder + 1
    print(name,icolor,spines[icolor])
# }
ah1.plot(avgdf.index.get_level_values("isi"),avgdf["mean"],color="k",linewidth=3,markersize=5,markerfacecolor='k',zorder=zorder+1)
ah1.vlines(avgdf.index.get_level_values("isi"),avgdf["mean"]+semdf["sem"],avgdf["mean"]-semdf["sem"],linewidth=3,color='k',zorder=zorder+2)
ah1.hlines(1,50,800,linestyle="--",linewidth=2,color="k")
fh1,ah1 = format_plot(fh1,ah1,"Inter-stimulus interval (ms)\n","\nPaired EPSP ratio","\nCA1 single spine glutamate uncaging")
# fh1,ah1 = format_plot(fh1,ah1,"Inter-stimulus interval (ms)\n","\nEPSP1 (mV)","\nCA1 single spine glutamate uncaging")
# ah1.legend(loc="best",framealpha=0,bbox_to_anchor=(1.1,1))
fh1.tight_layout()
# --------------------
fh1.savefig(os.path.join(dfpath,"ppf_all.png"),transparent=True,dpi=300)
# plt.show()
# ---------------------
# perform paired t-test
for isi in ddf["isi"].unique()[1:]:
    g1 = ddf[(ddf["isi"]==50)]["ppf"].to_numpy()
    gi = ddf[(ddf["isi"]==isi)]["ppf"].to_numpy()
    pval = scipystats.ttest_rel(g1,gi,nan_policy='omit')[1]
    print("Paired T-test between isi {} & isi {}:  pvalue = {} ".format(50,isi,pval)) 
# ---------------

