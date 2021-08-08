import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
import re
import os
import datetime
from plotting_functions import format_plot,colors_hex
from ephys_glu_uncage_generalized_analysis_functions import compute_ppf
from ephys_glu_uncage_generalized_analysis_functions import colors_rgb,colors_hex
from scipy import stats as scipystats


# -------------------------
# Ca1 single spine
dfpath="/home/anup/goofy/data/beiquelab/hc_ca1_glu_uncage_spine_cluster_epsp/" # local to the ephys project: Ca1 single spine
dfname = "hc_ca1_glu_uncage_spine_cluster_masterdata.csv"           # local to the ephys project
# load data
with open(os.path.join(dfpath,dfname),'r') as csvfile:
    df = pd.read_csv(csvfile)
# }
# preprocessing
# change column data types
df["expdate"] = pd.to_datetime(df["expdate"],format="%Y%m%d")
df["dob"] = pd.to_datetime(df["dob"],format="%Y%m%d")
df["age"] = (df["expdate"]-df["dob"]).dt.days
df["spineid"] = df["spineid"].astype("int16")
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
print(df["spine"].unique())
spines = df[df["spineid"]>10]["spine"].unique()
spines = [spine[5:] for spine in spines]
print(spines)
def compute_cluster_linearsum(x):
    # print(x[(x["spineid"].isin([1,2,3,4,5])) & (x["istim"]>0)]["spineid"].unique())
    # print(x[(x["spineid"].isin([1,2,3,4,5])) & (x["istim"]>0)]["spine"].unique())
    clusterpeak = x[(x["spineid"]>10)]["peak"].to_numpy()[0]
    linsumpeak = x[(x["spineid"].isin([1,2,3,4,5]))]["peak"].to_numpy().sum()
    spineids = x[(x["spineid"]<10)]["spineid"].unique()
    isweep = x[(x["spineid"]<10)]["isweep"].unique()[0]
    istim = x[(x["spineid"]<10)]["istim"].unique()[0]
    clusterid = str(x[(x["spineid"]>10)]["spineid"].unique()[0])        
    spines = x[(x["spineid"]<10)]["spine"].unique()
    cluster = x[(x["spineid"]>10)]["spine"].unique()[0]
    singlepeaks = []
    for spineid in spineids:
        singlepeaks.append((x[(x["spineid"]==spineid)]["peak"]).to_numpy())
    # }
    columns = ["name"]
    [columns.append(str(elm)) for elm in spineids]
    columns.append("cluster")
    columns.append("sum")
    data = [cluster]
    data = data + singlepeaks
    data.append(clusterpeak)
    data.append(linsumpeak)
    outdf = pd.DataFrame(dict(zip(columns,data)),index=[0])
    return(outdf)
grouped_cluster = df[(df["clamp"]=="cc") & (df["istim"]>0) & (df["isi"] == 1000)].groupby(["expdate","region","area","side","neuronid","dendriteid","group","isi","isweep","istim"])
# dfcc = grouped.apply(compute_ppf).reset_index()
dfls = grouped_cluster.apply(compute_cluster_linearsum).reset_index()[["name","isweep","istim","1","2","3","4","5","cluster","sum"]]
dfls = dfls.groupby(["name"]).apply(np.mean).reset_index()[["name","1","2","3","4","5","cluster","sum"]]
print(dfls)
dflsavg = dfls[["1","2","3","4","5","cluster","sum"]].apply(lambda x: np.mean(x),axis=0)
dflssem = dfls[["1","2","3","4","5","cluster","sum"]].apply(lambda x: np.std(x)/np.sqrt(len(x)),axis=0)
print(dflsavg)
print(spines)
input()

# plot ppf
fh1 = plt.figure(figsize=(6,4))
ah1 = fh1.add_subplot(111)
zorder = 0
categories = ["Spine 1","Spine 2","Spine 3","Spine 4","Spine 5","Cluster","Sum"]
for name in dfls["name"].unique():
    icolor = [j for j in np.arange(0,len(spines)) if name[5:] == spines[j]][0]
    values = dfls[dfls["name"]==name][["1","2","3","4","5","cluster","sum"]].to_numpy().reshape((7))
    ah1.plot(np.arange(0,len(values)),values,color = colors_rgb[icolor,:],label=spines[icolor],marker='o',markersize=5,markerfacecolor=colors_rgb[icolor,:],zorder=zorder,linestyle="-",alpha=1)
    zorder = zorder + 1
    print(name,icolor,spines[icolor])
# }
# ah1.plot(categories,dflsavg,linewidth=3,color="k",markersize=5,markerfacecolor='k',zorder=zorder+1)
font_path = '/home/anup/.matplotlib/fonts/arial.ttf'
fontprop = font_manager.FontProperties(fname=font_path,size=18)
ah1.bar(categories,dflsavg,width=0.8,color="k",zorder=zorder+1,alpha=0.5)
ah1.vlines(categories,dflsavg+dflssem,dflsavg-dflssem,linewidth=3,color='k',zorder=zorder+2)
ah1.set_xticks(categories)
ah1.set_xticklabels(categories,rotation=45,fontproperties=fontprop)
ah1.spines["right"].set_visible(False)
ah1.spines["top"].set_visible(False)
ah1.spines["bottom"].set_linewidth(1)
ah1.spines["left"].set_linewidth(1)
ah1.set_ylim(0,10)
# ah.set_title(title,fontproperties=fontprop)
ah1.set_xlabel("",fontproperties=fontprop)
ah1.set_ylabel("EPSP amplitude (mV)",fontproperties=fontprop)
ah1.tick_params(axis='both',length=6,direction='out',width=1,which='major')
ah1.tick_params(axis='both',length=3,direction='out',width=1,which='minor')
ah1.tick_params(axis='both', which='major', labelsize=14)
ah1.tick_params(axis='both', which='minor', labelsize=12)
# box = ah.get_position()
# ah.set_position([box.x0+0.03, box.y0+0.03, box.width * 0.9, box.height*0.9])
# ah1.hlines(1,30,1000,linestyle="--",linewidth=2,color="k")
# fh1,ah1 = format_plot(fh1,ah1,"Inter-stimulus interval (ms)\n","\nPaired EPSP ratio","\nCA1 cluster spine glutamate uncaging")
# fh1,ah1 = format_plot(fh1,ah1,"Inter-stimulus interval (ms)\n","\nEPSP1 (mV)","\nCA1 single spine glutamate uncaging")
ah1.legend(loc="best",framealpha=0,bbox_to_anchor=(1.1,1))
fh1.tight_layout()
fh1.savefig(os.path.join(dfpath,"fig3.png"),transparent=True,dpi=300)
plt.show()
# --------------------------
# isis = [30,40,50,100,200,400,600,800,1000]
# param="peak"
# ddf = dfcc[(dfcc["istim"]==1) & (dfcc["isi"].isin(isis)) & (dfcc["spineid"]>10)][["spine","isweep","isi",param]].groupby(["spine","isi"]).apply(lambda x: pd.DataFrame({param: [np.mean(x[param])]})).reset_index()
# ddf = ddf.groupby(["spine"]).filter(lambda x: len(x)==len(isis)) # filter spines with missing isi
# # ddf = ddf.groupby(["spine"]).filter(lambda x: x[x["isi"]==40]["ppf"] <= 1) # filter two spines that have large ppf
# print(ddf)
# print(ddf["spine"].unique())
# avgdf = ddf.groupby(["isi"]).apply(lambda x: pd.DataFrame({"mean": [np.nanmean(x[param])]}))
# semdf = ddf.groupby(["isi"]).apply(lambda x: pd.DataFrame({"sem": [np.nanstd(x[param])/np.sqrt(len(x))]}))
# print(ddf)
# print(ddf[ddf["spine"]==ddf["spine"].unique()[0]])
# print(avgdf)
# print(semdf)
# --------------------
# plot ppf
fh1 = plt.figure(figsize=(6,4))
ah1 = fh1.add_subplot(111)
zorder = 0
for name,group in ddf.groupby(["spine"]):
    icolor = [j for j in np.arange(0,len(spines)) if name[5:] == spines[j]][0]
    ah1.semilogx(group["isi"].to_numpy(),group[param].to_numpy(),color = colors_rgb[icolor,:],label=spines[icolor],marker='o',markersize=5,markerfacecolor=colors_rgb[icolor,:],zorder=zorder,linestyle="-",alpha=1)
    zorder = zorder + 1
    print(name,icolor,spines[icolor])
# }
ah1.semilogx(avgdf.index.get_level_values("isi"),avgdf["mean"],color="k",linewidth=3,markersize=5,markerfacecolor='k',zorder=zorder+1)
ah1.vlines(avgdf.index.get_level_values("isi"),avgdf["mean"]+semdf["sem"],avgdf["mean"]-semdf["sem"],linewidth=3,color='k',zorder=zorder+2)
# ah1.hlines(1,30,1000,linestyle="--",linewidth=2,color="k")
# fh1,ah1 = format_plot(fh1,ah1,"Inter-stimulus interval (ms)\n","\nPaired EPSP ratio","\nCA1 cluster spine glutamate uncaging")
fh1,ah1 = format_plot(fh1,ah1,"Inter-stimulus interval (ms)\n","\nEPSP1 (mV)","\nCA1 single spine glutamate uncaging")
ah1.legend(loc="best",framealpha=0,bbox_to_anchor=(1.1,1))
fh1.tight_layout()
# --------------------
fh1.savefig(os.path.join(dfpath,"fig2.png"),transparent=True,dpi=300)
plt.show()
# perform paired t-test
for isi in ddf["isi"].unique()[0:-1]:
    g1 = ddf[(ddf["isi"]==1000)][param].to_numpy()
    gi = ddf[(ddf["isi"]==isi)][param].to_numpy()
    pval = scipystats.ttest_rel(g1,gi,nan_policy='omit')[1]
    print("Paired T-test between isi {} & isi {}:  pvalue = {} ".format(1000,isi,pval)) 
# ---------------
