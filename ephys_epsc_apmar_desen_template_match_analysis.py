import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

import re
import os
import datetime
from plotting_functions import format_plot,colors_hex

def compute_ppf(df):
    # compute paired-pulse facilitation
    # get all the factors
    # ----------------
    expdate = df.index.unique(level="expdate").strftime("%Y%m%d").values
    region = df.index.unique(level="region").values
    area = df.index.unique(level="area").values
    compartment = df.index.unique(level="compartment").values
    neuronid = df.index.unique(level="neuronid").values
    dendriteid = df.index.unique(level="dendriteid").values
    spineid = df.index.unique(level="spineid").values
    blocker = df.index.unique(level="blocker").values
    isi = df.index.unique(level="isi").values
    isweep = df.index.unique(level="isweep").values
    # ------------------------
    # sweeps = df.index.unique(level="isweep").values
    # stims = df.index.unique(level="istim").values
    #facilitation = stim2/stim1: stim0 = noise
    # peak1 = df.xs([1],level=["istim"])[["peak"]].values[0][0]
    # peak2 = df.xs([2],level=["istim"])[["peak"]].values[0][0]
    peak1 = df[df["istim"]==1]["peak"].values[0]
    peak2 = df[df["istim"]==2]["peak"].values[0]
    ppf = peak2/peak1
    # ----------
    outdf = pd.DataFrame({
        "expdate":expdate,
        "region":region,
        "area":area,
        "compartment":compartment,
        "neuronid":neuronid,
        "dendriteid":dendriteid,
        "spineid":spineid,
        "blocker":blocker,
        "isi":isi,
        "isweep":isweep,
        "ppf":ppf})
    # # -----------
    outdf = outdf.reset_index()
    # create multiindex
    outdf = outdf.set_index(["expdate","region","area","compartment","neuronid","dendriteid","spineid","blocker","isi","isweep"]).sort_index()
    # # select columns to include in the outdf
    outdf = outdf[["ppf"]]
    # # print(expdate,region,isi,sweeps,stims)
    return(outdf)
# }

# analysis of single spine uncaging glutamate CA1 & PFC-L5 for AMPAR-Desen
dfpath = "/home/anup/goofy/data/beiquelab/glu_uncage_spine_single_epsc"
dfname = "ephys_analysis_epscs_deconvolve_template_match_ampar_desen.csv"

# load data
with open(os.path.join(dfpath,dfname),'r') as csvfile:
    df = pd.read_csv(csvfile)

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
df["isi"] = (df["isi"]*1000).astype(int)
# df.set_index(["expdate","ephysfile","neuronid","dendriteid","spineid","isweep","istim"],inplace=True)
print(df)
print(df.columns)
print(df["age"].unique())
print(df["blocker"].unique())
print(df["region"].unique())
print(df["area"].unique())
print(df["compartment"].unique())
print(df["expdate"].unique())
print(df["cellfolder"].unique())
print(df["neuronid"].unique())
print(df["dendriteid"].unique())
print(df["spineid"].unique())
print(np.sort(df["isi"].unique()))
print(df["isweep"].unique())
print(df["istim"].unique())
# ------------------------------
# filter out the bad files
bad_ephysfiles = ["20d02040","20d02042","20d02049","20d02066","20d08075","20d09080","20d09120","20d09136","20d09145","20d16107","20d18167"]
bad_ephysfiles = ["".join((bad_ephysfile,".abf")) for bad_ephysfile in bad_ephysfiles]
# df = df[~df.ephysfile.isin(bad_ephysfiles)]
# ----------------
df.set_index(["expdate","region","area","compartment","neuronid","dendriteid","spineid","blocker","isi","isweep"],inplace=True)
print(df.columns)
grouped = df.groupby(level=["expdate","region","area","compartment","neuronid","dendriteid","spineid","blocker","isi","isweep"],as_index=False)
print(grouped.level)
ppfdf = grouped.apply(compute_ppf)
print(ppfdf)
grouped = ppfdf.groupby(level=["expdate","region","area","compartment","neuronid","dendriteid","spineid","blocker","isi"])
avgdf = grouped.mean()       # averaged across sweep
avgdf.reset_index(inplace=True)
spines = avgdf[["expdate","region","area","compartment","neuronid","dendriteid","spineid","blocker"]].astype("str").agg("_".join,axis=1)
avgdf[["spine"]] = spines
avgdf.set_index(["expdate","region","area","compartment","neuronid","dendriteid","spineid","blocker","isi"],inplace=True)
print(avgdf)
# --------------------------------
# compute mean and sem per isi
isis = np.array([10,20,30,40,50,100,200,400,600,800,1000])
selectdf = avgdf.xs(["hc","ca1","apical"],level=["region","area","compartment"],drop_level=False)
# selectdf = avgppfdf.xs(["hc","ca1","basal"],level=["region","area","compartment"],drop_level=False)
# selectdf = avgppfdf.xs(["pfc","l5","apical"],level=["region","area","compartment"],drop_level=False)
# selectdf = avgppfdf.xs(["pfc","l5","basal"],level=["region","area","compartment"],drop_level=False)

ctrlspines = selectdf["spine"][selectdf["spine"].str.contains('ctrl')].unique()
ctrlspines = [re.findall(r'.*[^_ctrl]',spine)[0] for spine in ctrlspines]
ctzspines = selectdf["spine"][selectdf["spine"].str.contains('ctz')].unique()
ctzspines = [re.findall(r'.*[^_ctz]',spine)[0] for spine in ctzspines]
ctrlctzspines = list(set(ctrlspines).intersection(set(ctzspines)))
for spine in ctrlctzspines:
    print(spine)
    input()
input()
selectdf.set_index(["expdate","neuronid","dendriteid","spineid","blocker","isi","spine"],inplace=True)
# grouped = selectdf.groupby(level=(["spine"]))
for key,group in selectdf.groupby(level=["spine"]):
    print(key,group)
    input()

input()
# -----------------------
grouped = selectdf.groupby(level=(["blocker","isi"]))
plotdf = grouped[["ppf"]].apply(lambda x:pd.DataFrame({"mean":np.mean(x),"sem":np.std(x)/np.sqrt(len(x)),"count":len(x)}))
print(plotdf)
fh = plt.figure(figsize=(8,6))
ah = fh.add_subplot(111)
colorcycle = ['#000000', '#ff0000'] # rgb
fh.subplots_adjust(
    top=0.9,
    bottom=0.2,
    left=0.15,
    right=0.9,
    hspace=0.2,
    wspace=0.2
)
count = 0
# -------
for key,group in plotdf.groupby(level=["blocker"]):
    print(key)
    print(group)
    group = group[group.index.get_level_values("isi").isin(isis)]
    xvals = np.sort(group.index.unique(level="isi").values)
    yavgs = group["mean"].values
    ysems = group["sem"].values
    nsamples = int(np.mean(group["count"].values))
    print(xvals,yavgs,ysems)
    label = "".join((key," (",str(nsamples),")"))
    line, = ah.plot(xvals,yavgs,marker='o',linestyle='--',linewidth=1,markersize=7,color = colorcycle[count],label=label)
    for i in range(len(ysems)):
        ah.plot([xvals[i],xvals[i]],[yavgs[i]-ysems[i],yavgs[i]+ysems[i]],color = colorcycle[count])
    # }
    count = count + 1
# }
ah.hlines(1,isis[0],isis[-1],linestyle="--",linewidth=1,color='k')
ah.set_xscale('log')
xlabel = "\nInter-stimulus interval (ms)"
ylabel = "Paired-pulse ratio (EPSC2/EPSC1)"
title= "Neurons: HC CA1 pyramidal - Apical"
# ah.set_xlabel("\nInter-stimulus interval (ms)",fontsize=20)
# ah.set_ylabel("Paired-pulse ratio (EPSC2/EPSC1)",fontsize=20)
# ah.set_title()
# ah.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ah.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))
ah.set_xticks([10,100,1000])
ah.set_xticklabels([10,100,1000],fontsize=16)
ah.set_ylim([0.4,2.6])
ah.set_yticks([0.5,1,1.5,2,2.5])
ah.set_yticklabels([0.5,1,1.5,2,2.5],fontsize=16)
lh = plt.legend()
lh.get_frame().set_facecolor("none")
# --------------------------
# save figure
fh,ah = format_plot(fh,ah,xlabel,ylabel,title)
plt.show()
figname = "hc_ca1_apical_epsc_ppr_single_spine_amapar_desen.png"
# fh.savefig(os.path.join(dfpath,figname),dpi=300)

