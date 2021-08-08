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
# ------------------------------
# df.set_index(["expdate","region","area","side","neuronid","dendriteid","spineid","group","isi","isweep"],inplace=True)
print(df.columns)
spines = df["spine"].unique()
spines = [spine[5:] for spine in spines]
print(spines)
# --------------------
grouped = df[df["clamp"]=="cc"].groupby(["expdate","region","area","side","neuronid","dendriteid","spineid","group","isi","isweep"])
ppfdf = grouped.apply(compute_ppf).reset_index()[["spine","isi","ppf"]]
ppfdf = ppfdf.groupby(["spine","isi"]).apply(np.mean)[["ppf"]].reset_index()
ppfdf = ppfdf[ppfdf["isi"]==50].reset_index()
print(ppfdf)
# -------------------
# find spines that are in both dfs
# compute correlation
def calculate(x):
    return(pd.DataFrame({"peak":[np.mean(x["peak"])]}))
tdf = df[df["istim"]==1].groupby(["spine","clamp"]).apply(calculate).reset_index()
tdf = tdf[(tdf["clamp"]=="vc") & (tdf["spine"].isin(ppfdf["spine"].unique()))].reset_index()
print(tdf)
corrdf = pd.merge(ppfdf[["spine","ppf"]],tdf[["spine","peak"]],on="spine")
print(corrdf)
xvals = corrdf["peak"].to_numpy()
yvals = corrdf["ppf"].to_numpy()
print(xvals)
print(yvals)
slope,intercept,r_value,p_value,std_err =  scipystats.linregress(xvals,yvals)
fvals = (xvals*slope) + intercept
print(fvals)
print(slope,intercept,r_value,p_value)
# ---------------------------
# plot correlation
fh1 = plt.figure(figsize=(4,4))
ah1 = fh1.add_subplot(111)
zorder=0
for name,group in corrdf.groupby(["spine"]):
    icolor = [j for j in np.arange(0,len(spines)) if name[5:] == spines[j]][0]
    ah1.plot(group["peak"],group["ppf"],color = colors_rgb[icolor,:],label=spines[icolor],marker='o',markersize=5,markerfacecolor=colors_rgb[icolor,:],zorder=zorder,linestyle="")
    zorder = zorder + 1
    print(name,icolor,spines[icolor])
# }
ah1.plot(xvals,fvals,marker="",linestyle="-",color="k")
ah1.set_xlim([-60,0])
# ah1.set_ylim([0,4])
ah1.set_ylim([0.6,1])
# ah1.text(-50,3.1,"".join(('Correlation = ',str(round(r_value,2)),"\nP value = ",str(round(p_value,2)),"\nConductance = ",str(round(slope,3)*1000),"nS")),fontsize=14)
ah1.text(-50,0.62,"".join(('Correlation = ',str(round(r_value,2)),"\nP value = ",str(round(p_value,2)))),fontsize=14)
# fh1,ah1 = format_plot(fh1,ah1,"EPSC peak (pA)","EPSP peak (mV)","")
fh1,ah1 = format_plot(fh1,ah1,"EPSC peak (pA)","EPSP2/EPSP1","")
# ah1.legend(loc="best",framealpha=0,bbox_to_anchor=(1.1,1))
fh1.tight_layout()
fh1.savefig(os.path.join(dfpath,"epsc_ppf_corr.png"),transparent=True,dpi=300)
plt.show()
# ----------------------------------------------
