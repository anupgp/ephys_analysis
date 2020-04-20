import numpy as np
import pandas as pd
import math
import seaborn as sns
import re
from scipy import stats as scipystats
import os

# Set some pandas options
pd.set_option('display.notebook_repr_html', False)
# ----------------
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
sns.set()
sns.set_style("white")
sns.set_style("ticks")
# ---------------


def format_plot(fh,ah,xlab="",ylab="",title=""):
    font_path = '/Users/macbookair/.matplotlib/Fonts/Arial.ttf'
    fontprop = font_manager.FontProperties(fname=font_path,size=18)
    
    ah.spines["right"].set_visible(False)
    ah.spines["top"].set_visible(False)
    ah.spines["bottom"].set_linewidth(1)
    ah.spines["left"].set_linewidth(1)
    ah.set_title(title,FontProperties=fontprop)
    ah.set_xlabel(xlab,FontProperties=fontprop)
    ah.set_ylabel(ylab,fontproperties=fontprop)
    ah.tick_params(axis='both',length=6,direction='out',width=1,which='major')
    ah.tick_params(axis='both',length=3,direction='out',width=1,which='minor')
    ah.tick_params(axis='both', which='major', labelsize=16)
    ah.tick_params(axis='both', which='minor', labelsize=12)
    return(fh,ah)

## Calculate facilitation (ppratio) for each spineid+clamp+isi
def compute_ppr(x):
    nstims = len(x)
    firstpeak = x[x["istim"] == 0]["peak"].to_numpy()
    if (nstims>1):
        lastpeak = x[x["istim"] == nstims-1]["peak"].to_numpy()
        ppr = np.repeat((lastpeak/firstpeak),(nstims))
    else:
        ppr = 0
    dftemp = pd.DataFrame({"ppr" : ppr},index = x.index)
    return(dftemp)

def compute_pprt(x):
    nstims = len(x)
    firstpeakt = x[x["istim"] == 0]["peakt"].to_numpy()
    if (nstims>1):
        lastpeakt = x[x["istim"] == nstims-1]["peakt"].to_numpy()
        pprt = np.repeat((lastpeakt/firstpeakt),(nstims))
    else:
        pprt = 0
    dftemp = pd.DataFrame({"pprt" : pprt},index = x.index)
    return(dftemp)

# compute simple statistics functions
def nanmean(x):
    val = x.mean(axis=0,skipna=True,numeric_only=True)
    return(val)

def nanvar(x):
    val = x.var(axis=0,skipna=True,numeric_only=True)
    return(val)

def nanstd(x):
    val =x.std(axis=0,skipna=True,numeric_only=True)
    return(val)

def nansem(x):
    val = x.std(axis=0,skipna=True,numeric_only=True)/np.sqrt(x.count(axis=0,numeric_only=True))
    return(val)

def nancount(x):
    val = x.count(axis=0,numeric_only=True)
    return(val)

def compute_stats(x):
    df=pd.DataFrame({"nanmean":nanmean(x),
                     "mean":x.mean(),
                     "nanvar":nanvar(x),
                     "var":x.var(),
                     "nanstd":nanstd(x),
                     "std":x.std(),
                     "nansem":nansem(x),
                     "nancount":nancount(x),
                     "count":x.count()})
    return(df)


# Set some pandas options
pd.set_option('display.notebook_repr_html', False)
# pd.set_option('display.max_columns', 10)
# pd.set_option('display.max_rows', 10)

# open the csv file containing the main dataframe
datapath = '/Users/macbookair/goofy/data/beiquelab/glu_uncage_ca1/' 
fname = 'ephys_analysis_uncage_pairedpulse_V3.csv'
with open(datapath+fname,'r') as csvfile:
    df = pd.read_csv(csvfile)

df["isi"] = df["isi"] *1000
df["isi" ] = df["isi"].astype('int64')
df["istim" ] = df["istim"].astype('int64')
df["irun"] = df.groupby(["clamp","spineid","isi","istim"]).apply(lambda x:pd.DataFrame({"len":np.arange(1,len(x)+1)},index=x.index))
df[["ppr"]] = df.groupby(["clamp","fileid"]).apply(compute_ppr)
df[["pprt"]] = df.groupby(["clamp","fileid"]).apply(compute_pprt)

# save the DataFrame
df.sort_values(["spineid","istim","isi"],inplace=True)
df.to_csv(datapath+"ephys_analysis_uncage_pairedpulse_cclampV3out.csv",index=False)

# -----------------------
# plots: PPR, EPSCP versus ISI
param = "ppr"
istim = 0
dfstats = df[(df["istim"]==istim) & (df["irun"] == 1) & (df["clamp"] == "CC")].groupby(["isi"])["ppr","peak","pprt","peakt"].apply(compute_stats)
# create a new MultiIndex for prpt
dfmidx = df.set_index(["spineid","isi"],drop=True)
statmean = "nanmean"
staterror = "nansem"
pltstats = dfstats.xs((param),level=1,axis=0)[["nanmean","nansem","nanstd"]]
pltstats["isi"] = pltstats.index
pltstats = pltstats[(pltstats["isi"]>0) & (pltstats["isi"]<=800)]
print(pltstats)

# ----------------------------------
# plotting ISI versus PPR
# fh = plt.figure(figsize=(7,4))
# ah = fh.add_subplot(111)
# ahpos = ah.get_position()
# ah.set_position([ahpos.x0-0.01,ahpos.y0-0.01,ahpos.width,ahpos.height])
# sns.scatterplot(x="isi",y=param,hue="spineid",data=df[(df["istim"] == istim) & (df["irun"] == 1) & (df["isi"] > 0) & (df["isi"]<=800)],ax=ah,s=100)
# ah.plot([50,800],[1,1],linewidth=1,linestyle='--',color='gray')
# ah.errorbar(pltstats["isi"],pltstats[statmean],pltstats[staterror],color="k",linewidth=2)
# fh,ah = format_plot(fh,ah,"Inter-stimulus interval (ms)","Paired-pulse ratio \n (EPSP2/EPSP1)","")
# # fh,ah = format_plot(fh,ah,"Inter-stimulus interval (ms)","$1^{st}$ EPSP amplitude (mV)","")
# # fh,ah = format_plot(fh,ah,"Inter-stimulus interval (ms)","$2^{nd}$ EPSP amplitude (mV)","")
# xticks = [50,100,200,400,600,800]
# xticklabels = ["50","100","200","400","600","800"]
# # ah.set_xlim([40,900])
# ah.set_xticks(xticks)
# ah.set_xticklabels(xticklabels)
# yticks = [0,1,2,3]
# yticklabels = ["0","1","2","3"]
# ah.set_yticks(yticks)
# ah.set_yticklabels(yticklabels)
# ah.set_ylim([0,3])
# # Shrink current axis by 20%
# box = ah.get_position()
# # Put a legend to the right of the current axis
# ah.legend().set_visible(False)
# # ah.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()
# fh.savefig(datapath+"glu_uncage_ppr_isi_find_peak.png",transparent=True,dpi=300)
# # fh.savefig(datapath+"glu_uncage_2nd_epsp_isi_find_peak.png",transparent=True,dpi=300)
# # perform paired t-test
# g1 = df[(df["istim"] == istim) & (df["isi"] == 100) & (df["irun"] == 1)][param].to_numpy()
# g2 = df[(df["istim"] == istim) & (df["isi"] == 200) & (df["irun"] == 1)][param].to_numpy()
# pval = scipystats.ttest_ind(g1,g2)
# print("pvalue: ",pval) 
# plt.show()

# -------------------------
# plot bar EPSC amplitude
# df4plot = df[df["clamp"] == "VC"]
# fh = plt.figure(figsize=(4,6))
# ah = fh.add_subplot(111)
# ahpos = ah.get_position()
# ah.set_position([ahpos.x0-0.01,ahpos.y0-0.01,ahpos.width,ahpos.height])
# # ah.plot(np.repeat([0],len(df4plot)),df4plot)
# fh,ah = format_plot(fh,ah,"","EPSC amplitude (pA)","")
# ah.get_xaxis().set_visible(False)
# ah.set_frame_on(False)
# sns.barplot(x="istim",y="peak",color="grey",data=df4plot,ci=None)
# sns.scatterplot(x=np.repeat([0],len(df4plot)),y="peak",hue="spineid",data=df4plot,ax=ah,s=80)
# # ah.set_ylim([0,3])
# ah.set_xticks([-1,1])
# ah.legend().set_visible(False)
# plt.show()

# ------------------------
# plot correlation between EPSC and EPSP amplitude
# fh = plt.figure(figsize=(5,5))
# ah = fh.add_subplot(111)
# pltdf = pd.DataFrame()
# pltdf["epsppeak"] = df[(df["clamp"]=="CC") & (df["istim"]==0) & (df["irun"]==1)].groupby(["spineid"])["peak"].mean()
# pltdf["epscpeak"] = df[(df["clamp"]=="VC") & (df["istim"]==0) & (df["irun"]==1)].groupby(["spineid"])["peak"].mean()
# pltdf["spineid"] = pltdf.index
# outlier = ["20200305_C1S4"]
# pltdf = pltdf[~pltdf["spineid"].isin(outlier)]
# print(pltdf)
# # linear fit to data
# x = pltdf["epscpeak"][:,np.newaxis]
# x = np.concatenate((x,np.ones((x.shape[0],1))),axis=1)
# y = pltdf["epsppeak"][:,np.newaxis]
# print(x.shape,y.shape)
# m1,c1 = np.linalg.lstsq(x,y,rcond=None)[0]
# print("slope: ",m1*1e3," pS")
# yfit = m1*x + c1
# covariance = np.cov(pltdf["epscpeak"],pltdf["epsppeak"])
# corr_value,corr_pvalue = scipystats.pearsonr(pltdf["epscpeak"],pltdf["epsppeak"])
# print(covariance)
# print(corr_value,corr_pvalue)
# sns.scatterplot(x="epscpeak",y="epsppeak",hue="spineid",data=pltdf,ax=ah,s=100)
# fh,ah = format_plot(fh,ah,"\nEPSC amplitude (pA)","EPSP amplitude (mV)\n")
# ah.plot(x,yfit,color='k',linestyle='dashdot',linewidth=1)
# ah.get_legend().remove()
# # Shrink current axis by 20%
# box = ah.get_position()
# # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# # tight_layout(fig, rect=[0, 0, 0.5, 1])
# ah.set_position([box.x0+0.12, box.y0+0.1, box.width * 0.9, box.height*0.9])
# ah.set_xlim([-50,0])
# ah.set_ylim([0,2])
# ah.text(-45,2.1,'Synaptic conductance = '+str(round(m1[0]*1e3,2))+' pS',fontsize=14)
# ah.text(-45,1.95,'Covariance = '+str(round(covariance[0,1],2)),fontsize=14)
# ah.text(-45,1.8,'Correlation = '+str(round(corr_value,2)) + ", P value = " + str(round(corr_pvalue,3)),fontsize=14)
# fh.savefig(datapath+"glu_uncage_epsc_epsp.png",transparent=True,dpi=300)
# plt.show()


# ------------------------
# plot correlation between EPSC amplitude and pprt
# fh = plt.figure(figsize=(6,5))
# ah = fh.add_subplot(111)
# pltdf = pd.DataFrame()
# pltdf["epscpeak"] = df[(df["clamp"]=="VC") & (df["istim"]==0) & (df["irun"]==1)].groupby(["spineid"])["peak"].mean()
# pltdf["pprt"] = df[(df["clamp"]=="CC") & (df["istim"]==0) & (df["irun"]==1) & (df["isi"] <= 100)].groupby(["spineid"])["pprt"].mean()
# pltdf["spineid"] = pltdf.index
# # remove row with nan
# nanspines = pltdf[pltdf["pprt"].isnull()]["spineid"]
# # outlier = ["20200305_C1S2"]
# pltdf = pltdf[~pltdf["spineid"].isin(nanspines)]
# print(pltdf)
# # linear fit to data
# x = pltdf["epscpeak"][:,np.newaxis]
# x = np.concatenate((x,np.ones((x.shape[0],1))),axis=1)
# y = pltdf["pprt"][:,np.newaxis]
# print(x.shape,y.shape)
# m1,c1 = np.linalg.lstsq(x,y,rcond=None)[0]
# yfit = m1*x + c1
# covariance = np.cov(pltdf["epscpeak"],pltdf["pprt"])
# corr_value,corr_pvalue = scipystats.pearsonr(pltdf["epscpeak"],pltdf["pprt"])
# print(covariance)
# print(corr_value,corr_pvalue)
# sns.scatterplot(x="epscpeak",y="pprt",hue="spineid",data=pltdf,ax=ah,s=100)
# fh,ah = format_plot(fh,ah,"\nEPSC amplitude (pA)","Paired-pulse ratio\n",title=" ISI = 50-100ms" )
# ah.plot(x,yfit,color='k',linestyle='dashdot',linewidth=1)
# ah.get_legend().remove()
# # Shrink current axis by 20%
# box = ah.get_position()
# # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# # tight_layout(fig, rect=[0, 0, 0.5, 1])
# ah.set_position([box.x0+0.12, box.y0+0.1, box.width * 0.9, box.height*0.9])
# ah.set_xlim([-50,0])
# ah.set_ylim([0,2])
# ah.text(-40,1.8,'Covariance = '+str(round(covariance[0,1],2)),fontsize=14)
# ah.text(-40,1.65,'Correlation = '+str(round(corr_value,2)) + ", P value = " + str(round(corr_pvalue,2)),fontsize=14)
# fh.savefig(datapath+"glu_uncage_epsc_pprt_isi50_100ms.png",transparent=True,dpi=300)
# plt.show()

# ------------------------
# plot correlation between EPSC amplitudes computed without and with deconvolution template match
fh = plt.figure(figsize=(6,5))
ah = fh.add_subplot(111)
pltdf = pd.DataFrame()
isival = 200
pltdf["epsppeak"] = df[(df["clamp"]=="CC") & (df["istim"]==1) & (df["irun"]==1) & (df["isi"] == isival)].groupby(["spineid"])["peak"].mean()
pltdf["epsppeakt"] = df[(df["clamp"]=="CC") & (df["istim"]==1) & (df["irun"]==1) & (df["isi"] == isival)].groupby(["spineid"])["peakt"].mean()
pltdf["spineid"] = pltdf.index
# remove row with nan
nanspines = pltdf[pltdf["epsppeak"].isnull()]["spineid"]
pltdf = pltdf[~pltdf["spineid"].isin(nanspines)]
print(pltdf)
# linear fit to data
x = pltdf["epsppeak"][:,np.newaxis]
x = np.concatenate((x,np.ones((x.shape[0],1))),axis=1)
y = pltdf["epsppeakt"][:,np.newaxis]
print(x.shape,y.shape)
m1,c1 = np.linalg.lstsq(x,y,rcond=None)[0]
yfit = m1*x + c1
covariance = np.cov(pltdf["epsppeak"],pltdf["epsppeakt"])
corr_value,corr_pvalue = scipystats.pearsonr(pltdf["epsppeak"],pltdf["epsppeakt"])
print(covariance)
print(corr_value,corr_pvalue)
sns.scatterplot(x="epsppeak",y="epsppeakt",hue="spineid",data=pltdf,ax=ah,s=100)
fh,ah = format_plot(fh,ah,"\n Raw $2^{nd}$EPSP amplitude (mv)","Deconvolved $2^{nd}$EPSP peak (mv)\n",title=" ISI = " + str(round(isival)) + "ms" )
ah.plot(x,yfit,color='k',linestyle='dashdot',linewidth=1)
ah.plot(x,x,color='pink',linestyle=':',linewidth=1)
ah.get_legend().remove()
# Shrink current axis by 20%
box = ah.get_position()
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# tight_layout(fig, rect=[0, 0, 0.5, 1])
ah.set_position([box.x0+0.12, box.y0+0.1, box.width * 0.9, box.height*0.9])
# ah.set_xlim([-50,0])
# ah.set_ylim([0,2])
# ah.text(-40,1.8,'Covariance = '+str(round(covariance[0,1],2)),fontsize=14)
# ah.text(-40,1.65,'Correlation = '+str(round(corr_value,2)) + ", P value = " + str(round(corr_pvalue,2)),fontsize=14)
fh.savefig(datapath+"glu_uncage_2nd_epsp_raw_convolved_isi200ms.png",transparent=True,dpi=300)
plt.show()
