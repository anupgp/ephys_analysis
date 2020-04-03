import numpy as np
import pandas as pd
import math
import seaborn as sns
import re
from scipy import stats as scipystats

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
    lastpeak = x[x["istim"] == 1]["peak"].to_numpy()
    firstpeak = x[x["istim"] == 0]["peak"].to_numpy()
    ppr = np.repeat((lastpeak/firstpeak),(nstims))
    dftemp = pd.DataFrame({"ppr" : ppr},index = x.index)
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
fname = 'ephys_analysis_uncage_pairedpulse_cclampV1.csv'
with open(datapath+fname,'r') as csvfile:
    df = pd.read_csv(csvfile)

df["isi"] = df["isi"] *1000
df["isi" ] = df["isi"].astype('int64')
df["istim" ] = df["istim"].astype('int64')
df["irun"] = df.groupby(["spineid","isi","istim"]).apply(lambda x:pd.DataFrame({"len":np.arange(1,len(x)+1)},index=x.index))
df[["ppr"]] = df.groupby(["fileid"]).apply(compute_ppr)

# save the DataFrame
df.sort_values(["spineid","istim","isi"],inplace=True)
df.to_csv(datapath+"ephys_analysis_uncage_pairedpulse_cclampV1out.csv",index=False)
# -----------------------

dfstats = df[(df["istim"]==1) & (df["irun"] == 1)].groupby(["isi"])["ppr","peak"].apply(compute_stats)
print(dfstats.head)

# create a new MultiIndex for prpt
dfmidx = df.set_index(["spineid","isi"],drop=True)
statmean = "nanmean"
staterror = "nansem"
param = "ppr"
fh = plt.figure(figsize=(7,4))
ah = fh.add_subplot(111)
ahpos = ah.get_position()
ah.set_position([ahpos.x0-0.01,ahpos.y0-0.01,ahpos.width,ahpos.height])
sns.scatterplot(x="isi",y=param,hue="spineid",data=df[(df["istim"] == 1) & (df["isi"]<=800)],ax=ah,s=80)
ah.plot([50,800],[1,1],linewidth=1,linestyle='--',color='gray')
pltstats = dfstats.xs((param),level=1,axis=0)[["nanmean","nansem","nanstd"]]
pltstats["isi"] = pltstats.index
pltstats = pltstats[pltstats["isi"]<=800]
ah.errorbar(pltstats["isi"],pltstats[statmean],pltstats[staterror],color="k",linewidth=2)
fh,ah = format_plot(fh,ah,"Inter-stimulus interval (ms)","Paired-pulse ratio \n (EPSP2/EPSP1)","")
# fh,ah = format_plot(fh,ah,"Inter-stimulus interval (ms)","$1^{st}$ EPSP amplitude (mV)","")
# fh,ah = format_plot(fh,ah,"Inter-stimulus interval (ms)","$2^{nd}$ EPSP amplitude (mV)","")
xticks = [50,100,200,400,600,800]
xticklabels = ["50","100","200","400","600","800"]
# ah.set_xlim([40,900])
ah.set_ylim([0,3])
ah.set_xticks(xticks)
ah.set_xticklabels(xticklabels)
# Shrink current axis by 20%
box = ah.get_position()
# Put a legend to the right of the current axis
ah.legend().set_visible(False)
# ah.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
fh.savefig(datapath+"glu_uncage_ppr_isi.png")
# fh.savefig(datapath+"glu_uncage_2nd_epsp_isi.png")
plt.show()

