from ephys_class import EphysClass
import ephys_class
import ephys_functions as ephysfuns
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
# plt.ion()
# font_path = '/home/anup/.matplotlib/fonts/arial.ttf'
# fontprop = font_manager.FontProperties(fname=font_path)

matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.sans-serif'] = "Arial"
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)
import pandas as pd
import os
import datetime
# -------------------------
def plot_timecourse(t,avgs,noises,tt,yy,fh,ah,ah2,group,mni):
    label = "Baseline"
    color = "lightblue"
    sweepid = 5
    loc = "lower right"
    if (mni == "Tocris"):
        loc = "center right"
    if ((group == "mni")):
        # label = "".join((group.upper(),"-",mni))
        label = "".join(("MNI-",mni))
        color = "orange"
        sweepid = -30
        if(len(np.where(t>28)[0])>0):
            itmax = np.where(t>28)[0][0]
            t = t[0:itmax]
            avgs = avgs[0:itmax]
            noises = noises[0:itmax]
            # clip time to 15 mins
    ah.plot(t,avgs,'o',label=label,color=color)
    for s in range(len(t)):
        ah.plot([t[s],t[s]],[avgs[s]-noises[s]/2,avgs[s]+noises[s]/2],'-',color="grey")
    # -----------------
    ah2.plot(tt*1000,yy,color=color)
    # -----------------
    fontsize=16
    ah.set_xlabel("Time (mins)",fontsize=fontsize)
    ah.set_ylabel("Membrane potential (mV) \n",fontsize=fontsize)
    xlabels = ah.get_xticks()
    ah.set_xticklabels(xlabels,fontsize=fontsize)
    ylabels = ah.get_yticks()
    ah.set_yticklabels(ylabels,fontsize=fontsize)
    ah.spines["right"].set_visible(False)
    ah.spines["top"].set_visible(False)
    ah.set_title("Mean & deviation (max-min) of Mem. potential (750ms)",fontsize=fontsize)
    ah.legend(loc=loc,fontsize=fontsize)
    ah.set_ylim([-75,-52])
    # ----------
    fontsize=12
    ah2.set_xlabel("Time (ms)",fontsize=fontsize)
    ah2.set_ylabel("Mem. pot. (mV)",fontsize=fontsize)
    xlabels = ah2.get_xticks()
    ah2.set_xticklabels(xlabels,fontsize=fontsize)
    ylabels = ah2.get_yticks()
    ah2.set_yticklabels(ylabels,fontsize=fontsize)
    ah2.spines["right"].set_visible(False)
    ah2.spines["top"].set_visible(False)
    # ---------------
    fh.tight_layout()
    return(fh,ah)


# ------------------------
# paths to ephys data
path_ephysdata="/home/anup/gdrive-beiquelabdata/Ephys Data/Olympus 2P/Anup/" # common to all ephys data
# -------------------------
# Ca1 single spine
# path to ephys data master excelfile
path_masterdf="/home/anup/jclabproject/mni-testing/" # local to the ephys project: Ca1 single spine
file_masterdf = "mni_testing_mastersheet.xlsx" # local to the ephys project
# dfname = "hc_ca1_glu_uncage_spine_single_masterdata.csv"           # local to the ephys project
path_fig = os.path.join(path_masterdf)
# -----------------------
# specify columns to load
loadcolumns = ["expdate","dob","sex","animal","strain","region","area","group","neuron","clamp","ephysfile","mni"]
masterdf = pd.read_excel(os.path.join(path_masterdf,file_masterdf),header=0,usecols=loadcolumns,index_col=None,sheet_name=0)
masterdf = masterdf.dropna(how='all') # drop rows with all empty columns
masterdf = masterdf.reset_index(drop=True) # don't increment index for empty rows
print(masterdf.head())
df = pd.DataFrame()
# ----------------------------------
# provide channel names
reschannel = "ImLEFT"
clampchannel = "IN1"
stimchannel = "IN7"
expdates = masterdf["expdate"].unique()

for expdate in expdates:
    print(expdate)
    fh = plt.figure(figsize=(8,5))
    ah = fh.add_subplot(111)
    # ah2 = plt.axes([0,0,1,1])
    # ah2.set_axes_locator(InsetPosition(ah, [0.2,0.6,0.3,0.2]))
    # ah2.spines["right"].set_visible(False)
    # ah2.spines["top"].set_visible(False)
    starttime = 0
    ephysfiles = masterdf[(masterdf["expdate"]==expdate) & (masterdf["clamp"]=="cc")]["ephysfile"].to_list()
    for ephysfile in ephysfiles:
        group = masterdf[(masterdf["expdate"]==expdate) & (masterdf["ephysfile"] == ephysfile)]["group"].to_list()[0]
        neuron = masterdf[(masterdf["expdate"]==expdate) & (masterdf["ephysfile"] == ephysfile)]["neuron"].to_list()[0]
        clamp = masterdf[(masterdf["expdate"]==expdate) & (masterdf["ephysfile"] == ephysfile)]["clamp"].to_list()[0]
        mni = masterdf[(masterdf["expdate"]==expdate) & (masterdf["ephysfile"] == ephysfile)]["mni"].to_list()[0]
        print(ephysfile)
        print(group)
        print(neuron)
        print(clamp)
        print(mni)
        isweeps = [-6,-7,-8,-9,-10]
        color = "lightblue"
        loc = "best"
        label = "Baseline"
        if (expdate == 20210714):
            ah.set_ylim([-70,-62])
            mni_traces = [-20,-21,-22,-23]
        if (expdate == 20210715):
            ah.set_ylim([-73,-50])
            mni_traces = [-30,-31,-32,-33]
        if (group == "mni"):
            color = "orange"
            label = "".join(("MNI-",mni))            
            isweeps = mni_traces
            # ----------------
        fnamefull = os.path.join(path_ephysdata,str(expdate),"".join(("c",str(neuron))),"".join((str(ephysfile),".abf")))
        print(fnamefull)
        ephys = EphysClass(fnamefull,loaddata=True,clampch=clampchannel,resch=reschannel,stimch=stimchannel)
        ephys.extract_indices_holdingsteps(1,1)
        t,avgs,noises,tt,yy = ephys.extract_noise(group)
        t = [datetime.datetime.fromtimestamp(item) for item in t]
        t = [starttime + (item-t[0]).total_seconds()/60 for item in t]
        starttime = t[-1]
        t = np.array(t)
        # fh,ah = plot_timecourse(t,avgs,noises,tt,yy,fh,ah,ah2,group,mni)
        tt,yy = ephys.extract_traces(isweeps)
        ah.plot(tt*1000,yy,color=color,label=label)
        # -----------------
    fontsize=16
    ah.set_xlabel("\nTime (msec)",fontsize=fontsize)
    ah.set_ylabel("Membrane potential (mV) \n",fontsize=fontsize)
    xticks = [0,500,1000,1500]
    ah.set_xticks(xticks)
    ah.set_xticklabels(xticks,fontsize=fontsize)
    ylabels = ah.get_yticks()
    ah.set_yticklabels(ylabels,fontsize=fontsize)
    ah.spines["right"].set_visible(False)
    ah.spines["top"].set_visible(False)
    ah.set_title("Representative traces of mem. potential",fontsize=fontsize)
    handles, labels = ah.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ah.legend(by_label.values(), by_label.keys(),loc = loc, fontsize=fontsize)
    # ah.legend(loc=loc,fontsize=fontsize)

    fh.tight_layout()
    # ----------
    fh_name = "".join((str(expdate),"_c",str(neuron),"_mem_potential_traces.png"))
    fh.savefig(os.path.join(path_fig,fh_name))
    # ----------------
    plt.show()
