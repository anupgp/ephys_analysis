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

# ------------------------
# paths to ephys data
path_ephysdata="/home/anup/gdrive-beiquelabdata/Ephys Data/Olympus 2P/Anup/" # common to all ephys data
# -------------------------
# MNI-testing
# path to ephys data master excelfile
path_masterdf="/home/anup/jclabproject/mni-testing/" # local to the ephys project: Ca1 single spine
file_masterdf = "mni_testing_uncaging_compare.xlsx" # local to the ephys project
path_fig = os.path.join(path_masterdf)
# -----------------------
# specify columns to load
loadcolumns = ["expdate","clamp","ephysfile","mni","group","neuron"]
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
offsetdur = 0.2
sweepdur = 1                 # trace duration poststim

for expdate in expdates:
    print(expdate)
    fh = plt.figure(figsize=(8,5))
    ah = fh.add_subplot(111)
    ephysfiles = masterdf[(masterdf["expdate"]==expdate) & (masterdf["clamp"]=="vc")]["ephysfile"].to_list()
    for ephysfile in ephysfiles:
        group = masterdf[(masterdf["expdate"]==expdate) & (masterdf["ephysfile"] == ephysfile)]["group"].to_list()[0]
        neuron = masterdf[(masterdf["expdate"]==expdate) & (masterdf["ephysfile"] == ephysfile)]["neuron"].to_list()[0]
        clamp = masterdf[(masterdf["expdate"]==expdate) & (masterdf["ephysfile"] == ephysfile)]["clamp"].to_list()[0]
        mni = masterdf[(masterdf["expdate"]==expdate) & (masterdf["ephysfile"] == ephysfile)]["mni"].to_list()[0]
        fnamefull = os.path.join(path_ephysdata,str(expdate),"".join(("c",str(neuron))),"".join((str(ephysfile),".abf")))
        print(fnamefull)
        ephys = EphysClass(fnamefull,loaddata=True,clampch=clampchannel,resch=reschannel,stimch=stimchannel)
        ephys.extract_indices_holdingsteps(1,1)
        ephys.extract_stim_props()
        t,yy = ephys.extract_response(-0.05,0.1)
        t,yy2 = ephys.extract_response(0.05,0.2)
        yy = yy - yy[0,:]
        yy2 = yy2 - yy2[0,:]
        ah.plot(t*1000,yy,color="black",label="MNI-Femtonics")
        ah.plot(t*1000,yy2,color="red",label="MNI-Tocris")
        # -------------
    # -------------
fontsize=16
ah.set_xlabel("\nTime (msec)",fontsize=fontsize)
ah.set_ylabel("EPSC (pA) \n",fontsize=fontsize)
xticks = [0,50,100]
ah.set_xticks(xticks)
ah.set_xticklabels(xticks,fontsize=fontsize)
ylabels = ah.get_yticks()
ah.set_yticklabels(ylabels,fontsize=fontsize)
ah.spines["right"].set_visible(False)
ah.spines["top"].set_visible(False)
ah.set_title("Representative EPSC traces",fontsize=fontsize)
loc = "lower right"
handles, labels = ah.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ah.legend(by_label.values(), by_label.keys(),loc = loc, fontsize=fontsize)
# ah.legend(loc="upper right",fontsize=fontsize)
fh.tight_layout()
# ----------
fh_name = "".join((str(expdate),"_c",str(neuron),"_EPSC_traces_femtonics_vs_tocris.png"))
fh.savefig(os.path.join(path_fig,fh_name))
plt.show()
