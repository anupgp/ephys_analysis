from ephys_class import EphysClass
import ephys_class
import ephys_functions as ephysfuns
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import os
import re
import math
# from scipy.fft import fft,fftfreq,ifft
# from sklearn.decomposition import FastICA
from scipy import signal
from scipy import stats

class ephys_interactive:
    def __init__(self,t,y,istim,nstim):
        self.t = t
        self.y = y
        self.istim = istim
        self.nstim = nstim
        self.nsweep = self.y.shape[1]
        self.ibase = np.zeros((self.nstim,self.nsweep),dtype=np.int)
        self.ipeak = np.zeros((self.nstim,self.nsweep),dtype=np.int)
        self.peaks = np.zeros((self.nstim,self.nsweep),dtype=np.float)
        self.sweep = 0
        self.region = None
        self.stim = None
        self.initialize_plot()
        self.connect()
        plt.show()
        # >>>>>>>>>>>>>>>>>>>>>>>>>>
    def initialize_plot(self):
        self.fh = plt.figure()
        self.ah = self.fh.add_subplot(111)
        self.ah.set_autoscale_on(True)
        self.ah.autoscale_view(True,True,True)
        self.yph, = self.ah.plot(self.t,self.y[:,self.sweep],color="black")
        ymin = self.y[:,self.sweep].min()
        ymax = self.y[:,self.sweep].max()
        self.sph = []
        for istim in range(self.nstim):
            self.sph.append(self.ah.plot([t[self.istim[istim,self.sweep]],t[self.istim[istim,self.sweep]]],[ymin,ymax],color='red')[0])
            # -----------
        # self.bph, = self.ah.plot([self.t[0],self.t[-1]],[ymin,ymin],color='black',linewidth=2)
        self.bph = []
        self.pph = []
        for i in range(self.nstim):
            self.bph.append(self.ah.plot(t[0],y[0,self.sweep],color='blue',linestyle=None,marker="o",markersize=8,alpha=0.5)[0])
            self.pph.append(self.ah.plot(t[0],y[0,self.sweep],color='red',linestyle=None,marker="o",markersize=8,alpha=0.5)[0])
            # -------------
        self.th = self.ah.text(0.5,1.05,"Sample",transform=self.ah.transAxes,bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},ha="center")
        # >>>>>>>>>>>>>>>>>>>>>>>>>>
    def plot_sweep(self):
        ymin = self.y[:,self.sweep].min()
        ymax = self.y[:,self.sweep].max()
        self.yph.set_xdata(self.t)
        self.yph.set_ydata(self.y[:,self.sweep])
        self.yph.figure.canvas.draw()
        # self.ph1.figure.canvas.flush_events()
        # plt.draw()
        for istim,ph in zip(range(self.nstim),self.sph):
            ph.set_xdata([t[self.istim[istim,self.sweep]],t[self.istim[istim,self.sweep]]])
            ph.set_ydata([ymin,ymax])
            ph.figure.canvas.draw()
        # ------------------
        for istim in range(self.nstim):
            self.bph[istim].set_xdata(self.t[self.ibase[istim,self.sweep]])
            self.bph[istim].set_ydata(self.y[self.ibase[istim,self.sweep],self.sweep])
            self.bph[istim].figure.canvas.draw()
        # ------------------
        for istim in range(self.nstim):
            self.pph[istim].set_xdata(self.t[self.ipeak[istim,self.sweep]])
            self.pph[istim].set_ydata(self.y[self.ipeak[istim,self.sweep],self.sweep])
            self.pph[istim].figure.canvas.draw()
        # ------------------
        self.ah.autoscale()
        self.update_title()
        # >>>>>>>>>>>>>>>>>>>>>>>>>>
    def __call__(self,event):
        self.event = event
        # print(event)
        if((event.inaxes is not None) and (self.region == "baseline") and (event.xdata<=self.t[-1]) & (self.stim is not None)):
            # self.ibase[self.stim,self.sweep] = [event.xdata]
            # print(event.xdata,event.ydata)
            self.plot_baseline()
            self.update_title()
            # --------------
        if ((event.inaxes is not None) and (event.name == 'key_press_event')):
            self.key = self.event.key
            self.get_key_function()
            # -------------
        if ((event.name == "button_press_event") and (event.xdata<=self.t[-1]) and (self.region == "baseline") and (self.stim is not None)):
            # print("button press event")
            ix = int(np.where(self.t>=self.event.xdata)[0][0])
            self.ibase[self.stim-1,self.sweep] = ix
            self.update_title()
            self.region = None
            self.stim = None

            # --------------
        if((event.inaxes is not None) and (self.region == "peak") and (event.xdata<=self.t[-1]) & (self.stim is not None)):
            # print(event.xdata,event.ydata)
            self.plot_peak()
            self.update_title()
            # --------------
        if ((event.name == "button_press_event") and (event.xdata<=self.t[-1]) and (self.region == "peak") and (self.stim is not None)):
            print("button press event")
            ix = int(np.where(self.t>=self.event.xdata)[0][0])
            self.ipeak[self.stim-1,self.sweep] = ix
            self.update_title()
            self.region = None
            self.stim = None
            # --------------
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def get_key_function(self):
        switcher = {
            'left':self.plot_previous_sweep, # plots the previous sweep
            'right':self.plot_next_sweep, # plots the next sweep
            '1':self.stim_select,        # select the stimulation 1:2
            '2':self.stim_select,        # select the stimulation 1:2
            'b':self.select_baseline,
            'a':self.select_peak
        }
        func = switcher.get(self.key,lambda:"Invalid key")
        return(func())
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def plot_previous_sweep(self):
        if(self.sweep >0):
            self.sweep = self.sweep - 1
            # ------------
        self.plot_sweep()
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def plot_next_sweep(self):
        if(self.sweep < self.nsweep-1):
            self.sweep = self.sweep + 1
            # ------------
        self.plot_sweep()
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def stim_select(self):
        self.stim = int(self.key)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def select_baseline(self):
        self.region = "baseline"
        print(self.region)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def select_peak(self):
        self.region = "peak"
        print(self.region)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def update_title(self):
        title = ""
        for istim in range(self.nstim):
            title = title + "Sweep: " + str(self.sweep) + " Stim: " + str(istim+1) + " Baseline: " +  '%.2f' % self.y[self.ibase[istim,self.sweep],self.sweep]
            title = title + " Peak: " +  '%.2f' % self.y[self.ipeak[istim,self.sweep],self.sweep] + "\n"
            # ---------------
        self.th.set_text(title)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def plot_baseline(self):
        # self.bph.set_ydata(self.event.ydata)
        # self.bph.set_xdata([self.t[0],self.t[-1]])
        ydata = self.y[np.where(self.t>=self.event.xdata)[0][0],self.sweep]
        stim = self.stim-1
        self.bph[stim].set_xdata(self.event.xdata)
        self.bph[stim].set_ydata(ydata)
        self.bph[stim].figure.canvas.draw()
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def plot_peak(self):
        ydata = self.y[np.where(self.t>=self.event.xdata)[0][0],self.sweep]
        stim = self.stim-1
        self.pph[stim].set_xdata(self.event.xdata)
        self.pph[stim].set_ydata(ydata)
        self.pph[stim].figure.canvas.draw()
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def return_data(self):
        for isweep in range(self.nsweep):
            for istim in range(self.nstim):
                self.peaks[isweep,istim] = y[isweep,base[isweep,istim]]-y[isweep,peak[isweep,istim]]
        return(self.peaks)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def connect(self):
        'connect to all the events'
        self.cidpress = self.fh.canvas.mpl_connect('button_press_event', self)
        self.cidrelease = self.fh.canvas.mpl_connect('button_release_event', self)
        self.cidmotion = self.fh.canvas.mpl_connect('motion_notify_event', self)
        self.cidkeypress = self.fh.canvas.mpl_connect('key_press_event',self)
        # self.cidfigenter = self.fig.canvas.mpl_connect('figure_enter_event', self.press)
        # self.cidfigleave = self.fig.canvas.mpl_connect('figure_leave_event', self.press)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def disconnect(self):
        'disconnect all the stored connection ids'
        self.fh.canvas.mpl_disconnect(self.cidpress)
        self.fh.canvas.mpl_disconnect(self.cidrelease)
        self.fh.canvas.mpl_disconnect(self.cidmotion)
        self.fh.canvas.mpl_disconnect(self.cidkeypress)
        # self.fig.canvas.mpl_disconnect(self.cidfigenter)
        # self.fig.canvas.mpl_disconnect(self.cidfigleave)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def __del__(self):
        print("".join(('An object of class ',self.__class__.__name__,' is deleted!')))
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>
# -------------------------
# paths to ephys data
path_ephysdata="/home/anup/gdrive-beiquelabdata/Ephys Data/Olympus 2P/Anup/" # common to all ephys data
# -------------------------
# Ca1 single spine
# path to ephys data master excelfile
path_masterdf="/home/anup/jclabproject/hc_ca1_glu_uncage_spine_single_epsp/" # local to the ephys project: Ca1 single spine
file_masterdf = "hc_ca1_glu_uncage_spine_single_mastersheet.xlsx" # local to the ephys project
dfname = "hc_ca1_glu_uncage_spine_single_masterdata.csv"           # local to the ephys project
# ------------------------
# Ca1 cluster spine
# path_masterdf="/home/anup/jclabproject/hc_ca1_glu_uncage_spine_cluster_epsp/" # local to the ephys project:  
# file_masterdf = "hc_ca1_glu_uncage_spine_cluster_mastersheet.xlsx" # local to the ephys project
# dfname = "hc_ca1_glu_uncage_spine_cluster_masterdata.csv"           # local to the ephys project
# -----------------------
path_fig = os.path.join(path_masterdf,"figures")
# -----------------------
# specify columns to load
loadcolumns1 = ["expdate","dob","sex","animal","strain","region","area","side","group","neuronid","dendriteid","spineid"]
loadcolumns2 = ["unselectfile","badsweeps","ephysfile","clamp","stimpower","isi"]
loadcolumns = loadcolumns1 + loadcolumns2
masterdf = pd.read_excel(os.path.join(path_masterdf,file_masterdf),header=0,usecols=loadcolumns,index_col=None,sheet_name=0)
masterdf = masterdf.dropna(how='all') # drop rows with all empty columns
masterdf = masterdf.reset_index(drop=True) # don't increment index for empty rows
# remove files that are marked to unselect from analysis
masterdf = masterdf[masterdf["unselectfile"]!=1]
# create a unique spine name
masterdf["spinename"] = ephysfuns.combine_columns(masterdf,['expdate','group','neuronid','dendriteid','spineid'])
# convert ISI to int
masterdf["isi"] = masterdf["isi"].astype(int)
spines = masterdf["spinename"].unique()
print("spines: ",spines)
print(masterdf.head())
roidf = pd.DataFrame()
# ----------------------
# provide channel names
reschannel = "ImLEFT"
clampchannel = "IN1"
stimchannel = "IN7"
ephysfile_ext = "abf"
offsetdur = 0.2
sweepdur = 1.5                 # trace duration poststim
nstim = 2
isis = []
# ------------------
for s in range(0,len(spines)):
    spine = spines[s]
    isis_uni = np.sort(masterdf[(masterdf["clamp"] == "cc") & (masterdf["spinename"] == spines[s])]["isi"].unique())
    print(isis_uni)
    for i in range(0,len(isis_uni)):
        ephysfiles = masterdf[(masterdf["clamp"] == "cc") & (masterdf["spinename"] == spines[s]) & (masterdf["isi"] == isis_uni[i])]["ephysfile"].to_list()
        for e in range(0,len(ephysfiles)):
            neuronid = list(masterdf.loc[masterdf["ephysfile"] == ephysfiles[e],"neuronid"].astype(int).astype(str))[0]
            expdate = list(masterdf.loc[masterdf["ephysfile"] == ephysfiles[e],"expdate"].astype(int).astype(str))[0]
            fnamefull = os.path.join(path_ephysdata,expdate,"".join(("C",neuronid)),ephysfiles[e])
            print(spines[s],isis_uni[i],ephysfiles[e])
            print(fnamefull)
            badsweeps = list(masterdf.loc[masterdf["ephysfile"] == ephysfiles[e],"badsweeps"].astype(str))[0].split(',')
            badsweeps = [int(item)-1 for item in badsweeps if item != 'nan'] # trace 1 has index 0
            print('badsweeps: {}'.format(badsweeps))
            ephys = EphysClass(fnamefull,loaddata=True,badsweeps=badsweeps,clampch=clampchannel,resch=reschannel,stimch=stimchannel)
            ephys.extract_indices_holdingsteps(1,1)
            ephys.extract_stim_props()
            si = ephys.si
            isis.append(int(ephys.isi*1000))
            t,y,tstims,istims = ephys.get_sweeps(offsetdur,sweepdur)
            ephys_measure =   ephys_interactive(t,y,istims,nstim)

            print(ephys_measure.ibase)
            print(ephys_measure.ipeak)
            
            # fh = plt.figure()
            # ah = fh.add_subplot(111)
            # ah.plot(t,y)
            

