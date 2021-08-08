from neo import AxonIO
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.gridspec as gridspec
import re
from scipy import interpolate
from scipy import signal
from scipy.signal import freqz, detrend
from scipy.signal import find_peaks
from scipy.fft import fft,fftfreq,ifft
# from scipy.signal import correlate
from scipy.optimize import curve_fit
import plotting_functions
import os
import emd
from sklearn.decomposition import FastICA
import datetime

def smooth(y,windowlen=3,window = 'hanning'):
    s = np.concatenate([y[windowlen-1:0:-1],y,y[-2:-windowlen-1:-1]])
    if (window == 'flat'):
        w = np.ones(windowlen,dtype='double')
    else:
        w = eval('np.'+window+'(windowlen)')
        
    yy = np.convolve(w/w.sum(),s,mode='same')[windowlen-1:-windowlen+1]
    return (yy)

def butter_bandpass(x,fl,fh,fs,order):
    fnyq = fs/2
    fl = fl/fnyq
    fh = fh/fnyq
    sos = signal.butter(N=order,Wn=[fl,fh],btype='bandpass',output='sos')
    xf = signal.sosfiltfilt(sos,x)
    return(xf)
# }

def expfn(x,a1,tau1,a2,tau2):
    y = ((a1*np.exp(-(x)/tau1)) + (a2*np.exp(-(x)/tau2))).flatten()
    return(y)

def alphafn(x,t0,a,tr,td):
    if (td == tr):
        td = tr + 0.1
    tp = t0 + (td*tr/(td-tr) * np.log10(td/tr))
    b = 1/(-np.exp(((t0-tp)/tr)) + np.exp(((t0-tp)/td)))
    y = a*b*(np.exp((t0-x)/td) - np.exp((t0-x)/tr))
    return(y)

class EphysClass:
    
    def __init__(self,fname,loaddata=True,badsweeps=-1,vclamp=None,cclamp = None,fid=None,clampch="",resch="",stimch=""):
        self.fname = fname
        self.fid = fid
        if(fid  == None):
            self.fid = re.search('[0-9]{1,10}.abf',fname)[0][0:-4]
            # ------------
        self.loaddata = loaddata
        self.badsweeps = badsweeps
        self.vclamp = vclamp
        self.cclamp = cclamp
        self.clampch = clampch
        self.resch = resch
        self.stimch = stimch
        # }
        if(not os.path.exists(self.fname)):
            print("The file does not exist! check the path!")
            return()
        # -----------
        self.reader = AxonIO(filename=self.fname)
        self.nblocks = self.reader.block_count()
        self.nsweeps = np.zeros(self.nblocks,dtype=np.int)
        self.si = 1/self.reader.get_signal_sampling_rate()
        self.samplesize = self.reader.get_signal_size(block_index=0,seg_index=0)
        self.tstart = self.reader.segment_t_start(block_index=0,seg_index=0)
        self.tstop = self.reader.segment_t_stop(block_index=0,seg_index=0)
        self.header = self.reader.header
        # get number of sweeps in each block
        for i in np.arange(0,self.nblocks):
            self.nsweeps[i] = self.reader.segment_count(i)
        if(len(self.nsweeps)==1):
            self.nsweeps = self.nsweeps[0]
            self.nchannels = self.reader.signal_channels_count()
            self.sres = np.zeros((int(self.nsweeps)))
            self.nstim = 0
            self.stimprop = [dict({"nstim":0,"isi":0,"istims":[],"tstims":[]}) for sweep in np.arange(0,self.nsweeps)]
            self.clampprop = [dict({"iclamps":[],"tclamps":[]}) for sweep in np.arange(0,self.nsweeps)]
            # initialize good sweeps
        self.goodsweeps = np.setdiff1d(np.arange(0,self.nsweeps),self.badsweeps)
        # extract channel/signal properties for each signal/channel from header
        channelspecs = self.reader.header["signal_channels"].dtype.names
        # print(channelspecs)
        # print(self.nchannels)
        self.channelspecs = [dict.fromkeys(channelspecs) for channel in np.arange(0,self.nchannels)]
        for i in np.arange(0,self.nchannels):
            self.channelspecs[i] = self.reader.header["signal_channels"][i]
            # ----            
            # get number of samples from the first block, first segment
        self.nsamples = self.reader.get_signal_size(block_index=0,seg_index=0)
        # print(self.channelspecs)
        # input()
        if(self.loaddata):
            self.data=self.__loaddata()
            # extract date/time of the file
        self.datetime = self.reader.raw_annotations["blocks"][0]["rec_datetime"]
        self.date = self.datetime.date().isoformat()
        self.time = self.datetime.time().isoformat()
        # ============================================

    def find_neg_peaks(self,resch,stimch):
        # -------------
        # a method to extract reponse peaks
        # get the stimulus times
        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch,channelspec['name'])]
        istimch = [channelspec['id'] for channelspec in self.channelspecs if re.search(stimch,channelspec['name'])]
        iholdsteps = self.iholdsteps
        stimprop = self.stimprop
        sweeps = self.goodsweeps
        si = self.si
        y = self.data[:,iresch,:]
        t = np.arange(0,y.shape[-1]*si,si).reshape(y.shape[-1],1)
        ystim = np.zeros((y.shape[-1],1),dtype=np.float)
        tstim = np.arange(0,y.shape[-1]*si,si)
        toffsetpre = 0.01
        ioffsetpre = int(toffsetpre/si)
        toffsetpost = 0.05
        ioffsetpost = int(toffsetpost/si)
        ypeaks = np.zeros((len(sweeps),1))
        fh = plt.figure()
        ah = fh.add_subplot(111)
        # ------------------
        for s in range(0,len(sweeps)):
            isweep = sweeps[s]
            istims = stimprop[isweep]["istims"]
            isi = stimprop[isweep]["isi"]
            # create stim starts and stops
            # start/end positions of each stimultions, including the dummy at the start
            ibegins = np.concatenate(([istims[0] - ioffsetpre],istims)) 
            iends = np.concatenate((istims,[istims[-1] + ioffsetpost]))
            # ----------------------
            ysub = smooth(self.data[isweep,iresch,iholdsteps[isweep]["pos"][0]:istims[0]][0],windowlen=61,window='hanning')
            tsub = np.arange(0,(len(ysub)+1)*self.si,self.si)[0:len(ysub)][np.newaxis].T
            m1,c1 = np.linalg.lstsq(np.concatenate((tsub,np.ones(tsub.shape)),axis=1),ysub,rcond=None)[0]
            # ---------------------
            tsweep = t[ibegins[0]:iends[-1]]
            ysweep = y[isweep,iresch,ibegins[0]:iends[-1]].reshape(tsweep.shape) - (m1*tsweep + c1)
            ysweep = ysweep - max(ysweep)
            ysweep = smooth(ysweep[:,0],windowlen=31,window="hanning").reshape(len(ysweep),1)
            ipeak = np.argmin(ysweep)
            tpeak = tsweep[ipeak,0]
            ypeak = ysweep[ipeak,0]
            ypeaks[s,0] = ypeak
            ah.plot(tsweep,ysweep)
            ah.plot(tpeak,ypeak,'o',markersize=10)
            # ----------------
        # }
        plt.show()
        return(ypeaks)
    # }
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def show(self,ichannels=[],isweeps=[]):
        # display traces
        if (len(isweeps) == 0):
            isweeps = np.arange(0,self.nsweeps)
        if (len(ichannels) == 0):
            ichannels = np.arange(0,self.nchannels)
            # load data if not already loaded
        if(not self.loaddata):
            self.data = self.__loaddata()
        if not (all(isinstance(x,int) for x in ichannels)):
            print("Channel indices must be integers")
            exit()
        # }
        t = np.arange(self.tstart,self.tstop,self.si)
        fig = plt.figure(figsize=(10,8),constrained_layout=False)
        spec = gridspec.GridSpec(nrows=len(ichannels),ncols=1,figure=fig,width_ratios=[1],height_ratios=[1,1,1])
        ax = []
        ax.append(fig.add_subplot(spec[0,0]))
        # ax.append(fig.add_subplot(len(ichannels),1,1))
        pannelcount=0
        # set figure title
        figtitle = "".join((re.search("(.*/+)(.*)",self.fname)[2],"\n","#sweeps = ",str(len(isweeps))))
        ax[0].set_title(figtitle)
        for i in ichannels:
            pannelcount = pannelcount + 1
            if(pannelcount>1):
                ax.append(fig.add_subplot(spec[pannelcount-1,0],sharex=ax[0]))
            # }
            channelname = self.channelspecs[i]["name"]    
            channelunit = self.channelspecs[i]["units"]
            channelylabel = channelname+' (' + channelunit + ')'
            ax[pannelcount-1].set_ylabel(channelylabel)
            for j in isweeps:
                ax[pannelcount-1].plot(t,self.data[j,i,:],label=str(j+1))
            # }
        # }
        # display average
        ax[0].plot(t,self.data[isweeps,0,:].mean(axis=0),color='k')
        # ax[0].legend()
        plt.show()
    # }    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def __loaddata(self):
        """loads all the sweeps and channels into a numpy ndarray: timeseries * sweeps * channels """
        self.data= np.zeros((self.nsweeps,self.nchannels,self.nsamples))
        blk = self.reader.read_block(block_index=0,lazy=False,signal_group_mode='split-all')
        # blk = self.reader.read_block(lazy=False)
        iseg = -1
        for seg in blk.segments:
            iseg = iseg + 1
            isig = -1
            for i, asig in enumerate(seg.analogsignals):
                isig = isig + 1
                asig = asig.magnitude
                self.data[iseg,isig,:] = asig[:,0]                
            # }
        # }
        return(self.data)
    # }
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  
    def info(self):
        channelnames = [channelspec['name'] for channelspec in self.channelspecs]
        channelunits = [channelspec['units'] for channelspec in self.channelspecs]
        print("==================================================================")
        print('fname:','\t',self.fname)
        print('fid:','\t',self.fid)
        print('VClamp:','\t',self.vclamp)
        print('CClamp','\t',self.cclamp)
        print('nblocks:','\t',self.nblocks)
        print('nsweeps:','\t',self.nsweeps)
        print('goodsweeps:\t',self.goodsweeps)
        print('nchannels','\t',self.nchannels)
        print('channelnames','\t',channelnames)
        print('channelunits','\t',channelunits)
        print('sampling interval:','\t',self.si)
        print('t_start','\t',self.tstart)
        print('t_stop','\t',self.tstop)
        print('sample size:','\t',self.samplesize)
        print('Date:','\t',self.date)
        print('Time:','\t',self.time)
        print('----------------------------------------')
        print('iholdsteps:\t',self.iholdsteps)
        print('nstim:\t',self.nstim)
        print('stimprop:\t',self.stimprop)
        # print('isi:\t',self.isi)
        print('Series resistance','\t',self.sres,' Mohms')
        print("==================================================================")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
    def extract_indices_holdingsteps(self,NstepsPos=1,NstepsNeg=1):
        clampch = self.clampch
        iclamp = [channelspec['id'] for channelspec in self.channelspecs if re.search(clampch,channelspec['name'])]
        t = np.arange(self.tstart,self.tstop,self.si)
        self.iholdsteps = [dict({"neg":np.zeros(NstepsNeg,dtype=np.uint),"pos":np.zeros(NstepsPos,dtype=np.uint)}) for sweep in np.arange(0,self.nsweeps)]
        # -----------------
        for isweep in np.arange(0,self.nsweeps):
            clamp = self.data[isweep,iclamp,:].T # [isweep,ichannel,isample]
            dclamp = np.diff(clamp,axis=0)
            dclamp.resize(max(dclamp.shape))
            # find all negative peaks above threshold
            ineg_peaks = find_peaks(-dclamp,height=5,threshold=5)[0][0:NstepsNeg]
            # find all positive peaks above threshold
            ipos_peaks = find_peaks(dclamp,height=5,threshold=5)[0][0:NstepsPos]
            self.iholdsteps[isweep]["neg"] = ineg_peaks # negative direction
            self.iholdsteps[isweep]["pos"] = ipos_peaks # positive direction
            # plotting
            # fh = plt.figure()
            # ah = fh.add_subplot(111)
            # ah.plot(t,clamp)
            # ah.plot(t[0:len(dclamp)],dclamp)
            # inegpeaks = self.iholdsteps[isweep]["neg"]
            # ipospeaks = self.iholdsteps[isweep]["pos"]
            # print(inegpeaks,ipospeaks)
            # ah.plot(t[inegpeaks],dclamp[inegpeaks],marker='o',color='r',markersize=10)
            # ah.plot(t[ipospeaks],dclamp[ipospeaks],marker='o',color='g',markersize=10)
            # plt.show()
        # }
        self.vclamp = self.data[:,iclamp,:].mean().mean()
    # }
    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>         
    def seriesres_voltageclamp(self,resch,clampch):        
        # Measure series resistance in a voltage-clamp sweep"
        ires = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch,channelspec['name'])]
        iclamp = [channelspec['id'] for channelspec in self.channelspecs if re.search(clampch,channelspec['name'])]
        # fh = plt.figure()
        # ah = plt.subplot(111)
        t = np.arange(self.tstart,self.tstop,self.si)
        # istart = np.where(t>=tstart)[0][0]
        # istop = np.where(t>=tstop)[0][0]
        # print(istart,istop)
        for isweep in np.arange(0,self.nsweeps):
            y = self.data[isweep,ires,:].T # [isweep,ichannel,isample]
            dy = np.diff(y,axis=0)
            dy.resize(max(dy.shape))
            clamp = self.data[isweep,iclamp,:].T # [isweep,ichannel,isample]
            dclamp = np.diff(clamp,axis=0)
            dclamp.resize(max(dclamp.shape))
            # fh = plt.figure()
            # ah = fh.add_subplot(111)
            # ah.plot(t[:-1],-dclamp)
            # plt.show()
            # find all negative peaks above threshold
            inegpeak_clamp = find_peaks(-dclamp,height=5,threshold=5)[0][0]
            ipospeak_clamp = find_peaks(dclamp,height=5,threshold=5)[0][0]
            deltaclamp = abs(clamp[inegpeak_clamp]-clamp[ipospeak_clamp])*1e-3 # in volts
            ipospeak_y = inegpeak_clamp-20 + np.argmax(y[inegpeak_clamp-20:ipospeak_clamp+20])
            inegpeak_y = inegpeak_clamp-20 + np.argmin(y[inegpeak_clamp-20:ipospeak_clamp+20])
            baseliney = np.mean(y[0:inegpeak_y-20])
            peaky = -(y[inegpeak_y]-baseliney)*1e-12
            sres = (deltaclamp/peaky)*1e-6 # in mega ohms
            self.sres[isweep] = sres
            print("deltaclamp: {} deltay: {}, sres: {}".format(deltaclamp,peaky,sres))
            # fh = plt.figure()
            # ah = fh.add_subplot(111)
            # ah.plot(t,clamp)
            # ah.plot(t,y)
            # ah.plot(t[ipospeak_clamp],clamp[ipospeak_clamp],'o','red')
            # ah.plot(t[inegpeak_clamp],clamp[inegpeak_clamp],'o','blue')
            # ah.plot(t[ipospeak_y],y[ipospeak_y],'o','red')
            # ah.plot(t[inegpeak_y],y[inegpeak_y],'o','blue')
            # plt.show()
            # }
            
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def seriesres_currentclamp(self,resCh,clampCh):
        # Measure series resistance in a current-clamp sweep"
        ires = [channelspec['id'] for channelspec in self.channelspecs if re.search(resCh,channelspec['name'])]
        iclamp = [channelspec['id'] for channelspec in self.channelspecs if re.search(clampCh,channelspec['name'])]
        # fh = plt.figure()
        # ah = plt.subplot(111)
        t = np.arange(self.tstart,self.tstop,self.si)
        # istart = np.where(t>=tstart)[0][0]
        # istop = np.where(t>=tstop)[0][0]
        # print(istart,istop)
        for isweep in np.arange(0,self.nsweeps):
            v = self.data[isweep,ires,:].T # [isweep,ichannel,isample]
            dv = np.diff(v,axis=0)
            dv.resize(max(dv.shape))
            clamp = self.data[isweep,iclamp,:].T # [isweep,ichannel,isample]
            dclamp = np.diff(clamp,axis=0)
            dclamp.resize(max(dclamp.shape))
            # find all negative peaks above threshold
            ineg_peaksall, _ = find_peaks(-dclamp,height=10,threshold=10)
            # print("ineg_peaksall:\t",ineg_peaksall)
            # find the negative peak after tstart
            # ineg_peak = ineg_peaksall[np.where(ineg_peaksall>=istart)[0][0]]
            ineg_peak = ineg_peaksall[0]
            # find all positive peaks above threshold
            ipos_peaksall, _ = find_peaks(dclamp,height=10,threshold=10)
            # find the positive peak before tstop
            # ipos_peak = ipos_peaksall[np.where(ipos_peaksall>istop)[0][0]]
            ipos_peak = ipos_peaksall[0]
            # print("ipos_peaksall:\t",ipos_peaksall)
            # find the difference between before and after the current jump
            vdel_pos = abs(v[ipos_peak+8]-v[ipos_peak])*1e-3 # in volts
            vdel_neg = abs(v[ineg_peak+8]-v[ineg_peak])*1e-3 # in volts
            idel_pos = abs(clamp[ipos_peak+1]-clamp[ipos_peak])*1e-12 # in amperes
            idel_neg = abs(clamp[ineg_peak+1]-clamp[ineg_peak])*1e-12 # in amperes
            vdel = np.mean([vdel_pos,vdel_neg])
            idel = np.mean([idel_pos,idel_neg])
            sres = (vdel/idel)*1e-6 # in mega ohms
            self.sres[isweep] = sres
            # print('Series res:\t',self.sres[isweep],' Mega ohms')
            #     ah.plot(t,v)
            #     ah.plot([t[ipos_peak],t[ipos_peak+8]],[v[ipos_peak],v[ipos_peak+8]],color='k')
            #     ah.plot([t[ineg_peak],t[ineg_peak+8]],[v[ineg_peak],v[ineg_peak+8]],color='k')
            # plt.show()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def extract_res_props(self):
        # Find the peaks and other parameters of the respose (EPSP/EPSC)
        # fh = plt.figure()
        # ah1 = plt.subplot(211)
        # ah2 = plt.subplot(212,sharex=ah1)
        resch = self.resch
        stimch = self.stimch
        ires = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch,channelspec['name'])]
        itrg = [channelspec['id'] for channelspec in self.channelspecs if re.search(stimch,channelspec['name'])]
        t = np.arange(self.tstart,self.tstop,self.si)
        # find the trigger pulses
        triggerthres = 500
        tprestim = 0.02
        nprestim = int(tprestim/self.si)
        tpoststim = 0.2
        npoststim = int(tpoststim/self.si)
        for isweep in np.arange(0,self.nsweeps):
            tstims = self.stimprop[isweep]["tstims"]
            istims = self.stimprop[isweep]["istims"]
            ipre = istims[0]-nprestim
            ipost = istims[-1]+npoststim
            yt = self.data[isweep,itrg,ipre:ipost] # [isweep,ichannel,isample]
            yr = self.data[isweep,ires,ipre:ipost] # [isweep,ichannel,isample]
            tt = t[ipre:ipost]-t[ipre]
            yr.resize(max(yr.shape))
            yt.resize(max(yt.shape))

    def extract_stim_props(self):
        # get information about stimulation from the trigger channel
        # properties:
        # nstim: number of stimulations
        # istim: index of stimulations
        # tstims: time of each stimulation
        # isi: mean inter stimulus inverval
        stimch = self.stimch
        itrg = [channelspec['id'] for channelspec in self.channelspecs if re.search(stimch,channelspec['name'])]
        t = np.arange(self.tstart,self.tstop,self.si)
        # find the trigger pulses
        triggerthres = 100 # 3
        for isweep in np.arange(0,self.nsweeps):
            yt = self.data[isweep,itrg,:] # [isweep,ichannel,isample]
            yt.resize(self.samplesize)
            istims,_ = find_peaks(yt,height=triggerthres)
            if (len(istims)>0):
                self.nstim = len(istims)
                self.stimprop[isweep]["nstim"] = self.nstim
                [self.stimprop[isweep]["istims"].append(istim) for istim in istims]
                [self.stimprop[isweep]["tstims"].append(t[istim]) for istim in istims]
                self.stimprop[isweep]["isi"] = 0
                self.isi = 0
                if(len(self.stimprop[isweep]["tstims"])>1):
                    self.stimprop[isweep]["isi"] = np.mean(np.diff(self.stimprop[isweep]["tstims"]))
                    # compute average isi
                    self.isi = np.round(np.array([self.stimprop[sweep]["isi"] for sweep in self.goodsweeps]).mean(),2) # isi computed from only goodsweeps
                # }
            # }
        # }
    # }
    # ------------------
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def get_clampprop(self,resch,clampch):
        goodsweeps = self.goodsweeps
        iclampch = [channelspec['id'] for channelspec in self.channelspecs if re.search(clampch,channelspec['name'])]
        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch,channelspec['name'])]
        t = np.arange(self.tstart,self.tstop,self.si)
        yy = self.data[:,iclampch,:].T # [isweep,ichannel,isample]
        # find the start and end of the current pulse from the clamp currrent
        inegpeaks = np.zeros(self.nsweeps,dtype = np.int)
        ipospeaks = np.zeros(self.nsweeps,dtype = np.int)
        for isweep in np.arange(0,self.nsweeps):
            y = yy[:,0,isweep]
            dy = np.diff(y,n=1,axis=0)
            # find all negative peaks above threshold
            ineg_peaksall, _ = find_peaks(-dy,height=10,threshold=10)
            # find the negative peak after tstart
            # ineg_peak = ineg_peaksall[np.where(ineg_peaksall>=istart)[0][0]]
            inegpeaks[isweep] = ineg_peaksall[np.argmax(dy[ineg_peaksall])]-1
            # find all positive peaks above threshold
            ipos_peaksall, _ = find_peaks(dy,height=10,threshold=10)
            # find the positive peak before tstop
            # ipos_peak = ipos_peaksall[np.where(ipos_peaksall>istop)[0][0]]
            ipospeaks[isweep] = ipos_peaksall[np.argmax(dy[ipos_peaksall])]-1
            ipeaksall = np.sort(np.concatenate((ineg_peaksall,ipos_peaksall)))
            [self.clampprop[isweep]["iclamps"].append(iclamp) for iclamp in ipeaksall]
            [self.clampprop[isweep]["tclamps"].append(t[iclamp]) for iclamp in ipeaksall]

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def estimate_tau_access_res_from_epsp(self,resch,clampch):
        # estimate the tau and access resistance from charging by fitting a charging equation
        def charging_equation(t,a0,a1,tau):
            return(a0+a1*(np.exp(t/tau)))
        # ------------------
        goodsweeps = self.goodsweeps
        iclampch = [channelspec['id'] for channelspec in self.channelspecs if re.search(clampch,channelspec['name'])]
        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch,channelspec['name'])]
        t = np.arange(self.tstart,self.tstop,self.si)
        yy = self.data[:,iclampch,:].T # [isweep,ichannel,isample]
        # find the start and end of the current pulse from the clamp currrent
        inegpeaks = np.zeros(self.nsweeps,dtype = np.int)
        ipospeaks = np.zeros(self.nsweeps,dtype = np.int)
        for isweep in np.arange(0,self.nsweeps):
            y = yy[:,0,isweep]
            dy = np.diff(y,n=1,axis=0)
            # find all negative peaks above threshold
            ineg_peaksall, _ = find_peaks(-dy,height=10,threshold=10)
            # find the negative peak after tstart
            # ineg_peak = ineg_peaksall[np.where(ineg_peaksall>=istart)[0][0]]
            inegpeaks[isweep] = ineg_peaksall[np.argmax(dy[ineg_peaksall])]-1
            # find all positive peaks above threshold
            ipos_peaksall, _ = find_peaks(dy,height=10,threshold=10)
            # find the positive peak before tstop
            # ipos_peak = ipos_peaksall[np.where(ipos_peaksall>istop)[0][0]]
            ipospeaks[isweep] = ipos_peaksall[np.argmax(dy[ipos_peaksall])]-1
            ipeaksall = np.sort(np.concatenate((ineg_peaksall,ipos_peaksall)))
            
        # --------------
        avg_step_dur = (t[ipospeaks]-t[inegpeaks]).mean()
        ifitfirst = int(inegpeaks.mean())
        ifitlast = int(ipospeaks.mean())
        istep = yy[:,0,:].mean(axis=1)[ifitlast] - yy[:,0,:].mean(axis=1)[ifitfirst]
        iscale = 1e-12
        vscale = 1e-3
        print(istep)
        avg_y = self.data[goodsweeps,iresch,:].T.mean(axis=1) # avg_y from only goodsweeps
        yc = avg_y[ifitfirst:ifitlast]-avg_y[ifitfirst]
        tc = t[ifitfirst:ifitlast]-t[ifitfirst]
        # perform constrained optimization
        popt, pcov = curve_fit(charging_equation,tc, yc)
        print(popt)
        tau = abs(popt[-1])
        access_res = ((popt[0]+popt[1])*vscale)/(istep*iscale)
        print("Tau = ",tau)
        print("Access resistance = {} {}".format(access_res/1e6,"M ohms"))
        # popt, pcov = curve_fit(charging_equation,tc, yc, bounds=([-100,-100,-100], [10, 10, 10]))
        # display the results
        # fh = plt.figure(figsize=(10,8))
        # ah = fh.add_subplot(111)
        # ah.plot(tc,yc)
        # plt.plot(tc, charging_equation(tc, *popt), 'g--')
        # ah.plot(t,yy[:,0,:].mean(axis=-1))
        # ah.plot(t[ifitfirst],yy[:,0,:].mean(axis=1)[ifitfirst],color="red",marker='o',markersize=10)
        # ah.plot(t[ifitlast],yy[:,0,:].mean(axis=1)[ifitlast],color="red",marker='o',markersize=10)
        # plt.show()
        return(tau,access_res)
    # -------------------------------------------------------
    def get_sweeps(self,offsetdur,sweepdur):
        resch = self.resch
        stimch = self.stimch
        # get channel ids for response and trigger channels
        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch,channelspec['name'])]
        istimch = [channelspec['id'] for channelspec in self.channelspecs if re.search(stimch,channelspec['name'])]
        iholdsteps = self.iholdsteps
        stimprop = self.stimprop
        sweeps = self.goodsweeps
        si = self.si
        sweepdurn = int(sweepdur/si)
        offsetdurn = int(offsetdur/si)
        stims = np.array([stimprop[i]["istims"] for i in sweeps])
        indices = np.zeros(stims.shape,dtype=np.int)
        indices[:,0] = stims[:,0] - offsetdurn
        indices[:,-1] = stims[:,0] + sweepdurn
        # X = self.data[sweeps,iresch,(stimprop[0]["istims"][0]-prestimn):(stimprop[0]["istims"][-1]+prestimn)].T
        X = self.data[sweeps,iresch,:].T
        X = np.array([X[indices[i,0]:indices[i,-1],i] for i in range(0,indices.shape[0])]).T
        # X = self.data[sweeps,iresch,iholdsteps[0]["pos"][0]+offsetdurn:iholdsteps[0]["pos"][0]+offsetdurn+sweepdurn].T
        # X -= X[0,:]
        t = np.arange(0,len(X)*si,si)[0:len(X)].reshape(len(X),1)
        stims = stims - stims[:,[0]]  + np.repeat(np.array([offsetdurn,offsetdurn])[np.newaxis,:],X.shape[1],axis=0)
        tstims = t[stims,0]
        istims = stims
        return(t,X,tstims.T,istims.T)
    # ------------------------------------------------------
    def extract_response(self,tpre,tpost,min_isi=0):
        # extract signal from each sweep
        # reschannel: response channel
        # trgchannel: stimulus trigger channel
        # tpre:  time of the start of the response wrt to stimulus time
        # tpost:  time of the stop of the response wrt to stimulus time
        # min_isi: minimum inter-stimulus interval between responses
        resch = self.resch
        stimch = self.stimch
        # get channel ids for response and trigger channels
        ires = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch,channelspec['name'])]
        itrg = [channelspec['id'] for channelspec in self.channelspecs if re.search(stimch,channelspec['name'])]
        # check if isi is <= min_isi
        if ((self.stimprop[0]["nstim"] > 1) and (min_isi>0)):
            isi = [elm["isi"] for elm in  self.stimprop if not np.isnan(elm["isi"])]
            if (len(isi)>0):
                isi = np.mean(isi)
                if (isi < min_isi):
                    print("isi < min_isi")
                    return()
                # ---------
        sweeps = self.goodsweeps
        # declare array to hold all the responses
        nsamples = int((tpost-tpre)/self.si)
        res = [] 

        lowcut = 0.000001
        highcut = 3000
        fs = 1/self.si
        iholds = self.iholdsteps # gets the start and end positions of all the holding steps in the trace
        ioffset = int(0.1/self.si) # 100ms offset
        yout = []
        minsamples = 1000000
        # ---------------
        for s in range(0,len(sweeps)):
            isweep = sweeps[s]
            tstims = self.stimprop[isweep]["tstims"] # get stimulus delivery times
            istims = self.stimprop[isweep]["istims"] # get stimulus delivery indices
            ipos = iholds[isweep]['pos'][0]+ioffset # get the position of the postive rising phase of the clamp step
            tt = np.arange(0,(self.tstop-self.tstart),self.si)[np.newaxis].T[ipos:]
            yy = self.data[isweep,ires,ipos:].T # [isweep,ichannel,isample]: extract the portion of trace after the holding step
            # yy = butter_bandpass_filter(yy[:,0],lowcut,highcut,fs,order=1)[:,np.newaxis] # CAUTION! can cause artifacts in the trace
            # -------------
            # print(iholds[isweep])
            # input()
            # [isweep,ichannel,isample]: extract the portion of trace b/w current/voltage step and first stimulus
            ysub = smooth(self.data[isweep,ires,ipos:istims[0]][0],windowlen=101,window='hanning')
            tsub = np.arange(0,(len(ysub)+1)*self.si,self.si)[0:len(ysub)][np.newaxis].T
            print(tsub.shape,ysub.shape)
            # input()
            m1,c1 = np.linalg.lstsq(np.concatenate((tsub,np.ones(tsub.shape)),axis=1),ysub,rcond=None)[0]
            baseline = (tt*m1 + c1)
            yy = yy - baseline
            yy = yy-yy[istims[0]]
            for i in range(len(istims)):
                ifirst = np.where(tt>(tstims[i]+tpre))[0][0]
                ilast = np.where(tt>(tstims[i]+tpost))[0][0]
                y = yy[ifirst:ilast].reshape((ilast-ifirst)) # [isweep,ichannel,isample]
                y = smooth(y,windowlen=11,window='hanning')
                # y = y - y[0]
                t = tt[ifirst:ilast]-tt[ifirst] - tpre
                minsamples = min(minsamples,len(t))
                # print(minsamples,t.shape,y.shape)
                # print(tstims[i],t.shape,y.shape)
                yout.append(y)
                # check the trace
                # fh = plt.figure()
                # ah = fh.add_subplot(111)
                # ah.plot(t,y)
                # # ah.plot(tsub,ysub)
                # # ah.plot(tsub,tsub*m1 + c1)
                # ah.set_title("".join(("sweep: ",str(isweep),", stim: ",str(i))))
                # plt.show()
                # plt.close()
            # }
        # }
        # -------------
        yout = [elem[0:minsamples] for elem in yout] 
        yout = np.array(yout).T
        t = t[0:minsamples]
        return(t,yout)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def extract_traces(self,isweeps):
        resch = self.resch
        stimch = self.stimch
        # get channel ids for response and trigger channels
        ires = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch,channelspec['name'])]
        itrg = [channelspec['id'] for channelspec in self.channelspecs if re.search(stimch,channelspec['name'])]
        sweeps = self.goodsweeps
        iholds = self.iholdsteps # gets the start and end positions of all the holding steps in the trace
        ioffset = int(1/self.si) # 300ms offset
        idur = int(0.75/self.si)
        s1 = iholds[0]["neg"][0]+ioffset
        s2 = iholds[0]["neg"][0]+ioffset+idur
        yy = self.data[isweeps,ires,s1:].T # [isweep,ichannel,isample]
        yy.reshape((yy.shape[0],len(isweeps)))
        tt = np.arange(0,self.si*yy.shape[0],self.si)
        # fh = plt.figure()
        # ah = fh.add_subplot(111)
        # ah.plot(tt,yy)
        # plt.show()
        return(tt,yy)
            
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def extract_noise(self,group):
        resch = self.resch
        stimch = self.stimch
        # get channel ids for response and trigger channels
        ires = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch,channelspec['name'])]
        itrg = [channelspec['id'] for channelspec in self.channelspecs if re.search(stimch,channelspec['name'])]
        sweeps = self.goodsweeps
        iholds = self.iholdsteps # gets the start and end positions of all the holding steps in the trace
        ioffset = int(1/self.si) # 300ms offset
        idur = int(0.75/self.si)
        avgs = np.zeros(len(sweeps),dtype=np.float)
        noises = np.zeros(len(sweeps),dtype=np.float)
        t = np.zeros(len(sweeps),dtype=np.float)
        yyout = []
        # print(self.reader)
        # print(dir(self.reader))
        # print(self.reader._t_starts)
        # print(self.reader.segment_count(0))
        # print(self.reader.segment_t_start(1,28))
        starttime = self.reader.raw_annotations["blocks"][0]["rec_datetime"]
        # time2 = time + datetime.timedelta(seconds=60)
        # print(time.strftime("%H-%M"))
        # print(time2.strftime("%H-%M"))
        if (group == "mni"):
            sweepid = -30
        if (group == "pre_mni"):
            sweepid = 5
        # ---------------
        for s in range(0,len(sweeps)):
            t[s] = (starttime + datetime.timedelta(seconds = self.reader.segment_t_start(0,s))).timestamp()
            isweep = sweeps[s]
            yy = self.data[isweep,ires,iholds[s]["neg"][0]+ioffset:iholds[s]["neg"][0]+ioffset+idur].T # [isweep,ichannel,isample]: extract the portion of trace after the holding step
            # yy = butter_bandpass(yy[:,0],0.1,1000,1/self.si,order=1)[:,np.newaxis] # CAUTION! can cause artifacts in the trace
            tt = np.arange(0,self.si*idur,self.si)
            avgs[s] = yy.mean()
            noises[s] = yy.max()-yy.min()
            # yyout.append(yy)
            if (s == sweeps[sweepid]):
                yyout = yy
                ttout = tt
            # fh = plt.figure()
            # ah = fh.add_subplot(111)
            # ah.plot(yy)
            # plt.show()
        # yyout = np.array(yyout).reshape(len(tt),len(sweeps))
        return(t,avgs,noises,ttout,yyout)
            
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def detrend(self,resch,stimch):
        # get the stimulus times
        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch,channelspec['name'])]
        istimch = [channelspec['id'] for channelspec in self.channelspecs if re.search(stimch,channelspec['name'])]
        iholdsteps = self.iholdsteps
        stimprop = self.stimprop
        sweeps = self.goodsweeps
        si = self.si
        y = self.data[:,iresch,:]
        t = np.arange(0,y.shape[-1]*si,si).reshape(y.shape[-1],1)
        ystim = np.zeros((y.shape[-1],1),dtype=np.float)
        tstim = np.arange(0,y.shape[-1]*si,si)
        toffset = 0.15
        ioffset = int(toffset/si)
        for s in range(0,len(sweeps)):
            isweep = sweeps[s]
            istims = stimprop[isweep]["istims"]
            isi = stimprop[isweep]["isi"]
            # create stim starts and stops
            # start/end positions of each stimultions, including the dummy at the start
            ibegins = np.concatenate(([istims[0] - ioffset],istims)) 
            iends = np.concatenate((istims,[istims[-1] + ioffset]))
            # ------------------
            tsweep = t[iholdsteps[isweep]["pos"][0]+ioffset:iends[-1],:]
            ysweep = y[isweep,iresch,iholdsteps[isweep]["pos"][0]+ioffset:iends[-1]].reshape(tsweep.shape)
            m1,c1 = np.linalg.lstsq(np.concatenate((tsweep,np.ones(tsweep.shape)),axis=1),ysweep,rcond=None)[0]
            ysweep = ysweep - (tsweep*m1 + c1)
            # ysweep2 = smooth(ysweep[:,0],windowlen=51,window='hanning')[:,np.newaxis]
            fr = np.fft.fftfreq(len(ysweep),si)
            fou = np.fft.fft(ysweep[:,0])
            #make up a narrow bandpass with a Gaussian
            df=0.1
            f1,f2 = 0.00001,50        # Hz
            gpl= np.exp(- ((fr-f1)/(2*df))**2)+ np.exp(- ((fr-f2)/(2*df))**2)  # pos. frequencies
            gmn= np.exp(- ((fr+f1)/(2*df))**2)+ np.exp(- ((fr+f2)/(2*df))**2)  # neg. frequencies
            g=gpl+gmn    
            filt=fou*g  #filtered spectrum = spectrum * bandpass
            #ifft
            ysweep2=np.real(np.fft.ifft(filt))
            fh = plt.figure()
            ah1 = fh.add_subplot(311)
            ah1.plot(tsweep,ysweep)
            ah2 = fh.add_subplot(312)
            ah2.plot(np.fft.fftshift(fr) ,np.fft.fftshift(np.abs(fou)))
            ah3 = fh.add_subplot(313,sharex=ah2)
            ah3.plot(fr,np.abs(filt))
            # -------------------------
            # imf = emd.sift.sift(ysweep)
            # IP,IF,IA = emd.spectra.frequency_transform( imf, 1/si, 'nht' )
            # freq_edges,freq_bins = emd.spectra.define_hist_bins(0,10,50)
            # hht = emd.spectra.hilberthuang( IF, IA, freq_edges )
            #             plt.figure( figsize=(16,8) )
            # plt.subplot(311,frameon=False)
            # plt.plot(tsweep,ysweep,'k')
            # for j in range(imf.shape[1]):
            #     plt.plot(tsweep,imf[:,j]-j*2,'r')
            # plt.xlim(tsweep[0], tsweep[-1])
            # plt.subplot(312)
            # plt.plot(tsweep,ysweep2)
            # plt.plot(tsweep,imf[:,5:].sum(axis=1))
            # plt.grid(True)
            # plt.subplot(313)
            # # plt.pcolormesh(tsweep[:,0], freq_bins, hht, cmap='ocean_r' )
            # plt.ylabel('Frequency (Hz)')
            # plt.xlabel('Time (secs)')
            # fh = plt.figure()
            # ah1 = fh.add_subplot(211)
            # ah1.plot(tsweep,ysweep)
            # for i in range(1,len(ibegins)):
            #     ah1.plot([t[ibegins[i]],t[ibegins[i]]],[0,1],'r',linewidth=3) # stim marker
            # # }
            # # plot histogram
            # ah2 = fh.add_subplot(212,sharex=ah1)
            # ah2.plot(tsweep[:-1],np.diff(ysweep[:,0]))
            # ----------

            plt.show()
            
    def deconvolve_template(self,resch,stimch,ttemp,ytemp,direction,spine,path_fig): # correlate(a,v)
        # a method to seperate EPSCs using template matching
        # get the stimulus times
        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch,channelspec['name'])]
        istimch = [channelspec['id'] for channelspec in self.channelspecs if re.search(stimch,channelspec['name'])]
        iholdsteps = self.iholdsteps
        stimprop = self.stimprop
        sweeps = self.goodsweeps
        print(iholdsteps)
        print(stimprop)
        si = self.si
        tprestim= 0.3
        iprestim = int(tprestim/si)
        y = self.data[sweeps,iresch,(stimprop[0]["istims"][0]-iprestim):(stimprop[0]["istims"][-1]+iprestim)].T
        y = y.mean(axis=1).reshape(len(y),1)
        y -= y[0]
        t = np.arange(0,len(y)*si,si)[0:len(y)].reshape(len(y),1)
        if (direction == "positive"):
            ytemp = ytemp / max(ytemp) # normalize ytemp
        if (direction == "negative"):
            ytemp = ytemp / min(ytemp) # normalize ytemp
        ytemp = ytemp - ytemp[0]
        dytemp = np.diff(ytemp[:,0])[:,np.newaxis]
        dytemp = dytemp-dytemp[0,0]
        dttemp = ttemp[0:-1]
        dy = np.diff(y[:,0])[:,np.newaxis]
        dy = dy - dy[0,0]
        dt = t[0:-1]
        # ------------------
        fh = plt.figure()
        ah = fh.add_subplot(111)
        for s in range(0,y.shape[1]):
            isweep = sweeps[s]
            isi = stimprop[isweep]["isi"]
            istims = np.array(stimprop[isweep]["istims"])
            istims = istims-(istims[0]-iprestim)
            istims = [0, *istims]
            print(istims)
            # ------------------
            # matrices to solve AX = B
            A = np.zeros((dy.shape[0],len(istims)),dtype=np.float)
            # rightpos = 0
            for i in range(0,len(istims)): # create matrix A
                A[istims[i]:istims[i]+len(dytemp),i] = dytemp[:,0]
            # }
            B = np.zeros((dy.shape[0],1),dtype=np.float)
            B[:,0] = dy[:,0]
            X = np.linalg.lstsq(A,B,rcond=None)[0]
            F = A*X.T
            F = np.array([np.cumsum(F[:,k]) for k in range(0,F.shape[1])]).T
            print(F.shape)
            # -------------
            if (direction == "positive"):
                ipeaks = [np.argmax(F[:,k]) for k in range(F.shape[1])]
                peaks = [F[ipeaks[k],k] for k in range(F.shape[1])]
                print("peak: ",peaks)
                # }
            if (direction == "negative"):
                ipeaks = [np.argmin(F[:,k]) for k in range(F.shape[1])]
                peaks = [F[ipeaks[k],k] for k in range(F.shape[1])]
                print("peaks: ",peaks)
            # }
            # ------------------
            plotcolors=plotting_functions.colors_rgb[0:len(istims)]
            print(plotcolors)
            ah.plot(t,y,"grey",linewidth=3,label="actual") # actual trace
            ah.plot(dt,F.sum(axis=1),color='black',linewidth=3,label="reconvolved") # reconvolved trace - diff
            for k in range(F.shape[1]):
                ah.plot(dt,F[:,k],linewidth=3,color=plotcolors[k],label = "".join(("deconvolved stim:",str(k)))) # deconvolved traces
                ah.plot(dt[ipeaks[k]],peaks[k],'o',color='violet')
            # ah.vlines(t[istims],0,1,"red")
            # -------------------
            figtitle = "".join((spine,"_sweep",str(isweep),"_isi",str(int(isi*1000))+"ms"))
            ah.set_title(figtitle,fontsize=12)
            # ah.set_xlim([t[ibegins[0]-1000],t[iends[-1]]])
            # ylims = [min(X)+(min(X)*0.2),max(X)-(max(X)*10)]
            # print(ylims)
            # ah.set_ylim(ylims)
            ah.legend()
            if(len(path_fig)>10):
                fh.savefig(os.path.join(path_fig,"".join((figtitle,".png"))),format="png",dpi=300)
            # plt.show()
            # plt.close('all')
            # -------------------
        # }
        return(fh,ah,peaks)
    # }
        
    # step 2: compute normalized cross correlation
    
    # step 3: shift one wave to the maximum correlation
    def deconvolve_crop_reconvolve(self,resch,clampch,trgch,plot=False):
        # A method to extract individual voltage traces from an EPSP containing multiple short ISIs
        # This method is first described in the paper by Richardson & Silberberg, J Physio., 2008
        def integration_forward_scheme(x0,istimstart,dt,tau,d):
            v = np.zeros((len(d)+1))
            v[istimstart] = x0
            for i in np.arange(istimstart+1,len(v)):
                v[i] = v[i-1] + (dt*(d[i-1] - v[i-1])/tau)
                
            return(v)
        # ----------------
        def plot_results(t,yy,yreconv,ipeaks,tstims):
            fh = plt.figure(figsize=(8,5))
            ah = fh.add_subplot(111)
            si = t[1]-t[0]
            prestim = 0.05
            iprestim = int(prestim/si)
            poststim = 0.5
            ipoststim = int(poststim/si)
            for i in np.arange(0,yy.shape[1]):
                itfirst = int(tstims[i,0]/si) - iprestim
                itlast = int(tstims[i,-1]/si) + ipoststim
                ah.plot(t[itfirst:itlast],yy[itfirst:itlast,i]-baselines[itfirst:itlast,i],color="grey")
                ah.plot(t[itfirst:itlast],yreconv[itfirst:itlast,:,i].sum(axis=1),color="black",linewidth=3)
                for j in range(len(tstims[i])):
                    # ah.plot(t[:-1],ydeconv[:,j,:])
                    ah.plot(t[itfirst:itlast],yreconv[itfirst:itlast,j,i],color="red")
                    # display peaks
                    ah.plot(t[ipeaks[i,j]],yreconv[ipeaks[i,j],j,i],'o',markersize=5,color="blue")
                    # plot stimulus position
                    ah.plot([tstims[i][j],tstims[i][j]],[0,1],color="blue")
                # }
            # }
            return(fh,ah)
        # -------------
        if(len(self.goodsweeps) == 0): # return if no goodsweeps found
            return()

        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(resch,channelspec['name'])]
        itrgch = [channelspec['id'] for channelspec in self.channelspecs if re.search(trgch,channelspec['name'])]
        # ---------------
        tstims = np.array([self.stimprop[sweep]["tstims"] for sweep in self.goodsweeps]) # only goodsweeps
        istims = np.array([self.stimprop[sweep]["istims"] for sweep in self.goodsweeps]) # only goodsweeps
        isi = np.array([self.stimprop[sweep]["isi"] for sweep in self.goodsweeps]) # only goodsweeps
        iisi = np.array([int(self.stimprop[sweep]["isi"]/self.si) for sweep in self.goodsweeps if not np.isnan(self.stimprop[sweep]["isi"])]) # only goodsweeps
        goodsweeps = self.goodsweeps
        print("goodsweeps: ",goodsweeps)
        # load the full trace
        yy = self.data[goodsweeps,iresch,:].T 
        t = np.arange(0,(yy.shape[0]+2)*self.si,self.si)
        t = t[0:yy.shape[0]]
        t = t - t[0]
        t.resize((len(t),1))
        si = self.si;
        # tau = self.estimate_tau_access_res_from_epsp(resch,clampch)*2.5
        tau,access_res = self.estimate_tau_access_res_from_epsp(resch,clampch)
        # adjust tau to take into account of tau the synapse
        tau = tau *1.5
        tclamps = np.array([self.clampprop[goodsweep]["tclamps"] for goodsweep in goodsweeps])
        iclamps = np.array([self.clampprop[goodsweep]["iclamps"] for goodsweep in goodsweeps])
        peaks = np.zeros((len(goodsweeps),len(istims[0])))
        ipeaks = np.zeros((len(goodsweeps),len(istims[0])),dtype=np.int)
        prestimbaselinedur = 0.001
        resdur = 0.017
        ydeconv = np.zeros((len(t)-1,len(istims[0]),len(goodsweeps)))
        print(ydeconv.shape)
        yreconv = np.zeros((len(t),len(istims[0]),len(goodsweeps)))
        baselines = np.zeros((len(t),len(goodsweeps)))
        sweepcount = 0
        for iy,igoodsweep in zip(np.arange(0,yy.shape[1]),goodsweeps): # yy has only goodsweeps
            y = smooth(yy[:,iy],windowlen=31,window='hanning')
            # y = yy[:,i]
            ioffset = int(0.1/self.si)
            t1 = t[iclamps[iy][-1]+ioffset:]
            y1 = y[iclamps[iy][-1]+ioffset:]
            # print(ioffset)
            # input()            
            # fh = plt.figure()
            # ah = fh.add_subplot(111)
            # ah.plot(t1,y1)
            # plt.show()
            # m1,c1 = np.linalg.lstsq(np.concatenate((t1,np.ones((len(t1),1))),axis=1),y1,rcond=None)[0]
            # baselines[:,i] = (t*m1 + c1)[:,0]
            m2,m1,c1 = np.linalg.lstsq(np.concatenate((pow(t1,1.5),t1,np.ones((len(t1),1))),axis=1),y1,rcond=None)[0]
            baselines[:,iy] = (pow(t,1.5)*m2 + t*m1 + c1)[:,0]
            y = y - baselines[:,iy]
            dy = np.diff(y,n=1)
            d = ((dy/si)*tau) + y[0:-1]
            d = smooth(d,windowlen=7,window='hanning')
            # crop the individual responses
            for j in range(0,len(istims[iy])):
                print("iy: {},igoodsweep: {},j: {}".format(iy,igoodsweep,j))
                isstart = istims[iy][j] + int(prestimbaselinedur/si)
                isstop = istims[iy][j] + int(resdur/si)
                ydeconv[isstart:isstop,j,iy] = d[isstart:isstop]
                yreconv[:,j,iy] = integration_forward_scheme(0,isstart,si,tau,ydeconv[:,j,iy])
                ymax = max(np.abs(yreconv[:,j,iy]))
                ipeaksall,_ = find_peaks(np.abs(yreconv[:,j,iy]),prominence = ymax/2)
                ipeaks[iy,j] = ipeaksall[0]
                peaks[iy,j] = yreconv[ipeaks[iy,j],j,iy]
                # ----------------------
        fh,ah = plot_results(t,yy,yreconv,ipeaks,tstims)
        return(peaks,fh,ah)

    # ====================================================
    
    def __del__(self):
        print("Object of EphysClass has been deleted")
        pass

    
    def test(self):
        # a method to seperate EPSCs using template matching
        # get the stimulus times
        iresch = [channelspec['id'] for channelspec in self.channelspecs if re.search(self.resch,channelspec['name'])]
        istimch = [channelspec['id'] for channelspec in self.channelspecs if re.search(self.stimch,channelspec['name'])]
        iholdsteps = self.iholdsteps
        stimprop = self.stimprop
        sweeps = self.goodsweeps
        print(iholdsteps)
        print(stimprop)
        si = self.si
        tprestim= 0.3
        iprestim = int(tprestim/si)
        X = self.data[sweeps,iresch,(stimprop[0]["istims"][0]-iprestim):(stimprop[0]["istims"][-1]+iprestim)].T
        # X -= X[0,:]
        t = np.arange(0,len(X)*si,si)[0:len(X)].reshape(len(X),1)
        # ------------------
        # ica = FastICA(n_components=2)
        # S = ica.fit_transform(X)
        # X_  = ica.inverse_transform(S)
        # print(S.shape)
        # fh = plt.figure()
        # ah = fh.add_subplot(111)
        # ah.plot(t,X.mean(axis=1))
        # ah.plot(t,X_.mean(axis=1))
        # plt.show()
        # -----------------
        # U, S, V = np.linalg.svd(X,full_matrices=False)
        # print(U.shape,S.shape,V.shape)
        # X_ = np.dot(U[:,:-2] * S[:-2], V[:-2,:])
        # fh = plt.figure()
        # ah = fh.add_subplot(111)
        # ah.plot(t,X.mean(axis=1))
        # ah.plot(t,X_.mean(axis=1))
        # plt.show()
        # ------------------
        # filter noise based on FFT alone
        # XFr = np.array([np.real(fft(X[:,i])) for i in range(0,X.shape[1])]).T
        # XFc = np.array([np.imag(fft(X[:,i])) for i in range(0,X.shape[1])]).T
        # XF = XFr + 1j*XFc
        # f = fftfreq(X.shape[0],self.si)
        # print(XF.shape,f.shape)
        # def clipXF(XF,f,f1,f2):
        #     XF[(f>f2) & (f>=0)] = 0
        #     XF[(f<-f2) & (f<=0)] = 0
        #     XF[(f<f1) & (f>=0)] = 0
        #     XF[(f>-f1) & (f<=0)] = 0
        #     return(XF)
        # XF1 = clipXF(XF,f,2,500)
        # X1_ = np.array([np.real(ifft(XF1[:,i])) for i in range(0,XF1.shape[1])]).T
        # XF2 = clipXF(XF,f,1,6)
        # X2_ = np.array([np.real(ifft(XF2[:,i])) for i in range(0,XF2.shape[1])]).T
        # # print(XFr.shape)
        # # print(f,XFr)
        # fh = plt.figure()
        # ah = fh.add_subplot(111)
        # # ah.plot(f,np.abs(XF[:,0]))
        # ah.plot(t,(X-X[0,:]).mean(axis=1))
        # ah.plot(t,np.real(X1_.mean(axis=-1)))
        # ah.plot(t,np.real(X2_.mean(axis=-1)))
        # plt.show()
        # input()
        # -----------------------
        # filter noise based on FFT and ICA
        XFr = np.array([np.real(fft(X[:,i])) for i in range(0,X.shape[1])]).T
        XFi = np.array([np.imag(fft(X[:,i])) for i in range(0,X.shape[1])]).T
        XF = XFr + 1j*XFi
        ica = FastICA(n_components=4)
        # ica.fit(XFr)
        Sr = ica.fit_transform(XFr)
        Si = ica.fit_transform(XFi)
        Sr[:,3:] = 0
        Si[:,3:] = 0
        XFr_  = ica.inverse_transform(Sr)
        XFi_  = ica.inverse_transform(Si)
        XF_ = XFr_ + 1j*XFi_
        X_ = np.array([np.real(ifft(XF_[:,i])) for i in range(0,XF_.shape[1])]).T
        fh = plt.figure()
        ah = fh.add_subplot(111)
        ah.plot(t,X[:,0]-X[0,0])
        ah.plot(t,X_[:,0])
        plt.show()
        input()
        # ------------------------
        # fh = plt.figure()
        # ah = fh.add_subplot(111)
        # ph = ah.pcolormesh(np.arange(0,5),f[(f>=0) & (f<=f2)],np.abs(XF)[(f>=0) & (f<f2),:]/np.max(np.abs(XF)[(f>=0) & (f<f2),:]),vmin=0,vmax=1,cmap='hot')
        # fh.colorbar(ph)
        # plt.show()
        # input()
        # -----------------------
        # Ur, Sr, Vr = np.linalg.svd(np.real(XF),full_matrices=False)
        # Uc, Sc, Vc = np.linalg.svd(np.imag(XF),full_matrices=False)
        # print(Ur.shape,Sr.shape,Vr.shape)
        # input()
        # # XF_ = np.dot(Ur[:,0:1] * Sr[0:1], Vr[0:1]) +1j*(np.dot(Uc[:,0:1] * Sc[0:1], Vc[0:1]))
        # # XF_ = np.dot(Ur[:,1:] * Sr[1:], Vr[1:]) +1j*(np.dot(Uc[:,1:] * Sc[1:], Vc[1:]))
        # XF_ = np.dot(Ur[:,[1]] * Sr[1], Vr[[1]]) +1j*(np.dot(Uc[:,[1]] * Sc[1], Vc[[1]]))
        # print(XF_.shape)
        # X2_ = np.array([np.real(ifft(XF_[:,i])) for i in range(0,XF_.shape[1])]).T
        # fh = plt.figure()
        # ah = fh.add_subplot(111)
        # # ah.plot(f,np.abs(XF[:,0]))
        # ah.plot(t,(X-X[0,:]).mean(axis=1))
        # ah.plot(t,np.real(X1_.mean(axis=-1)))
        # ah.plot(t,np.real(X2_.mean(axis=-1)))
        # # ah.plot(t,np.real(X1_.mean(axis=-1))-np.real(X2_.mean(axis=-1)))
        # plt.show()
        # input()
