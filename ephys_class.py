from neo import AxonIO
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from scipy.signal import find_peaks
import re

class EphysClass:
    
    def __init__(self,fname,loaddata=True):
        self.fname = fname
        self.loaddata = loaddata
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
        # extract channel/signal properties for each signal/channel from header
        channelspecs = self.reader.header["signal_channels"].dtype.names
        self.channelspecs = [dict.fromkeys(channelspecs) for channel in np.arange(0,self.nchannels)]
        for i in np.arange(0,self.nchannels):
            self.channelspecs[i] = self.reader.header["signal_channels"][i]
        # ----            
        # get number of samples from the first block, first segment
        self.nsamples = self.reader.get_signal_size(block_index=0,seg_index=0)
        if(self.loaddata):
            self.data=self.__loaddata()
        # extract date/time of the file
        self.datetime = self.reader.raw_annotations["blocks"][0]["rec_datetime"]
        self.date = self.datetime.date().isoformat()
        self.time = self.datetime.time().isoformat()
        # ============================================
        
    def show(self,ichannels=[],isweeps=[]):
        # display traces
        if (len(isweeps) == 0):
            isweeps = np.arange(0,self.nsweeps)
        if (len(ichannels) == 0):
            ichannels = np.arange(0,self.nchannels)
        fig = plt.figure()
        pannelcount=0
        # load data if not already loaded
        if(not self.loaddata):
            self.data = self.__loaddata()

        t = np.arange(self.tstart,self.tstop,self.si)
        ax = []
        ax.append(fig.add_subplot(len(ichannels),1,1))
        for i in ichannels:
            pannelcount = pannelcount + 1
            if(pannelcount>1):
                ax.append(fig.add_subplot(len(ichannels),1,pannelcount,sharex=ax[0]))
            # --------------
            channelname = self.channelspecs[i]["name"]    
            channelunit = self.channelspecs[i]["units"]
            channelylabel = channelname+' (' + channelunit + ')'
            ax[pannelcount-1].set_ylabel(channelylabel)
            for j in isweeps:
                ax[pannelcount-1].plot(t,self.data[j,i,:])
        plt.show()
        # ===============================================

    def __loaddata(self):
        """loads all the sweeps and channels into a numpy ndarray: timeseries * sweeps * channels """
        self.data= np.zeros((self.nsweeps,self.nchannels,self.nsamples))
            # blk = self.reader.read_block(block_index=i,lazy=False,signal_group_mode='split-all',units_group_mode='split-all')
        blk = self.reader.read_block(lazy=False)
        iseg = -1
        for seg in blk.segments:
            iseg = iseg + 1
            isig = -1
            for i, asig in enumerate(seg.analogsignals):
                isig = isig + 1
                asig = asig.magnitude
                self.data[iseg,isig,:] = asig[:,0]                
        print('Data reading completed')
        print('Data shape: ',self.data.shape)
        return(self.data)
        # ===============================================
            
    def info(self):
        print('fname:','\t',self.fname)
        print('nblocks:','\t',self.nblocks)
        print('nsweeps:','\t',self.nsweeps)
        print('nchannels','\t',self.nchannels)
        channelnames = [channelspec['name'] for channelspec in self.channelspecs]
        print('channelnames','\t',channelnames)
        print('sampling interval:','\t',self.si)
        print('t_start','\t',self.tstart)
        print('t_stop','\t',self.tstop)
        print('sample size:','\t',self.samplesize)
        print('Date:','\t',self.date)
        print('Time:','\t',self.time)
        # ===============================================

    def determine_if_voltage_or_current_clamp(self):
        # This function checks and returns the clamp type of a given output signal"
        # !!!!!!!! TO DO !!!!!
        pass

    def seriesres_currentclamp(self,channelname,tstart,tstop,istep):
        # Measure series resistance in a current-clamp sweep"
        ichannel = [channelspec['id'] for channelspec in self.channelspecs if re.search(channelname,channelspec['name'])]
        fh = plt.figure()
        ah = plt.subplot(111)
        t = np.arange(self.tstart,self.tstop,self.si)
        istart = np.where(t>=tstart)[0][0]
        istop = np.where(t>=tstop)[0][0]
        print(istart,istop)
        for isweep in np.arange(0,self.nsweeps):
            y = self.data[isweep,ichannel,:].T # [isweep,ichannel,isample]
            dy = np.diff(y,axis=0)
            dy.resize(max(dy.shape))
            print(dy.shape)
            # find all positive peaks above threshold
            ipeaksP, _ = find_peaks(dy,height=0.1,threshold=0.5)
            print(ipeaksP)
            # find all negative peaks above threshold
            ipeaksN, _ = find_peaks(-dy,height=0.1,threshold=0.5)
            print(ipeaksN)
            # find the negative peak after tstart
            ipeakN = ipeaksN[np.where(ipeaksN>istart)[0][0]]
            # find the positive peak before tstop
            ipeakP = ipeaksP[np.where(ipeaksP>istop)[0][0]]
            # find the difference between before and after the current jump
            deltajumpN = abs(y[ipeakN+1]-y[ipeakN])
            deltajumpP = abs(y[ipeakP+1]-y[ipeakP])
            # self.sres[isweep] = (np.mean([deltajumpN,deltajumpP])*1e-3)/istep # average +ve and -ve voltage jumps
            self.sres[isweep] = (deltajumpP*1e-3)/istep # take only -ve jump
            print('Series resistance: ',self.sres[isweep]/1e6,'Mohms')
            ah.plot(t,y)
            ah.plot([t[ipeakP],t[ipeakP+1]],[y[ipeakP],y[ipeakP+1]],color='k')
            ah.plot([t[ipeakN],t[ipeakN+1]],[y[ipeakN],y[ipeakN+1]],color='k')
            print(isweep,t.shape,y.shape)
        plt.show()

    def extract_response(self,reschannel,trgchannel,tres,min_isi):
        # extract signal from each sweep
        # check if isi is <= min_isi
        if (self.stimprop[0]["nstim"] > 1):
            isi = [elm["isi"] for elm in  self.stimprop if not np.isnan(elm["isi"])]
            if (len(isi)>0):
                isi = np.mean(isi)
                if (isi < min_isi):
                    return()
                # ---------
        # ----------
        # declare array to hold all the responses
        y = np.zeros((int(tres/self.si),self.nsweeps))
        # get channel ids for response and trigger channels
        ires = [channelspec['id'] for channelspec in self.channelspecs if re.search(reschannel,channelspec['name'])]
        itrg = [channelspec['id'] for channelspec in self.channelspecs if re.search(trgchannel,channelspec['name'])]
        nres = int(tres/self.si)        
        for isweep in np.arange(0,self.nsweeps):
            tstims = self.stimprop[isweep]["tstims"]
            istims = self.stimprop[isweep]["istims"]
            ipre = istims[0]
            ipost = istims[-1]+nres
            print(ipre,ipost)
            y[:,isweep] = self.data[isweep,ires,ipre:ipre+nres] # [isweep,ichannel,isample]
            # ---------------
        return(y)
        # -------------
        
    def get_signal_props(self,reschannel,trgchannel):
        # Find the peaks and other parameters of the respose (EPSP/EPSC)
        # fh = plt.figure()
        # ah1 = plt.subplot(211)
        # ah2 = plt.subplot(212,sharex=ah1)
        ires = [channelspec['id'] for channelspec in self.channelspecs if re.search(reschannel,channelspec['name'])]
        itrg = [channelspec['id'] for channelspec in self.channelspecs if re.search(trgchannel,channelspec['name'])]
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
            print("shapes",tt.shape,yr.shape,yt.shape)
            # polynomial fit on smooth data
            # model = np.polyfit(tt,smooth(yr,windowlen=int(0.1/self.si),window='hanning'),1)
            # predicted = np.polyval(model,tt)
            # yrf = smooth(yr,windowlen=int(0.01/self.si),window='hanning')
            # yrfbase = np.mean(yrf[0:nprestim])
            # for j in np.arange(0,len(istims)):
            #     iyrfpeak = np.argmax(yrf[)
            # ah1.plot(tt,yr)
            # ah1.plot(tt,yrf,'k')
            # ah1.plot(tt[iyrfpeak],yrf[iyrfpeak],'o')
            # ah2.plot(tt,yt)
            # --------------
        # plt.show()

    def get_stimprops(self,trgchannel):
        # get information about stimulation from the trigger channel
        # properties:
        # nstim: number of stimulations
        # istim: index of stimulations
        # tstims: time of each stimulation
        # isi: mean inter stimulus inverval
        itrg = [channelspec['id'] for channelspec in self.channelspecs if re.search(trgchannel,channelspec['name'])]
        t = np.arange(self.tstart,self.tstop,self.si)
        # find the trigger pulses
        triggerthres = 500
        for isweep in np.arange(0,self.nsweeps):
            yt = self.data[isweep,itrg,:] # [isweep,ichannel,isample]
            yt.resize(self.samplesize)
            istims,_ = find_peaks(yt,height=triggerthres)
            self.nstim = len(istims)
            self.stimprop[isweep]["nstim"] = self.nstim
            [self.stimprop[isweep]["istims"].append(istim) for istim in istims]
            [self.stimprop[isweep]["tstims"].append(t[istim]) for istim in istims]
            self.stimprop[isweep]["isi"] = np.mean(np.diff(self.stimprop[isweep]["tstims"]))
            print('isweep:',isweep,'nstim',self.nstim)
            print('isi',self.stimprop[isweep]["isi"])
            print('tstims',self.stimprop[isweep]["tstims"])
            # ------------------
        # plot stimulation channel
        # fh = plt.figure()
        # ah1 = plt.subplot(111)
        # ah1.plot(t,yt)
        for isweep in np.arange(0,self.nsweeps):
            nstim = self.stimprop[isweep]["nstim"]
            istims = self.stimprop[isweep]["istims"]
            tstims = self.stimprop[isweep]["tstims"]
            # ah1.plot(tstims,yt[istims],'o','k')
            # --------------------------------
        # plt.show()
        
    def find_opposing_peak_pair(t1,x1,t2,x2,isi):
        # Find the best positive_negative peak pair that has similar peak and is separated by isi
        # find positive and negative peaks with similar amplitudes
        # !!!!!! NOT IMPLEMENTED !!!!!
        # ix1 = np.argsort(x1)
        # ix2 = np.searchsorted(x1[ix1],x2,side='left')
        # imin = np.argmin(np.arange(0,len(ix2)) - ix2)
        pass
                         

    def series_res_vclamp(self,signal_name):
        # Measure series resistance in a current-clamp sweep"
        pass
        
    def __del__(self):
        print("Object has been deleted")
        pass

    
# -----------------------
def smooth(y,windowlen=3,window = 'hanning'):
    s = np.concatenate([y[windowlen-1:0:-1],y,y[-2:-windowlen-1:-1]])
    if (window == 'flat'):
        w = np.ones(windowlen,dtype='double')
    else:
        w = eval('np.'+window+'(windowlen)')
        # -----------
    yy = np.convolve(w/w.sum(),s,mode='same')[windowlen-1:-windowlen+1]
    return (yy)
