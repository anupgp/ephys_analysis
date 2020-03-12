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
            self.sres[isweep] = (np.mean([deltajumpN,deltajumpP])*1e-3)/istep
            print('Series resistance: ',self.sres[isweep]/1e6,'Mohms')
            ah.plot(t,y)
            ah.plot([t[ipeakP],t[ipeakP+1]],[y[ipeakP],y[ipeakP+1]],color='k')
            ah.plot([t[ipeakN],t[ipeakN+1]],[y[ipeakN],y[ipeakN+1]],color='k')
            print(isweep,t.shape,y.shape)
        plt.show()

    def find_epsp_peak(self,sigchannel,trgchannel):
        
        pass
        
    def find_opposing_peak_pair(t1,x1,t2,x2,isi):
        # Find the best positive_negative peak pair that has similar peak and is separated by isi
        # find positive and negative peaks with similar amplitudes
        # !!!!!! NOT IMPLEMENTED !!!!!
        ix1 = np.argsort(x1)
        ix2 = np.searchsorted(x1[ix1],x2,side='left')
        imin = np.argmin(np.arange(0,len(ix2)) - ix2)
                         

    def series_res_vclamp(self,signal_name):
        # Measure series resistance in a current-clamp sweep"
        pass
        
    def __del__(self):
        print("Object has been deleted")
        pass

