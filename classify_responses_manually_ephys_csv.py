import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import os
import re
import sys
from scipy.signal import find_peaks

class traceClassifier:
    # classify each trace(column) in df (df) into a label (given in labels)  
    def  __init__(self,_df,_labels,_fh):
        self.labels = _labels
        self.endoftrace = False
        self.key = None
        self.fh = _fh
        self.df = _df
        self.traces = [acolumn for acolumn in self.df.columns if re.search("[0-9]{8,8}.*sweep[0-9]+",acolumn)]
        self.ntraces = len(self.traces)
        self.initialize()
        self.connect()

    def get_ylabels(self):
        return(self.tracelabels)

    def restart(self):
        print("*********  restarting classification ***********")
        [self.ah[i].clear() for i in np.arange(0,len(self.ah))]
        self.initialize()

        
    def get_tracenames(self):
        return(self.traces)
        
    def check_end_of_trace(self):
        if(self.itrace >=  self.ntraces):
            self.endoftrace = True
            print("***** Reached end of trace ********")
        
    def  __call__(self,event):
        if(event.name == "key_press_event"):
            self.key = event.key
            print(self.key)
            if (self.key == 'x'):
                self.restart()
                print('***** restarting form 0 *****')
                
            self.check_end_of_trace()
            for i in np.arange(0,len(self.labels)):
                if ((self.key == self.labels[i]["key"]) and (not self.endoftrace)):
                    self.tracelabels[self.itrace] = self.key 
                    ilabel = self.labels[i]["id"]
                    # add data into the class df
                    self.data[ilabel][self.traces[self.itrace]] = self.df[self.traces[self.itrace]]
                    self.update_figure(ilabel)

    def initialize(self):
        # create default labels for all traces in the object
        self.tracelabels = ['b' for i in np.arange(0,self.ntraces)]
        self.itrace = 0
        self.t = self.df['t']
        self.y = self.df[self.traces[self.itrace]]
        # create empty data frames to hold the data for each label
        self.data = []
        [self.data.append(pd.DataFrame({})) for i in np.arange(0,len(self.labels))]
        # generate a figure window with axes equals to the number of labels+1
        # self.fh = plt.figure(figsize=(5,5))
        # create axis for each label in labels
        self.ah = []
        [self.ah.append(self.fh.add_subplot(2,2,i)) for i in np.arange(1,5)]
        self.ph = []
        # a dummy average plot
        [self.ph.append(self.ah[i].plot(0,0,color='k')[0]) for i in np.arange(0,4)]
        # axis limits
        self.axislims = []
        [self.axislims.append(dict({"xlims":[0,0],"ylims":[0,0]})) for i in np.arange(0,len(self.ah))]
        plt.subplots_adjust(hspace=0.4)
        # plot the first trace in current trace panel
        self.ph[0].set_xdata(self.t)
        self.ph[0].set_ydata(self.df[self.traces[self.itrace]])
        self.ph[0].set_color("red")
        # set titles of plots
        titles = ["Current trace"]
        [titles.append(label["attr"]) for label in self.labels]
        [self.ah[iaxis].set_title(titles[iaxis],fontweight='bold') for iaxis in np.arange(0,len(self.ah))]
        # update the current trace axis limits
        self.update_axislimits([0])

    def update_axislimits(self,iaxis):
        # find positive and negative peak and set the real peak to be the one with high abs value
        istim = np.where(self.t>0)[0][0]
        ibuffpoints = 10+istim
        ipeak,_ = find_peaks(abs(self.y[ibuffpoints:]),prominence=(0.1,None),width=(10,None),height=(0.01,None))
        if(len(ipeak)>0):
            ipeak = ipeak[0] + ibuffpoints
            ypeakmax = self.y[ipeak]
            ypeakmin = self.y[ipeak]*0.25
        else:
            ypeakmax = max(abs(self.y[ibuffpoints:]))
            ypeakmin = min(abs(self.y[ibuffpoints:]))
            
        for j in iaxis:
            if self.axislims[j]["xlims"][0]>min(self.t): self.axislims[j]["xlims"][0] = min(self.t)
            if self.axislims[j]["xlims"][1]<max(self.t):self.axislims[j]["xlims"][1] = max(self.t)
            if (ypeakmax<0):
                if self.axislims[j]["ylims"][0]>ypeakmax:self.axislims[j]["ylims"][0] = ypeakmax
                if self.axislims[j]["ylims"][1]<-ypeakmin: self.axislims[j]["ylims"][1] = -ypeakmin
            else:
                if self.axislims[j]["ylims"][0]>-ypeakmin:self.axislims[j]["ylims"][0] = -ypeakmin
                if self.axislims[j]["ylims"][1]<ypeakmax: self.axislims[j]["ylims"][1] = ypeakmax
            # set limits
            self.ah[j].set_xlim(self.axislims[j]["xlims"])
            self.ah[j].set_ylim(self.axislims[j]["ylims"])
        
    def refresh_plots(self):
        self.fh.canvas.draw()
        self.fh.canvas.flush_events()

    def connect(self):
        self.cidkeypress = self.fh.canvas.mpl_connect('key_press_event',self)
        print("connected!")
                
    def update_figure(self,ilabel):
        columns = self.data[ilabel].columns
        # add new trace into the class plot 
        for i in np.arange(0,len(columns)):
            self.ah[ilabel+1].plot(self.t,self.data[ilabel][columns[i]],color='grey',zorder=0)
        # update the average plot
        self.ph[ilabel+1].set_xdata(self.t)
        yavg = self.data[ilabel].mean(axis=1)
        self.ph[ilabel+1].set_ydata(yavg)
        # update the current figure if not end of trace
        self.itrace = self.itrace+1
        self.y = self.df[self.traces[self.itrace]]
        self.check_end_of_trace()
        if (not self.endoftrace):
            self.ph[0].set_xdata(self.t)
            self.ph[0].set_ydata(self.df[self.traces[self.itrace]])
        # update the current trace axis and selected class panel axis limits
        self.update_axislimits([0,ilabel+1])
        
        
    def disconnect(self):
        self.fh.canvas.mpl_disconnect(self.cidkeypress)

    def __del__(self):
        print('keypress recorder object delected!')


if(__name__ == "__main__"):
    
    mainpath = '/Volumes/GoogleDrive/Shared drives/Beique Lab/CURRENT LAB MEMBERS/Anup Pillai/ephys_mininum_stim/'
    csvfiles = [afile for afile in os.listdir(mainpath) if re.search("[0-9]{8,8}_.*exp[\d]{1,2}.csv",afile)]
    print(str(len(csvfiles))+" csv files found: ",csvfiles)
    # create a figure window with three panels to classify each trace into success, failure or bad
    plt.ion()
    keys = ['1','0','b']
    ids = [0,1,2]
    attrs = ["Success","Failure","Bad"]
    nlabels = 3
    labels = [dict({"key":key,"id":id,"attr":attr}) for (key,id,attr) in zip(keys,ids,attrs)]

    for csvfile in csvfiles:
        df = pd.read_csv(os.path.join(mainpath,csvfile))
        columns = list(df.columns)
        fh = plt.figure()
        fh.suptitle(os.path.splitext(csvfile)[0])
        classifier = traceClassifier(df,labels,fh)
        # plt.show()
        input('key')
        tracenames = classifier.get_tracenames()
        ylabels = classifier.get_ylabels()
        print(ylabels,len(ylabels))
        # create new tracenames
        newtracenames = []
        [newtracenames.append(trace+'_'+ylabel) for trace,ylabel in zip(tracenames,ylabels)]
        print(newtracenames,len(newtracenames))
        # find induces of tracenames in columnnames
        traceindices = []
        [traceindices.append(i) for acolumn,i in zip(columns,np.arange(0,len(columns))) if re.search("[0-9]{8,8}.*sweep[0-9]+",acolumn)]
        print(traceindices,len(traceindices))
        # add trace labels to the column name
        for i,j in zip(traceindices,np.arange(0,len(newtracenames))):
            columns[i] = newtracenames[j]
        # replace old tracename with newones
        df.columns = columns
        print("New: ",df.columns)
        # save the dataframe with the new column names
        df.to_csv(os.path.join(mainpath,os.path.splitext(csvfile)[0]+"_withlabel"+".csv"))
        del classifier
        plt.close(fh)
