from ephys_class import EphysClass
import ephys_class
import ephys_functions as ephysfuns
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import re
import math
from scipy.fft import fft,fftfreq,ifft
from sklearn.decomposition import FastICA
from scipy import signal
from scipy import stats

def clipF(F,f,f1,f2):
    F[(f>f2) & (f>=0)] = 0
    F[(f<-f2) & (f<=0)] = 0
    F[(f<f1) & (f>=0)] = 0
    F[(f>-f1) & (f<=0)] = 0
    return(F)

def biexp(t,params):
    y = params["a1"]*np.exp(-t/params["tau1"]) - params["a2"]*np.exp(-t/params["tau2"])
    return(y)

def fit_biexp2template

def template_match(t,X,ts,tt,Xt):
    delay = 0.1
    si = (t[1]-t[0])
    idelay = int(delay/si)
    its = np.array([int(item/(t[1]-t[0])) for item in ts.flatten()],dtype=np.int).reshape(ts.shape)
    for i in range(0,X.shape[1]):
        A = np.zeros((X.shape[0],ts.shape[0]))
        ilags = np.zeros(ts.shape[0],dtype=np.int)
        lags = np.zeros(ts.shape[0])
        for j in np.arange(0,ts.shape[0]):
            x1 = X[its[j,i]:its[j,i] + idelay,i]
            x1 = x1-x1[0]
            corr = signal.correlate(x1,Xt,mode='same')
            ilags[j] = (len(x1)-np.argmax(corr))
            lags[j] = ilags[j]*si
            
            A[its[j,i]-ilags[j]:its[j,i]-ilags[j]+len(Xt),j] = Xt + X[its[j,i],i]
            print(ilags)
        # }
        fh = plt.figure()
        ah = fh.add_subplot(111)
        ah.plot(t,X[:,i])
        ah.plot(t,A)
        # ah.plot(t[iXs[j,i]:iXs[j,i] + idelay]-t[iXs[j,i]]+lag,x1)
            # ah.plot(tt,Xt)
        plt.show()
        input()
            
        
def create_template(t,X,Xs):
    delay = 0.05
    Xtdur = 0.2 
    si = (t[1]-t[0])
    idelay = int(delay/si)
    iXs = np.array([int(item/(t[1]-t[0])) for item in Xs.flatten()],dtype=np.int).reshape(Xs.shape)
    ipeaks = np.zeros(Xs.shape,dtype=np.int)
    Xt = np.ones((int(0.5/si),X.shape[1]))*np.nan
    if ~(Xt.shape[0]%2):
        Xt = np.zeros((int(Xtdur/si)+1,X.shape[1]))
    iXtmid = int(len(Xt)/2)
    for i in np.arange(0,X.shape[1]):
        for j in np.arange(0,Xs.shape[0]):
            # peaks,_ = signal.find_peaks(X[iXs[j,i]:iXs[j,i] + idelay,i],wlen=int(0.1/(t[1]-t[0])),prominence=0.5)
            peaks,_ = signal.find_peaks(X[iXs[j,i]:iXs[j,i] + idelay,i],wlen=int(0.1/(t[1]-t[0])),prominence=0.3)
            print(peaks)
            if (len(peaks)>0):
                ipeak = peaks[0] + iXs[j,i]
                ipeaks[j,i] = ipeak 
                it1 = np.where(t>(t[ipeak]-0.02))[0][0]
                it2 = np.where(t>(t[ipeak]+0.15))[0][0]
                itmid = int((it2-it1)/2)
                if ~(itmid%2):
                    itmid = itmid + 1
                # }
                # Xt[iXtmid-itmid:iXtmid,i] = X[it1:it1+itmid,i]
                # Xt[iXtmid:iXtmid+itmid,i] = X[it1+itmid:it1+int(2*itmid),i]
                Xt[iXtmid-itmid:iXtmid,i] = X[it1:it1+itmid,i] - X[it1,i]
                Xt[iXtmid:iXtmid+itmid,i] = X[it1+itmid:it1+int(2*itmid),i] - X[it1,i]
                # -----------
                # A = np.
                # B = np.array([X[it1,i],X[it2,i]])
                
                # slope,intercept,r_value,p_value, err = stats.linregress()
                # -----------
                # fh = plt.figure()
                # ah = fh.add_subplot(111)
                # ah.plot(X[it1:it2,i])
                # plt.show()
                
            # }
    print(Xt.shape)
    Xtt = np.arange(0,Xt.shape[0]*si,si)[:,np.newaxis]
    # fh = plt.figure()
    # ah = fh.add_subplot(111)
    # ah.plot(np.arange(0,Xt.shape[0]*si,si),Xt)
    # ah.plot(Xtt,np.nanmean(Xt,axis=1),'k',linewidth=2)
    # ah.hlines(0,0,Xt.shape[0]*si,linestyle='--')
    # ah.plot(t[ipeaks[:,i]],X[ipeaks[:,i],i],'o')
    # ah.vlines(t[iXs[:,i]],0,5,'k',linewidth=2,zorder=2)
    # plt.show()
    # input()
    return(Xtt,np.nanmean(Xt,axis=1))
    # }
    
# paths to ephys data
path_ephysdata="/home/anup/gdrive-beiquelabdata/Ephys Data/Olympus 2P/Anup/" # common to all ephys data
# -------------------------
# Ca1 single spine
# path to ephys data master excelfile
path_masterdf="/home/anup/goofy/data/beiquelab/hc_ca1_glu_uncage_spine_single_epsp/" # local to the ephys project: Ca1 single spine
file_masterdf = "hc_ca1_glu_uncage_spine_single_mastersheet.xlsx" # local to the ephys project
dfname = "hc_ca1_glu_uncage_spine_single_masterdata.csv"           # local to the ephys project
# ------------------------
# Ca1 cluster spine
# path_masterdf="/home/anup/goofy/data/beiquelab/hc_ca1_glu_uncage_spine_cluster_epsp/" # local to the ephys project:  
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
isis = masterdf["isi"].unique().astype(int)
# spines = masterdf["spinename"].unique()
print(masterdf.head())
print(isis)
roidf = pd.DataFrame()
# ----------------------------------
# provide channel names
reschannel = "ImLEFT"
clampchannel = "IN1"
stimchannel = "IN7"
ephysfile_ext = "abf"
offsetdur = 0.2
sweepdur = 1.5                 # trace duration poststim
# --------------------------------
for i in range(6,len(isis)):
    print(isis[i])
    X = np.zeros((35000,150),dtype=np.float)
    Xs = np.zeros((20,200),dtype=np.float)
    nsweeps = 0
    spines = masterdf[masterdf["isi"] == isis[i]]["spinename"].to_list()
    for j in range(0,len(spines)):
        print(spines[j])
        ephysfiles = masterdf[(masterdf["isi"] == isis[i]) & (masterdf["clamp"] == "cc") & (masterdf["spinename"] == spines[j])]["ephysfile"].to_list()
        for k in range(0,len(ephysfiles)):
            neuronid = list(masterdf.loc[masterdf["ephysfile"] == ephysfiles[k],"neuronid"].astype(int).astype(str))[0]
            expdate = list(masterdf.loc[masterdf["ephysfile"] == ephysfiles[k],"expdate"].astype(int).astype(str))[0]
            fnamefull = os.path.join(path_ephysdata,expdate,"".join(("C",neuronid)),ephysfiles[k])
            print(fnamefull)
            badsweeps = list(masterdf.loc[masterdf["ephysfile"] == ephysfiles[k],"badsweeps"].astype(str))[0].split(',')
            badsweeps = [int(item)-1 for item in badsweeps if item != 'nan'] # trace 1 has index 0
            print('badsweeps: {}'.format(badsweeps))
            ephys = EphysClass(fnamefull,loaddata=True,badsweeps=badsweeps,clampch=clampchannel,resch=reschannel,stimch=stimchannel)
            ephys.extract_indices_holdingsteps(1,1)
            ephys.extract_stim_props()
            t,y,s = ephys.get_sweeps(offsetdur,sweepdur)
            print(s.shape)
            Xs[0:s.shape[1],nsweeps:nsweeps+s.shape[0]] = s.T
            X[0:y.shape[0],nsweeps:nsweeps+y.shape[1]] = y
            nsweeps += y.shape[1]
            # print(ephys.data.shape)
        # }
    X = X[:,0:np.where(X.sum(axis=0)==0)[0][0]] # remove extra columns
    X = X[0:np.where(X.sum(axis=1)==0)[0][0],:] # remove extra rows
    Xs = Xs[:,0:np.where(Xs.sum(axis=0)==0)[0][0]] # remove extra columns
    Xs = Xs[0:np.where(Xs.sum(axis=1)==0)[0][0],:] # remove extra rows
    D = np.diff(X.copy(),axis=0)
    print(X.shape)
    print(Xs.shape)
    # -------------------

    Xtt,Xt = create_template(t[:-1],X,Xs)
    template_match(t,X-X[0,:],Xs,Xtt,Xt)
    # Xtt,Xt = create_template(t[:-1],np.cumsum(X_,axis=0),Xs)
    # template_match(t[:-1],np.cumsum(X_,axis=0),Xs,Xtt,Xt)
    # fh = plt.figure()
    # ah = fh.add_subplot(111)
    # ah.plot(Xtt,Xt)
    # plt.show()
    # create_template(t,X-X[0,:],s)
    input()



    # Fr = np.array([np.real(fft(X[:,i])) for i in range(0,X.shape[1])]).T
    # Fi = np.array([np.imag(fft(X[:,i])) for i in range(0,X.shape[1])]).T
    # F = Fr + 1j*Fi
    # f = fftfreq(F.shape[0],ephys.si)
    # f1 = 1.5
    # f2 = 300
    # F = clipF(F,f,f1,f2)
    # Pw = np.abs(F)/np.max(np.abs(F))
    # X_ = np.array([np.real(ifft(F[:,i])) for i in range(0,F.shape[1])]).T
    # -------------------
    # Ur, Sr, Vr = np.linalg.svd(np.real(F),full_matrices=False)
    # Uc, Sc, Vc = np.linalg.svd(np.imag(F),full_matrices=False)
    # ik = [0,1,2,3,4]
    # Ur = Ur[:,ik]
    # Sr = Sr[ik]
    # Vr = Vr[:,ik][ik,:]
    # Uc = Uc[:,ik]
    # Sc = Sc[ik]
    # Vc = Vc[:,ik][ik,:]
    # XF_ = np.dot(Ur * Sr, Vr) +1j*(np.dot(Uc * Sc, Vc))
    # XF_ = np.dot(Ur * Sr, Vr))
    # X_ = np.array([np.real(ifft(XF_[:,i])) for i in range(0,XF_.shape[1])]).T
    # Ur, Sr, Vr = np.linalg.svd(X,full_matrices=False)
    # X_ = np.dot(Ur * Sr, Vr)
    # -------------------
    # ica = FastICA(n_components=10)
    # Sr = ica.fit_transform(XFr)
    # Si = ica.fit_transform(XFi)
    # i1 = 0
    # i2 = 40
    # Sr[:,i1:i2] = 0
    # Si[:,i1:i2] = 0
    # XFr_  = ica.inverse_transform(Sr)
    # XFi_  = ica.inverse_transform(Si)
    # XF_ = XFr_ + 1j*XFi_
    # X_ = np.array([np.real(ifft(XF_[:,i])) for i in range(0,XF_.shape[1])]).T
    # ---------------------
    # fh = plt.figure()
    # ah = fh.add_subplot(111)
    # ph = ah.pcolormesh(np.arange(0,XF.shape[1]+1),f[(f>=f1) & (f<=f2)],Pw[(f>=f1) & (f<f2),:],vmin=0,vmax=1,cmap='hot')
    # fh.colorbar(ph)
    # ah.plot(t,X[:,0])
    # ah.plot(t,X_[:,0])
    # ah.plot(Sr,'o')
    # ah.plot(Sc,'o')
    # fh2 = plt.figure()
    # ah2 = fh2.add_subplot(111)
    # ah2.plot(t,X-X[0,:],'k',zorder=1)
    # ah2.vlines(s[0,:],0,5,'r',linewidth=2,zorder=2)
    # fh3 = plt.figure()
    # ah3 = fh3.add_subplot(111)
    # ah3.plot(t[0:-1],np.cumsum(X_,axis=0),'r',zorder=1)
    # ah3.vlines(s[0,:],0,5,'k',linewidth=2,zorder=2)
    # fh = plt.figure()
    # ah = fh.add_subplot(111)
    # ah.plot(t,X)
    # plt.show()
