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
from scipy.fft import fft,fftfreq,ifft
from sklearn.decomposition import FastICA
from scipy import signal
from scipy import stats

# --------------------------
# functions

def package_data(t,yy,nsweep,isis,istims,tstims,isi):
    index1 = np.where(isis>=isi)[0][0]
    try:
        index2 = np.where(isis>isi)[0][0]
    except:
        index2= len(isis)
    isweep1 = 0
    isweep2 = 0
    if (index1>0): isweep1 = np.max(np.cumsum(nsweep[:index1]))
    if (index2>0): isweep2 = np.max(np.cumsum(nsweep[:index2]))
    data = {}
    data["t"] = t
    data["y"] =  yy[:,isweep1:isweep2]
    data["isi"] = isi
    data["istim"] = istims[:,isweep1:isweep2]
    data["tstim"] = tstims[:,isweep1:isweep2]
    return(data)
# }

def get_weights(t,y,si,isi,dur,istim,tstim):
    w = np.zeros(y.shape,dtype=int)
    idur = int(dur/si)
    for i in range(0,len(istim)):
        istart = istim[i]
        istop = istart+idur
        w[istart:istop,0] = 1
    # }
    return(w)
# }


def joint_pdf(t,y,y0,w,si,tstim,params):
    nstim = len(params)
    nsamp = len(y)
    prior_norm = {"a":{"mu":-70,"sigma":5},"tr":{"mu":10e-3,"sigma":25e-3},"td":{"mu":100e-3,"sigma":50e-3}}
    yf = create_sweep(t,si,tstim,y0,params)
    log_pdf = []
    yerr = (yf-y)**2*w
    err = np.sum(yerr)
    # err = 1/(1 + np.exp(-err))
    std = np.std(y,axis=0)
    log_lh = (-1/(2*std))*err - ((nsamp/2) * np.log(std)) - ((nsamp/2)*np.log(2*np.pi))
    for i in range(nstim):
        keys = list(prior_norm.keys())
        for j in range(len(params[i])):
            key = keys[j]
            param = params[i][key]
            mu = prior_norm[key]["mu"]
            sigma = prior_norm[key]["sigma"]
            log_tpdf = np.log(stats.norm(mu,sigma).pdf(param))
            print(i,j,key,param,mu,sigma,log_tpdf)
            log_pdf.append(log_tpdf)
        # }
    # }
    log_pdf = np.array(log_pdf).sum()
    log_jpdf = log_lh + log_pdf
    print(log_lh)
    input()
    return(log_jpdf)
# }

def bayesian_inference(data,params):
    t = data["t"]
    y = data["y"].mean(axis=1)[:,np.newaxis]
    nsweeps = data["y"].shape[1]
    nsamples = data["y"].shape[0]
    dur = 100e-3
    isi = data["isi"]
    tstim = data["tstim"][:,0]
    istim = data["istim"][:,0]
    nstim = len(tstim)
    si = t[1,0]-t[0,0]
    w = get_weights(t,y,si,isi,dur,istim,tstim)

    nsamp = len(y)
    ymin = y.min()
    ymax = y.max()
    yinc = (ymax-ymin)/30
    edges = np.arange(ymin,ymax,yinc)
    hy,_ = np.histogram(y,edges,density=True)
    ihymax = np.argmax(hy)
    y0 = edges[ihymax]
    yf = create_sweep(t,si,tstim,0,params)
    hyf,_ = np.histogram(yf,edges,density=True)
    # yn = create_sweep(t,si,tstim,0,params)
    # yn = yn + stats.norm(y0,sig2**0.5).rvs(y.shape[0]).reshape(y.shape[0],1)
    # -------------------
    nruns = 500
    old_params = params.copy()
    new_params = params.copy()
    mus = {"a": 0, "tr": 0, "td": 0}
    sigmas = {"a": 5, "tr": 10e-3, "td": 50e-3}
    for r in range(nruns):
        for i in range(nstim):
            keys = list(old_params[i].keys())
            for j in range(len(old_params[i])):
                key = keys[j]
                mu = mus[key]
                sigma = sigmas[key]
                old_jpdf = joint_pdf(t,y,y0,w,si,tstim,old_params)
                new_params[i][key] = old_params[i][key] + stats.norm(mu,sigma).rvs(1)[0]
                new_jpdf = joint_pdf(t,y,y0,w,si,tstim,new_params)
                lh_ratio = new_jpdf/old_jpdf
                print(i,j,key,old_params,new_params,lh_ratio)
            # }
        # }
    # }

        
    # # -----------------
    fh = plt.figure()
    ah = fh.add_subplot(121)
    ah.plot(edges[0:-1],hy)
    ah.plot(edges[0:-1],hyf)
    ah.plot(y0,hy[ihymax],'o',markersize=5)
    ah = fh.add_subplot(122)
    # ah.plot(t,yn,'r')
    ah.plot(t,y)
    # ah.plot(t,yf,'k')
    # ah.plot(t,yerr+y0)
    ah.plot(t,data["y"].mean(axis=1))
    # print("Number of sweeps = {}".format(nsweeps))
    plt.show()
# }


def alphafun(x,a,tr,td):
    tp = td*tr/(td-tr)*np.log(td/tr)
    a1 = 1/(-np.exp((-tp)/tr) + np.exp((-tp)/td))
    ft = a*a1*(np.exp((-t)/td) - np.exp((-t)/tr))
    return(ft)
# }

def create_sweep(t,si,s,y0,params):
    y = np.zeros(t.shape)
    for i in range(len(s)):
        a = params[i]["a"]
        tr = params[i]["tr"]
        td = params[i]["td"]
        # offset = int(2e-3/si)
        offset = 0
        shift = int(s[i]/si) + offset        
        y = y + np.roll(alphafun(t,a,tr,td),shift)
    # }
    y = y + y0
    return(y)
# }

def unequal_list_to_numpy(l,nrows,ncols,dtype):
    nrow = np.max(nrows)
    dd = np.zeros((nrow,np.sum(ncols)),dtype=dtype)
    startcol=0
    for i,col in zip(np.arange(0,len(ncols)),ncols):
        dd[:,startcol:startcol+col] = l[i]
        startcol=startcol+col
    # }
    return(dd)
# }

def plot_sweeps(t,dd,ss,nsweeps,ah):
    cmap = matplotlib.cm.get_cmap('Paired')
    colors = cmap.colors
    startcol = 0
    for i,col in zip(np.arange(0,len(nsweeps)),nsweeps):
        y = dd[:,startcol:startcol+col]
        ah.plot(t,y,color = colors[i])
        s = ss[:,startcol:startcol+col]
        for j in np.arange(0,s.shape[0]):
            p = np.repeat(s[j,:][np.newaxis,:],2,axis=0)
            q = np.concatenate((np.zeros((1,p.shape[1]))+0.95,np.ones((1,p.shape[1]))),axis=0)*y.mean()
            ah.plot(p,q,color="k")
        startcol = startcol + col
    # }
    return(ah)
# }

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
# ----------------------------------
# provide channel names
reschannel = "ImLEFT"
clampchannel = "IN1"
stimchannel = "IN7"
ephysfile_ext = "abf"
offsetdur = 0.2
sweepdur = 1.5                 # trace duration poststim
# --------------------------------
for s in range(0,len(spines)):
    spine = spines[s]
    isis_uni = np.sort(masterdf[(masterdf["clamp"] == "cc") & (masterdf["spinename"] == spines[s])]["isi"].unique())
    print(isis_uni)
    yy = []
    tstims = []
    istims = []
    nsweep = []
    nsamp = []
    nstim = []
    isis=[]
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
            t,y,tstim,istim = ephys.get_sweeps(offsetdur,sweepdur)
            yy.append(y)
            tstims.append(tstim)
            istims.append(istim)
            nsweep.append(y.shape[1])
            nsamp.append(y.shape[0])
            nstim.append(tstim.shape[0])
        # }
    # }
    # convert the list to numpy array
    nsweep = np.array(nsweep)
    nsamp = np.array(nsamp)
    nstim = np.array(nstim)
    isis = np.array(isis)
    yy = unequal_list_to_numpy(yy,nsamp,nsweep,np.float)
    tstims = unequal_list_to_numpy(tstims,nstim,nsweep,np.float)
    istims = unequal_list_to_numpy(istims,nstim,nsweep,np.int)
    isisu = np.unique(isis)
    for i in range(0,len(isisu)):
        isi = isisu[i]
        data = package_data(t,yy,nsweep,isis,istims,tstims,isi)
        # print(data["y"].shape)
        # params = [{"a":2,"tr":3e-3,"td":30e-3},{"a":2,"tr":3e-3,"td":30e-3}]
        # bayesian_inference(data,params)
    #     print(i,isi,isis_uni)
    #     input()
    # --------------------
    params = [{"a":2,"tr":3e-3,"td":30e-3},{"a":2,"tr":3e-3,"td":30e-3}]
    isweep = 0
    fh = plt.figure()
    ah = fh.add_subplot(111)
    ah = plot_sweeps(data["t"],data["y"],data["tstim"],[12],ah)
    tstim = data["tstim"][:,isweep]
    istim = data["istim"][:,isweep]
    y0 = data["y"][istim,isweep]
    yc = create_sweep(t,si,tstim,y0,params)
    ah.plot(t,yc,"k")
    plt.show()
    # ----------------------
    print(yy.shape)
    print(isis,len(isis))
    print(isis_uni)
    print(nsweep,nsweep.sum())
    input()
    # }
