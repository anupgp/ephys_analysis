import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm


np.random.seed(1234)

def generate_normal_dist(means,sigmas,labels,nsamples):
    #
    nclusters = len(means)
    # generate data clusters
    X = np.array([np.random.normal(mean,sigma,nsamples) for mean,sigma in zip(means,sigmas)])
    labels =  np.array([np.zeros(nsamples)+label  for label in labels],dtype=np.int)
    X.resize(nclusters*nsamples)
    labels.resize(nclusters*nsamples)
    # create a pandas dataframe
    df = pd.DataFrame(columns = ["val","label"],data=np.concatenate((X[np.newaxis].T,labels[np.newaxis].T),axis=1))
    # convert label to int
    df["label"] = df["label"].astype(int)
    X.resize(nsamples,nclusters)
    return(X,df)

def EM_algorith(X,nclusters,iterations):
    # nclusters = len(means)
    # intialize random means, sigmas and weights
    means = np.random.rand(nclusters)
    sigmas = np.random.rand(nclusters)
    weights = np.random.rand(nclusters)
    ric = np.zeros(X.shape)
    vars = np.zeros(means.shape)
    for i in np.arange(0,iterations):
        gauss_dists = [norm(means[c],sigmas[c])  for c in np.arange(0,nclusters)]
        # expectation step
        # probability that sample i belongs to cluster c
        for c in np.arange(0,nclusters):
            ric[:,c] = gauss_dists[c].pdf(X[:,c])
        # make probabilities sum to one across cluster index
        ric = ric/ric.sum(axis=1)[np.newaxis].T
        # weight probabilities
        ric =  ric*weights
        # normalize sum
        ric = ric/ric.sum(axis=1)[np.newaxis].T
        # maximization step
        m_c = ric.sum(axis=0)
        weights = m_c/nsamples
        # calculate weighted mean
        for c in np.arange(0,nclusters):
            means[c] = np.sum(ric[:,c]*X[:,c])/m_c[c]
        for c in np.arange(0,nclusters):
            vars[c] = (ric[:,c]*(X[:,c]-means[c]).dot(X[:,c]-means[c])).sum(axis=0)/m_c[c]
            # --------
        sigmas = np.sqrt(vars)
    return(means,sigmas,weights)
        # -----------

# ---------------
# data = np.concatenate((np.random.normal(0,1,150),np.random.normal(3,1,150)),axis=0)
# data = np.concatenate((np.random.normal(0,1,150),np.random.gamma(2,4,150)),axis=0)
nsamples = 150
nclusters = 3
omeans = [2,5,10]
osigmas = [1,3,4]
labels = [0,1,2]
X,df = generate_normal_dist(omeans,osigmas,labels,nsamples)
print(X.shape)
iterations = 100
print("original parameters")
print("Starting means: {}".format(omeans))
print("Staring sigmas: {}".format(osigmas))
# EM algorithm
means,sigmas,weights = EM_algorith(X,3,iterations)
print("model estimates")
print(means)
print(sigmas)
print(weights)
# plot original data to visualize
for l in np.arange(0,len(labels)):
    if(l==0):
        ah = df.loc[df["label"] == labels[l],"val"].plot.hist(density=True)
    else:
        df.loc[df["label"] == labels[l],"val"].plot.hist(alpha=0.5,density=True,ax=ah)
# plot fits
x = np.arange(0,np.max(means)*4,0.01)
for i in np.arange(0,means.shape[0]):
    ah.plot(x,norm(means[i],sigmas[i]).pdf(x),'-',linewidth=2)
plt.show()

    


