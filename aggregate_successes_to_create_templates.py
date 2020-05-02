import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import re


# find the list of csvfiles that has "withlabel" identifier at the end
mainpath = '/Volumes/GoogleDrive/Shared drives/Beique Lab/CURRENT LAB MEMBERS/Anup Pillai/ephys_mininum_stim/'
csvfiles = [item for item in os.listdir(mainpath) if re.search("[0-9]{8,8}.*_withlabel.csv",item)]
# number of templates to create
categories = ["vc-70","vc40"]
labelcodes = ["1"]
labelnames = ["successes"]
# empty container to store traces for each category
data={}
selection = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],dtype = np.uint)
csvfiles2 = []
[csvfiles2.append(csvfiles[i]) for i in np.arange(0,len(selection))  if selection[i] == 1]
[data.update(dict({category+labelname:[]})) for labelname in labelnames for category in categories]
print(csvfiles,len(csvfiles))
print(csvfiles2,len(csvfiles2))
for csvfile in csvfiles2:
    df = pd.read_csv(os.path.join(mainpath,csvfile))
    for category in categories:
        for l in np.arange(0,len(labelnames)):
            pattern1 =  r"[0-9]{8,8}.*"
            pattern2 = category
            pattern3 = r".*_"
            pattern4 = labelcodes[l]
            pattern = "".join([pattern1, pattern2,pattern3,pattern4])
            pattern = re.compile(pattern)
            successes = [column for column in df.columns if re.search(pattern,column)]
            if (len(successes)>0):
                key = category+labelnames[l]
                data[key].append(df[successes].to_numpy())
                # fh = plt.figure()
                # ah = fh.add_subplot(111)
                # ah.plot(df["t"],df[successes])
                # plt.show()
                # selection.append(input("enter selection: "))

for category in categories:
    for l in np.arange(0,len(labelnames)):
        key = category+labelnames[l]
        print(len(data[key]))
        temp = np.concatenate(data[key],axis=1)
        print(temp.shape)
        temp = temp - temp[0,:]  
        data[key] = temp
        templatename = key+"_avg_template.csv"
        # save template
        pd.DataFrame({"t":df["t"],"y":data[key].mean(axis=1)}).to_csv(os.path.join(mainpath+templatename),header=True,index=False)
        # fh = plt.figure()
        # ah = fh.add_subplot(111)
        # ah.plot(df["t"],data[key])
        # ah.plot(df["t"],data[key].mean(axis=1),'k',linewidth=2)
        # plt.show()  
        # input('key')

