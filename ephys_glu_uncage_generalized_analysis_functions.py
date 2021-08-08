import pandas as pd
import numpy as np
import matplotlib

def compute_ppf(df):
    # compute paired-pulse facilitation
    # ------------------------
    #facilitation = stim2/stim1: stim0 = noise
    # peak1 = df.xs([1],level=["istim"])[["peak"]].values[0][0]
    # peak2 = df.xs([2],level=["istim"])[["peak"]].values[0][0]
    peak1 = df[df["istim"]==1]["peak"].values[0]
    peak2 = df[df["istim"]==2]["peak"].values[0]
    ppf = peak2/peak1
    # ----------
    df["ppf"] = ppf
    return(df)
# }

colors_rgb = np.array([
    [0,0,255],
    [0,255,0],
    [255,0,0],
    [0,255,255],
    [255,0,255],
    [255,255,0],
    [0,0,128],
    [0,128,0],
    [128,0,0],
    [0,128,128],
    [128,0,128],
    [128,128,0],
    [0,0,64],
    [0,64,0],
    [64,0,0],
    [0,64,64]
])/255
colors_hex = [matplotlib.colors.to_hex(colorval) for colorval in colors_rgb]
