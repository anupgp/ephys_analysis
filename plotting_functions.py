import matplotlib
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import transforms
import numpy as np

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
    [128,128,0]
])/255
colors_hex = [matplotlib.colors.to_hex(colorval) for colorval in colors_rgb]
              
def format_plot(fh,ah,xlab="",ylab="",title=""):
    font_path = '/home/anup/.matplotlib/fonts/arial.ttf'
    fontprop = font_manager.FontProperties(fname=font_path,size=18)
    ah.spines["right"].set_visible(False)
    ah.spines["top"].set_visible(False)
    ah.spines["bottom"].set_linewidth(1)
    ah.spines["left"].set_linewidth(1)
    ah.set_title(title,fontproperties=fontprop)
    ah.set_xlabel(xlab,fontproperties=fontprop)
    ah.set_ylabel(ylab,fontproperties=fontprop)
    ah.tick_params(axis='both',length=6,direction='out',width=1,which='major')
    ah.tick_params(axis='both',length=3,direction='out',width=1,which='minor')
    ah.tick_params(axis='both', which='major', labelsize=16)
    ah.tick_params(axis='both', which='minor', labelsize=12)
    box = ah.get_position()
    ah.set_position([box.x0+0.03, box.y0+0.03, box.width * 0.9, box.height*0.9])
    return(fh,ah)

