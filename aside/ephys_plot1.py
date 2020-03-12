from neo import AxonIO
from scipy import signal
import numpy as np
import quantities as pq
from matplotlib import pyplot as plt
#matplotlib.use('TKAgg')

fname="/Users/macbookair/Downloads/20180913/18913005.abf"
reader = AxonIO(filename=fname)
#fig = plt.figure()
#ax = fig.add_axes([0.15, 0.1, 0.7, 0.7]) #l,b,w,h
fig, ax = plt.subplots(figsize=(10, 5))
bl = reader.read() # reads the entire blocks
for i in range(0,10):
    #print(bl.readable_objects)
    seg = bl[0].segments[i]
    #seg = reader.read_segment()
    data,units = enumerate(seg.analogsignals);
    si = seg.analogsignals[0].sampling_rate
    print(seg.analogsignals[0].units)
    print(seg.analogsignals[0].sampling_rate)
    print(seg.analogsignals[0].t_start)
    print(seg.analogsignals[0].t_stop)
    print(seg.analogsignals[0].times)

    if (i == 0):
       t = np.arange(seg.analogsignals[0].t_start,seg.analogsignals[0].t_stop,1/seg.analogsignals[0].sampling_rate);
       t = np.asmatrix(t)
       t = t.reshape(np.size(t),1)
    y = seg.analogsignals[0];

    #plt.plot(t,y)
    ax.plot(t,y)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (mV)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ofname='/Users/macbookair/Downloads/20180913/20180913c1iv.png'
#plt.show()
plt.savefig(ofname,dpi=300)
#input("Hit Enter To Close")
plt.close()
    
