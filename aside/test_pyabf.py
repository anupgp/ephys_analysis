import pyabf
from matplotlib import pyplot as plt
from neo import AxonIO

fname = "/Volumes/Anup_2TB/raw_data/beiquelab/zen/data_anup/20190627/C3/2019_06_27_0152.abf"
abf=pyabf.ABF(fname)
nabf = AxonIO(filename=fname)

fig = plt.figure(figsize=(8, 5))

ax1 = fig.add_subplot(311)
ax1.grid(alpha=.2)
ax1.set_title("Digital Outputs")
ax1.set_ylabel("Digital Outputs")

# plot the digital output of the first sweep
abf.setSweep(sweepNumber=0,channel=0)

# ax1.plot(abf.sweepX, abf.sweepD(0), color='red')
# ax1.plot(abf.sweepX, abf.sweepD(1), color='green')
# ax1.plot(abf.sweepX, abf.sweepD(2), color='blue')
# ax1.plot(abf.sweepX, abf.sweepD(3), color='black')
# ax1.plot(abf.sweepX, abf.sweepD(4), color='red')
# ax1.plot(abf.sweepX, abf.sweepD(5), color='green')
# ax1.plot(abf.sweepX, abf.sweepD(6), color='blue')
# ax1.plot(abf.sweepX, abf.sweepD(7), color='black')

ax1.set_yticks([0, 1])
ax1.set_yticklabels(["OFF", "ON"])
ax1.axes.set_ylim(-.5, 1.5)

ax2 = fig.add_subplot(312, sharex=ax1)
ax2.grid(alpha=.2)
ax2.set_title("Recorded Waveform")
ax2.set_xlabel(abf.sweepLabelY)
ax2.set_ylabel(abf.sweepLabelC)

# plot the data from every sweep
for sweepNumber in abf.sweepList:
    abf.setSweep(sweepNumber,channel=0)
    ax2.plot(abf.sweepX, abf.sweepY, color='C0', alpha=.8, lw=.5)

ax3 = fig.add_subplot(313, sharex=ax1)
ax2.grid(alpha=.2)
ax2.set_title("Recorded Waveform")
ax2.set_xlabel(abf.sweepLabelY)
ax2.set_ylabel(abf.sweepLabelC)
# plot the data from every sweep
for sweepNumber in abf.sweepList:
    abf.setSweep(sweepNumber,channel=4)
    ax3.plot(abf.sweepX, abf.sweepC, color='C0', alpha=.8, lw=.5)
# zoom in on an interesting region
# ax2.axes.set_xlim(1.10, 1.25)
# ax2.axes.set_ylim(-150, 50)
plt.show()
