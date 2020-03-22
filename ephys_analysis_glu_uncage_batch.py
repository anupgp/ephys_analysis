from ephys_class import EphysClass

# fname = "/Volumes/Anup_2TB/raw_data/beiquelab/zeiss880/data_anup/20190627/C3/2019_06_27_0152.abf"
# fname = "/Volumes/Anup_2TB/raw_data/beiquelab/o2p/ephys/20200213/C1/20213005.abf"
fname = "/Users/macbookair/goofy/data/beiquelab/rawdata/ephys/20200305/C1/20305010.abf"
ephys1 = EphysClass(fname,loaddata=True)
ephys1.info()
# ephys1.show([0,1],[1])           # [channels],[sweeps]
ephys1.seriesres_currentclamp("ImLEFT",0.096,0.296,30e-12)
print('Series resistance = ','\t',ephys1.sres)
# ephys1.get_stimprops("IN7")
# ephys1.get_signal_props("ImLEFT","IN7")

