from ephys_class import EphysClass

# fname = "/Volumes/Anup_2TB/raw_data/beiquelab/zeiss880/data_anup/20190627/C3/2019_06_27_0152.abf"
# fname = "/Volumes/Anup_2TB/raw_data/beiquelab/o2p/ephys/20200213/C1/20213005.abf"
fname = "/Volumes/Anup_2TB/raw_data/beiquelab/o2p/ephys/20200305/C1/20305010.abf"
ephys1 = EphysClass(fname,loaddata=True)
ephys1.info()
# ephys1.show([],[])           # [channels],[sweeps]
ephys1.seriesres_currentclamp("ImLEFT",0.096,0.296,50e-12)
print('Series resistance = ','\t',ephys1.sres)
