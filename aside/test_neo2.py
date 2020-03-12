from neo import AxonIO
import numpy as np
from matplotlib import pyplot as plt
import datetime

# fname = "/Volumes/Anup_2TB/raw_data/beiquelab/zen/data_anup/20190627/C3/2019_06_27_0152.abf"
fname = "/Volumes/Anup_2TB/raw_data/beiquelab/o2p/ephys/20200213/C1/20213008.abf"  
reader = AxonIO(filename=fname)
print(dir(reader))
# print('block count: ',reader.block_count())
# print('segment count: ',reader.segment_count(0))
# print('signal_channels_count: ',reader.signal_channels_count())
# print('signal sampling rate: ',reader.get_signal_sampling_rate())
# print('segment t_start: ', reader.segment_t_start(block_index=0,seg_index=0))
# print('segment t_start: ',reader.segment_t_stop(block_index=0,seg_index=0))
# print('header: ',reader.header) - implemented
# print('analog_signal_size: ',reader.get_signal_size(block_index=0,seg_index=0))
print(type(reader.get_analogsignal_chunk()))
print('filename',reader.filename)
# print('print_annotations',reader.print_annotations())
print('raw_annotations',reader.raw_annotations)
rawannot = reader.raw_annotations
# print(rawannot.keys())
# print(rawannot["blocks"])
# print(rawannot["blocks"][0].keys())
# doe = rawannot['blocks'][0]["rec_datetime"]
# print(doe.time())
# print(rawannot['blocks'][0]['segments'][0])
# print("protocol",reader.read_protocol())
# protocol_seg=reader.read_protocol()
# print(dir(protocol_seg[0]))
# print(protocol_seg[0].merge_annotations)
# print("raw_protocol",reader.read_raw_protocol())
# help(reader.read_channelindex)


# print('protocol: ',reader.read_protocol())
# protocol_segs = reader.read_protocol()
# print(dir(protocol_segs[0]))
# print('protocol_segs[0].size',protocol_segs[0].size)
# print(protocol_segs[0].analogsignals[0])

# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212,sharex=ax1)
# ax1.grid(alpha=.2)
# blks = reader.read_block(block_index=0,lazy=False,signal_group_mode='split-all',units_group_mode='split-all')

# blks = reader.read_block(lazy=False,signal_group_mode='split-all',units_group_mode='split-all')
# print(dir(reader))
# print('segment_count: ',reader.segment_count(0))
# print('signal_channels_count: ',reader.signal_channels_count())
# print(blks)
# print(dir(blks))
# print('num segments: ',blks.size['segments'])
# segs = blks.segments
# print('nsegs: ',reader.read_block().size['segments'])
# help(reader.read_block())
# print(segs)
# si = segs[0].analogsignals[0].sampling_rate
# units = segs[0].analogsignals[0].units
# tstart = segs[0].analogsignals[0].t_start
# tstop = segs[0].analogsignals[0].t_stop
# times = segs[0].analogsignals[0].times
# print(dir(segs[0]))
# print('nsegs: ',segs[0].size)
# print('Sampling rate: ',si)
# print('Units: ',units)
# print('T start: ',tstart)
# print('T stop: ',tstop)
# print('Times: ',times)


# sig0 = np.asmatrix(segs[1].analogsignals[0])
# sig1 = np.asmatrix(segs[1].analogsignals[1])
# sig2 = np.asmatrix(segs[1].analogsignals[3])
# sig3 = np.asmatrix(segs[1].analogsignals[4])

# ax1.plot(protocol_segs[2].analogsignals[1])
# ax1.plot(times,sig0)
# ax2.plot(times,sig1)
# ax1.plot(times,sig2*400)
# ax1.plot(times,sig3*400)
# plt.show()



