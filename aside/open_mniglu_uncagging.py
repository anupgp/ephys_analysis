from neo import AxonIO as axonOA
import numpy as np
from matplotlib import pyplot as plt
import os
import common_functions


class axonclass:
    def __init__(self,filepath):
        if(check_path(filepath)):
            self.filepath = filepath
        
        
    def print_attributes(self):
        print(self.filepath)
        
def read_axon_file(filename):
    # reads an ABF datafile and returns the given channel of a trial
    exit_if_path_not_present(filename)    # checks if file is present    
    reader = axon(filename=filename)
    block = reader.read()       # returns a list of block objects
    print(reader.block_count()) # returns number of blocks in the file
    print('Event channels count:',reader.event_channels_count())
    print(reader.header['signal_channels'])
    print(reader.channel_name_to_index(['ImRightP','VmRightS','trigger1','trigger2']))
    print(reader.channel_id_to_index([0,1,2,3]))
    print(reader.header['unit_channels'])
    print(reader.header['nb_block'])
    print(reader.header['nb_segment'])
    print(reader.event_count(block_index=0,seg_index=0,event_channel_index=0))
    print(reader.segment_count(0)) # returns number of segments(trials) in the given block index
    print(reader.signal_channels_count()) # returns number of signal channels
    print(reader.get_signal_sampling_rate(0)) # returns the sampling rate of the given channel index
    print(reader.get_signal_size(block_index=0,seg_index=0,channel_indexes=0)) # returns the number of samples
    print(reader.get_signal_t_start(block_index=0,seg_index=0,channel_indexes=0)) # returns start_time
    # print(reader.channel_name_to_index('ImRightP'))
    segs = block[0].segments[0:5] # each segment is a trial of recording with all the channels
    print('-------------------')
    raw_protocol = reader.read_raw_protocol()
    # print(raw_protocol)
    print(reader._axon_info.keys())
    print(reader._axon_info['protocol'].keys())
    print(reader._axon_info['sections'].keys())
    print(reader._axon_info['EpochInfo'])
    print(reader._axon_info['listDACInfo'])
    print(reader._axon_info['protocol']['nActiveDACChannel']) # information of the active DAC channel
    print('**************')
    print(reader._axon_info['dictEpochInfoPerDAC'][1][0]) # [DAC channel][Epoch number]
    print(reader._axon_info['dictEpochInfoPerDAC'][1][1]) # [DAC channel][Epoch number]
    print(reader._axon_info['dictEpochInfoPerDAC'][1][2]) # [DAC channel][Epoch number]
    print(reader._axon_info['dictEpochInfoPerDAC'][1][3]) # [DAC channel][Epoch number]
    print(reader._axon_info['dictEpochInfoPerDAC'][1][4]) # [DAC channel][Epoch number]
    print('**************')
    print(reader._axon_info['sections']['ProtocolSection'])
    print(reader._axon_info['sections']['ADCSection'])
    print(reader._axon_info['sections']['EpochSection'])
    print(reader._axon_info['sections']['EpochPerDACSection'])
    print('-------------------')
    # print(raw_protocol[:][:][1:10])
    # print(raw_protocol[:][:][-3])
    # print(raw_protocol[:][:][-2])
    # print(raw_protocol[:][:][-1])
    print(type(raw_protocol))
    print(len(raw_protocol))
    for item in raw_protocol:
        print(type(item))
        print(len(item))
        print(item,'\n')
    print(raw_protocol[0][0][0].shape)
    # print(dir(block))
    # print(dir(segs))
    # print(reader.block_count())
    #reader.block_count()
    # protocol = reader.read_protocol()
    # print(protocol.index)
    print('------------------')
    # print(segs[4].analogsignals)
    # print(dir(segs[0]))
    # print(segs[5].size)
    # print(segs[0].t_start)
    # print(segs[0].t_stop)
    # print(segs[0].size)
    # print(segs[0].file_datetime)
    # print('------------------')
    # print(dir(block[0]))
    # print(block[0].size)        # lists the number of segments and number of channels/segment
    # print(block[0].rec_datetime) # returns the file creation date
    # print(dir(block[0].annotate))

    return(block[0].segments[0].analogsignals[1:2])
    

def plot_data(x,y):
    fig = plt.figure()
    plt.plot(x,y)
    plt.show(block=False)
    input('press any key')
    plt.close()

# filename='/Volumes/Anup_2TB/raw_data/beiquelab/zen/data_anup/20190123/C2/2019_01_23_0042.abf'
filename='/Users/macbookair/goofy/data/beiquelab/2019_01_23_0042.abf'
ephysdata = axonclass(filename)
ephysdata.print_attributes()

# traces=read_axon_file(filename)
# print(np.shape(traces))
# plot_data(traces[0][:][0],traces[0][:][1])





