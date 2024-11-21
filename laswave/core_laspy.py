import numpy as np
import laspy
from scipy.interpolate import interp1d

# WAVEFORM BYTES THROUGH LASPY
def extractWdp(filePath, wdpPath):
    '''Extracts the waveform as Binary, the offset to each pulses waveform
    ,and the length of each pulses waveform.'''
    inFile = laspy.file.File(filePath, mode = "r")
    #if inFile.get_header().get_version() == '1.4':
    try: wf_byt_off = inFile.get_byte_offset_to_waveform_data()
    except NameError: wf_byt_off = []
    if inFile.header.minor_version < 4:
        try: wf_byt_len = inFile.get_waveform_packet_size()
        except NameError: wf_byt_len = []
    # workaround for bug in laspy as not consistent for las version 1.4
    if inFile.header.minor_version == 4:
        try: wf_byt_len = inFile._reader.get_dimension('wavefm_pkt_size')
        except NameError: wf_byt_len = []
    wf_byt_len = np.array(wf_byt_len, dtype='uint64') # conversion to 'uint64' neccessary for further use
    with open(wdpPath, "rb") as binary_file:
            # Read the whole file at once
            wf_byt = binary_file.read()
    return(wf_byt, wf_byt_off, wf_byt_len)



def extractWdpPulse(byte_data, byte_offset, byte_count, byte_size):
    '''Extracts the waveform byte position and the size per pulse from a provided
    binary array.'''
    # problem with extracting the last data point as unknown how long the
    # last pulse in th .wdp is, currently is set to 0
    # wf_pulse = np.frombuffer(byte_data[byte_offset: int(byte_offset + (byte_count * byte_size))],
    #                             dtype=np.uint8, count= int(byte_count * byte_size))
    wf_pulse = np.frombuffer(byte_data[byte_offset: int(byte_offset + (byte_count))],
                                dtype=np.uint8, count= int(byte_count))
    # wf_pulse = wf_pulse.reshape(int(byte_count), byte_size)
    wf_pulse = wf_pulse.reshape(int(byte_count/byte_size), byte_size) # reshape into two columns [X, 255 * Y]
    wf_pulse = wf_pulse * np.array([1,255]) # Y is so far only an integer representing full 255 cycles
    wf_pulse = np.sum(wf_pulse, axis=1)
    #wf_pulse = densifyData(wf_pulse, factor = 10)[0]
    return(wf_pulse)



def densifyData(arr_y, arr_x = None, factor = 1):
    '''Densifies the fullwaveform (or other) data  using a cubic interpolation.
    It aims to provide a more dense data distribution in the voxelSpace.
    For easier utilisation a density factor is translated to step_size.
    Variable step_size is describing the step size in comparisson to the orignal,
    step_size unit of arr_x. Therefore if the step_size unit of arr_x is provided
    in picoseconds at step_size = 0.1 corresponds to one one tens of a picosecond.'''
    if arr_x is None: # creates index array for x values
        arr_x = np.arange(0, len(arr_y))
    func = interp1d(arr_x, arr_y, kind= 'cubic') # fitted function
    step_size = (1 / factor)
    arr_inter_x = np.arange(0, np.nanmax(arr_x), step_size) # spacing array for densified data
    arr_inter_y = func(arr_inter_x).astype(int) # desified values
    return(arr_inter_y, arr_inter_x)


# pulse = extract_wdp_pulse(wf_byt, wf_byt_off[0], wf_byt_len[0])
# last_pulse = extract_wdp_pulse(wf_byt, wf_byt_off[-1], wf_byt_len[-1])
#
#
# # ## HOW TO APPLY IT TO DERIVE THE PULSES
# filePath = '/Users/Jordn/Documents/RSIV/fullwaveform/data/Golm-FWF-Base/Haus29_ID05_FWF.las'
# wdpPath ='/Users/Jordn/Documents/RSIV/fullwaveform/data/Golm-FWF-Base/Haus29_ID05_FWF.wdp'
# outPath = '/Users/Jordn/Documents/RSIV/fullwaveform/out/files'
#
# wf_byt, wf_byt_off, wf_byt_len = extractWdp(filePath, wdpPath)
#
# from itertools import repeat
# fwf = map(extract_wdp_pulse, repeat(wf_byt), wf_byt_off, wf_byt_len)
# fwf = np.array(list(out))


def printLasVar(file):
    '''Overview for all attributes (point format) of
    entries in a laspy file.'''
    inFile = laspy.file.File(file, mode = "r")
    pointformat = inFile.point_format
    for spec in inFile.point_format:
        print(spec.name)



def extendFromFiles(paths):
    '''attempts to create grasp the spatial extent from the maximum/ minimum
    extend found in the provided files'''
    files = {}
    if type(paths) is dict: # to be run if a dictionary is provided
        for subfile in paths:
            files[subfile] = laspy.file.File(paths[subfile][0], mode = "r")
    elif type(paths) is list: # to be run if a list of absolute file paths is provided
        for subfile in paths:
            files[subfile] = laspy.file.File(subfile, mode = "r")
    elif type(paths) is str:
        files = [laspy.file.File(subfile, mode = "r")]
    ## GLOBAL EXTEND
    # extract minimum and maximum extend of points
    input_min = []
    input_max = []
    for subfile in files:
        file_min = files[subfile].get_header().get_min()
        file_max = files[subfile].get_header().get_max()
        input_min.append(file_min)
        input_max.append(file_max)
    coord_min = np.nanmin( np.array(input_min, dtype= np.float64).transpose(), axis = 1)
    coord_max = np.nanmax( np.array(input_max, dtype= np.float64).transpose(), axis = 1)
    return(coord_min, coord_max)


##############################################################################
## OLD EXTRACTION METHOD

# def loadFwfFromWdp(file_path, wdp_path):
#     '''Loading fwf data directly from .wdp. Additionally a .las is required
#     as the correct position of the wdp is stored within that file. Consequently
#     the fwf is loaded through utilising the byte_offset from the .las file.'''
#     las_file = laspy.file.File(file_path, mode = "r")
#     wf_byt_off = las_file.byte_offset_to_waveform_data
#     with open(wdp_path, "rb") as binary_file:
#         wf_file = binary_file.read()
#     fwf = fwfFromByteoff(wf_file, wf_byt_off)
#     return(fwf)
#
#
#
# def fwfFromByteoff(wf_file, wf_byt_of):
#     '''extracting fwf from the .wdp (wf_file) utilising the byteoffset
#     (wf_byt_of). Currently not very efficient.'''
#     fwf = []
#     for i, e_byte in enumerate(wf_byt_of):
#         if i > 0:
#             raw_sig = ( wf_file[ s_byte: e_byte ] )
#             sig = sigFromByteoff(raw_sig,2)
#             fwf.append(sig)
#         s_byte = e_byte
#     return( np.array(fwf) )
#
#
#
# def sigFromByteoff(raw_sig, buffer=2):
#     '''Extracting single Signal from raw byte data (raw_sig). The buffer is
#     necessary as the byte is stored as integer values (uint8) and thus only store
#     values between 0 and 255. The second byte in bytepair stores the factor of
#     255 that needs to be added (2 * 255 = 510) '''
#     byte_count = int(len( raw_sig) / buffer)
#     wf_bytes = np.frombuffer(raw_sig, dtype=np.uint8, count= byte_count)
#     wf_bytes = wf_bytes.reshape( int(byte_count / buffer), buffer)
#     wf_bytes_val = wf_bytes * np.array([1,255])
#     wf_byte_val = np.sum(wf_bytes_val, axis=1)
#     return(wf_byte_val)
