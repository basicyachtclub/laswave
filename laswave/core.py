# TOOLSET FOR FULL WAVEFORM ANALYSIS
import numpy as np
import laspy
import laswave.auxillary
from scipy.signal import find_peaks

'''
DICTIONARY FOR COLUMNS IN ASCII FILE FROM .las VERSION 1.3:

xcoord     X Coordinate of the calculated return point (X COORDINATE)
ycoord     X Coordinate of the calculated return point (X COORDINATE)
zcoord     X Coordinate of the calculated return point (X COORDINATE)
int        intensity (INTENSITY)
ret        number of returns for signal (RETURNS)
retnr      return number within signal (RETURN NR)
gpst       GPS time when emitting signal (GPS TIME)
??         - Maybe -  classification
mpos       memory position (in bits) (MEMORY POSITION)
mlen       memory length (in bits) (MEMORY LENGTH)
post       time in signal at which return point was estimated (POSITIONAL TIME)
vecx       vector orientation X of the signal (X VECTOR)
vecy       vector orientation Y of the signal (Y VECTOR)
vecz       vector orientation Z of the signal (Z VECTOR)
'''

header_format = ['xcoord','ycoord','zcoord','int','ret','retnr','gpst','??',
                'mpos','mlen','post','vecx','vecy','vecz']





def loadFwfFromAscii(fn, comments = '#', skip_col = 17, time_col = None):
    '''Loading in an ASCII file created from .wdp file. skip_col specifies
    at which collumn the break between meta data and waveform data is done'''
    wf = []
    counter = 0
    file_end = False
    with open(fn, 'r') as f:
        while file_end == False:
            l = f.readline()
            try:
                if l[0] == comments:
                    print('line ' + str(counter) + ' excluded as comment')
                    counter += 1
                else:
                    i = l.split(' ')
                    wf.append(np.array(i[skip_col:], dtype = 'int'))
            except:
                # print('reached end of ASCII file.')
                file_end = True
    m = np.loadtxt(fn, usecols = np.arange(skip_col), comments = comments)
    #t = np.loadtxt(fn, usecols = time_col, comments = comments) # time
    print('ASCII file loaded into memory.')
    return (m, wf)

#m, wf = loadFwfFromAscii('/Users/Jordn/Documents/RSIV/fullwaveform/data/\
#Golm-FWF/ID04/Haus29_ID04_FWF_V13_xyzinrtWV.asc')



def pullMeta(string, meta, dict):
    '''extracts one variable from the meta data, generated from an ASCII file'''
    col_pos = 0
    for ele in dict:
        if ele == string:
            return(meta[:,col_pos])
        col_pos += 1
    raise Exception('could not find ' + ele)



def getSignal(data, pos):
    '''returns single Signal from entire dataset'''
    return(data[pos])



def getMeta(meta_data, pos, var=None, header_format=None):
    '''returns single meta data subset or single meta data variable
    from entire meta dataset'''
    if var is None and header_format is None:
        return(meta_data[pos])
    else:
        return(meta_data[pos, header_format.index(var)])



# column 10 contains proposed time of maximum return etimated by Riegl algorithm
def peakOffset(signal, mpos_a):
    '''calculates how much the peak predicted by the riegl algorithm is
    differing from the discrete maximum within the return signal
    i.e. how much the riegl algorithm is offset from the discrete
    maximum (riegl - discrete). Returns bot offset and time algorithmic
    maximum.'''
    # convert pico to full nano seconds
    mpos_a = np.around((mpos_a / 1000), decimals = 3)
    # get position of discrete maximum in the return signal (timestamp)
    mpos = np.where(signal == np.nanmax(signal))[0][0]
    return(np.around((mpos_a - mpos), decimals = 1), mpos_a)



def primitivSkew(data, main_sig):
    '''simple skew, positive if right, negative if left skewed. Takes in
    a Subset of the fwf data (e.g. the main signal) with relative indices.
    from 0 to X.'''
    if main_sig.size == 0: # check if signal exists
        return(np.nan)
    else:
        sig_len = main_sig.size
    # getting maximum position within the main signal
    sig_val = data[main_sig]
    # position of the main signal maximum within the main signal
    mpos = np.where(data[main_sig] == np.nanmax(sig_val))[0][0]
    #mpos = mpos + main_sig[0] # position of main signal maximum within the entire signal
    skew = ((sig_len - mpos) - (mpos)) #/ sig_len
    #skew = int(skew)
    return(skew)



def subSignals(sig, noise = 0, max_sig_len = None):  #getSignalPosition
    '''Determines where the signal yields returns above a certain threshold (noise).
    Will return an empty array if no returns are observed above the noise level.
    Will also return an empty array if all entries exceed the max_sig_len length.'''
    ind = np.array(np.where(sig > noise), dtype = np.int16)[0] # creates an array containing all indices above the noise
    sub_sig_idx = np.array(consecutive(ind)) # split array into consecutive signales (bundled into one array)
    if max_sig_len is None:
        return(sub_sig_idx)
    else:
        i = 0
        while i < len(sub_sig_idx):
            if len(sub_sig_idx[i]) > max_sig_len:
                #sub_sig_idx[i] = np.array([],  dtype = np.int16)
                sub_sig_idx = np.delete(sub_sig_idx, i)
            else:
                i += 1
        return(sub_sig_idx)

#sig = getSignal(wf, 1)
#subSignals(sig, 158)
#subSignals(sig, 158, 5)

    #if sub_sig.size > 0:
    #    return(sub_sig)
    #else:
    #    if sub_sig.size == 0:
            # occurs if the noise is consuming the signal (only zero values)
            #print('No Signal detected, Noise level to high.')
    #        return(np.array([])) # no 'stretch' in the signal
            #
        #elif sub_sig.size == sig.size:
            # occurs if the signal is covering the entire area (no zero values)
            #print('Signal covers entire length.')





def getSignalPeaks(data,  signal_width, noise = 0):
    '''Peak finding within a Signal or Sub-Signal.'''
    peak_pos, heights = find_peaks(data, height= noise, width= signal_width)
    return(peak_pos)



def getSignalPeaksArr(sig, sig_width, noise):
    ''' Extracts the peak positions for all Signal entries in an array.
    Currently Noise level is static.'''
    pks = map(getSignalPeaks, sig, repeat(sig_width), repeat(noise)) #noise is set to zero
    pks = np.array(list(pks))
    return(pks)



def getMainSignal(sig, sub_sig_idx):
    '''Main Signal here corresponds to the biggest integral of detected signals.
    Also returns the main signals integral. Thus it is important to identify the
    right noise level beforehand.'''
    if sub_sig_idx.size == 0: # here to make sure no empty signal is evaluated
        return(sub_sig_idx, sub_sig_idx) # twice to account for the signal strengh as well
    #
    sig_str = np.empty(sub_sig_idx.size, dtype=np.int16) # empty array for signal strengh
    i = 0
    #
    for index in sub_sig_idx: # calculating the 'integrals' of the individual signals
        sig_str[i] = np.nansum(sig[index], dtype=np.int16)
        i += 1
    #
    pos = np.where(sig_str == np.max(sig_str))[0] # position of the main signal
    #sig_str_main = sig_str[pos]
    #sig_main = sub_sig_idx[pos]
    return(sub_sig_idx[pos][0],sig_str[pos][0]) # returns Main Signal and the signal 'integral'



from matplotlib import pyplot as pl

def valueToRgb(fn, var, pnts, create_baseline = True, exclude_x_percent = 0,
                cmap = pl.cm.YlOrRd): #pl.cm.RdBu #pl.cm.Spectral
    '''Create a new .las file mapping the speficid variable to a colormap.'''
    v = var.copy()
    # Trunkate the data to exclude outliers as specified
    v[v >= np.nanpercentile(v, (100 - exclude_x_percent))] = \
                            int (np.nanpercentile(v, (100 - exclude_x_percent)) )
                            # truncate data (cut off high values)
    v[v <= np.nanpercentile(v, (0 + exclude_x_percent))] = \
                            int (np.nanpercentile(v, (0 + exclude_x_percent)) )
                            # (cut off low values)
    #
    if np.nanmin(v) > 0:
        v = v - np.nanmin(v)
    #if norm == True:
    if np.nanmax(v) >= -np.nanmin(v):
        norm = np.nanmax(v)
    else:
        norm = -np.nanmin(v)
    v = v/norm # normalizing the variable
    #
    if np.nanmin(v) < 0:
        v /= 2 # normalizing the variable so the maximum is 0.5
        v += 0.5 # lifting all values into the positive spectrum
    #if create_baseline == True:
    #    v = v - np.nanmin(v) # part of normalizing the variable, creating a baseline
    rgb = cmap(v) # maps a colormap to the values
    rgb = rgb[:, :3] # uses all columns execpt the third one (which contains only 1.) maybe intensity??
    rgb *= 65535 # convert to 16bits color model
    rgb = rgb.astype('uint') # convert to unsigned integer (fixed range, in this case 0 to 65535)
    header = laspy.header.Header() # create laspy Header class
    header.data_format_id = 2 # number of user-defined header records in the header
    f = laspy.file.File(fn, mode = 'w', header = header) # creates the blanco file
    f.header.scale = [0.001, 0.001, 0.001] # https://pythonhosted.org/laspy/header.html#laspy.header.HeaderManager.scale
    f.header.offset = [pnts[:,0].min(), pnts[:,1].min(), pnts[:,2].min()] # Why do we use the offset here?
    f.x = pnts[:, 0]
    f.y = pnts[:, 1]
    f.z = pnts[:, 2]
    if pnts.shape[1] == 4:
        f.intensity = pnts[:, 3]
    f.set_red(rgb[:, 0]) # assigned pre calculated color values to rgb values inside the .las
    f.set_green(rgb[:, 1])
    f.set_blue(rgb[:, 2])
    f.close()
    print('  new .las file created')




def waveformToCoords(time, vector, point):
    '''Transforms a waveform signal, a positional vector, and a return point
    into coordinates. Done for all observed erturn pulses in the signal.'''
    coord =np.array(list(map(lambda pulses: ( (pulses*vector)+point) , time)))
    return(coord)



def waveformToCoordsArr(peak_pos_arr, vec_dir_arr, pts_arr):
    '''Transforms waveform array, positional vector array, return
    point array into a coordinate array of all entries provided in
    the waveform array'''
    coords_arr = np.empty(0)
    for peak_pos, vec_dir, pts in zip(peak_pos_arr, vec_dir_arr, pts_arr):
        coords = (waveformToCoords(peak_pos, vec_dir, pts))
        coords_arr = np.append(coords_arr, coords)
        coords_arr = coords_arr.reshape(coords.shape[0]/3, 3)
    return(coords_arr)





def fwfFromByteOffsetAscii(file, ascii, asc_col):
    '''Secondary function for extraction of fwf data (array of signals).
    Can be used if the waveform comes in ASCII format rather then in .wdp'''
    inFile = laspy.file.File( file, mode = "r")
    #
    # getting byte offset of the fwf data for the las file
    fwf_byte_offset_las = inFile.get_byte_offset_to_waveform_data()
    #
    # data type for consintency
    used_dtype = fwf_byte_offset_las.dtype
    #
    # getting the meta data
    meta, fwf_ascii = loadFwfFromAscii(ascii)
    #
    # extract all byte offsets from the meta data
    fwf_byte_offset_asc = np.array(meta[:,asc_col], dtype= used_dtype)
    #
    # now extraction from the file, bringing the two together
    fwf_las = []
    meta_las = []
    #
    for i in fwf_byte_offset_las:
        pos = np.where(fwf_byte_offset_asc == i)[0][0]
        fwf_las.append(fwf_ascii[pos])
        meta_las.append(meta[pos])
    #
    return(meta_las, fwf_las)




# def ExtractFwfByReturn(file, ascii, asc_col, return_nr = 0):
#     '''Secondary function to exctract fwf. Extracting specific returns from
#     a .las and a corresponding ascii file.'''
#     inFile = laspy.file.File( file, mode = "r")
#     #
#     # getting byte offset of the fwf data for the las file
#     fwf_return_las = inFile.get_return_num()
#     #
#     # data type for consintency
#     used_dtype = fwf_return_las.dtype
#     #
#     # getting the meta data
#     meta, fwf_ascii = loadFwfFromAscii(ascii)
#     #
#     # extract all byte offsets from the meta data
#     fwf_return_asc = np.array(meta[:,asc_col], dtype= used_dtype)
#     #
#     if return_nr != 0:
#         fwf_return_las = np.array([return_nr])
#     #
#     fwf_las = []
#     meta_las = []
#     #
#     # STILL NEEDS WORKING OUT HOW TO ONLY RETURN VALUES FROM THE SUBSET
#     # SO FAR ALL SECOND RETURNS ARE RETURNED.
#     for i in fwf_return_las:
#         pos = np.where(fwf_return_asc == i)[0]
#         fwf_las = fwf_ascii[pos]
#         meta_las = meta[pos]
#     #
#     return(meta_las, fwf_las)
