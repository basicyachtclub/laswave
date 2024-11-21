import numpy as np
#from laswave_core import *
import laswave.core 

def fwfAnalysis(arr, meta, noise = 0, epsilon = 0.1, init_noise_len = 3):
    '''Function wrapping the entire analysis. Lets do a real wrapper this time.'''
    ## initialisation of the arrays that will hold all data
    # Position
    pos = []
    error_pos = []
    #
    # Noise
    # noise = int(noise * (1 +  epsilon))
    noise = []
    #
    # Main Signal
    sig_main = []
    sig_str = []
    error_main = []
    error_str = []
    #
    # Peaks
    sig_peak = []
    sig_peak_extra = []
    error_peak = []
    #
    # Skew
    sig_skew = []
    error_skew = []
    #
    # Offset
    peak_offset = []
    error_offset = []
    #
    # Riegl Peak
    riegl_peak = []
    #
    counter = 0 # incremental counter
    for i in arr: # RUN THROUGH ALL SIGNAL INSTANCES
        meta_i = meta[counter] # get the meta data for the corresponding fwf
        # check if the signal is continuous
        # on the fly noise detection as a mean from the first X signals
        noise_i = round((np.sum(i[0:3]) / init_noise_len ))
        noise_i = round(noise_i * (1 + epsilon))
        noise.append(noise_i)
        #
        #
        ##### Deriving the signal above the noise threshold
        try:
            pos_i = subSignals(i, noise_i)
            sig_main_i, sig_str_i = getMainSignal(i, pos_i)
            sig_main.append(sig_main_i)
            sig_str.append(sig_str_i)
        except:
            sig_main.append([np.nan,np.nan])
            sig_str.append(np.nan)
            error_main.append(counter)
            error_str.append(counter)
        #
        #
        ##### get local maxima and the peak values
        try:
            sig_peak_i, sig_peak_extra_i = getSignalPeaks(i, 1, noise_i)
            sig_peak.append(sig_peak_i)
            sig_peak_extra.append(sig_peak_extra_i)
        except:
            sig_peak.append(np.nan)
            sig_peak_extra.append(np.nan)
            error_peak.append(counter)
        #
        #
        ##### derive skew
        try:
            skew_i = primitivSkew(i , sig_main_i)
            sig_skew.append(skew_i)
        except:
            sig_skew.append(np.nan)
            error_skew.append(counter)
        #
        #
        ##### derive offset
        try:
            peak_offset_i, riegl_peak_i = peakOffset(i, meta_i)
            peak_offset.append(peak_offset_i)
            riegl_peak.append(riegl_peak_i)
        except:
            peak_offset.append(np.nan)
            riegl_peak.append(np.nan)
            error_offset.append(counter)
        #
        #
        counter += 1
    #
    #
    # Convert to Numpy array
    arr = np.array(arr)
    meta = np.array(meta)
    noise = np.array(noise)
    sig_main = np.array(sig_main)
    sig_str = np.array(sig_str)
    sig_peak = np.array(sig_peak)
    #sig_peak_extra = np.array(sig_peak_extra)
    sig_skew = np.array(sig_skew)
    peak_offset = np.array(peak_offset)
    riegl_peak = np.array(riegl_peak)
    #
    #### Print out errors
    print('ERROR LOG:')
    print('main signal:  ' + str(len(error_main)))
    print('signal strength:  ' + str(len(error_str)))
    print('peaks:  ' + str(len(error_peak)))
    print('skew:  ' + str(len(error_skew)))
    print('offset:  ' + str(len(error_offset)))
    #
    return(arr, meta, noise, sig_main, sig_str, sig_peak, sig_peak_extra, sig_skew, peak_offset, riegl_peak)
