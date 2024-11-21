## SETUP
import laspy
import numpy as np
from lmfit import Model
from lmfit.models import ConstantModel, GaussianModel, ExponentialGaussianModel, LognormalModel, SkewedGaussianModel
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from itertools import repeat
from laswave.core_laspy import densifyData
from laswave.core_georef import blockBuildCrdArr


def lasToMemory(file_path): # , wdp_path):
    inFile = laspy.file.File(file_path, mode = "r")
    return(inFile)

def make_Gaussian_model(num, amplitude_guess, center_guess, fwhm_guess):
    pref = "f{0}_".format(num)
    model = GaussianModel(prefix = pref)
    model.set_param_hint(pref+'amplitude', value=amplitude_guess, min=amplitude_guess/2, max=3*amplitude_guess)
    model.set_param_hint(pref+'center', value=center_guess, min=center_guess-2, max=center_guess+2)
    model.set_param_hint(pref+'fwhm', value=fwhm_guess, min=4, max=30)
    return model

def make_ExponentialGaussian_model(num, amplitude_guess, center_guess, fwhm_guess):
    pref = "f{0}_".format(num)
    model = ExponentialGaussianModel(prefix = pref, nan_policy = 'propagate')
    model.set_param_hint(pref+'amplitude', value=amplitude_guess)
    model.set_param_hint(pref+'center', value=center_guess)
    model.set_param_hint(pref+'sigma', value=fwhm_guess, min=4, max=30)
    return model

def make_SkewedGaussian_model(num, amplitude_guess, center_guess, fwhm_guess):
    pref = "f{0}_".format(num)
    model = SkewedGaussianModel(prefix = pref, nan_policy = 'propagate')
    model.set_param_hint(pref+'amplitude', value=amplitude_guess)
    model.set_param_hint(pref+'center', value=center_guess)
    model.set_param_hint(pref+'sigma', value=fwhm_guess, min=4, max=30)
    return model

def make_Lognormal_model(num, amplitude_guess, center_guess):
    pref = "f{0}_".format(num)
    model = LognormalModel(prefix = pref, nan_policy = 'propagate')
    model.set_param_hint(pref+'amplitude', value=amplitude_guess)
    model.set_param_hint(pref+'center', value=center_guess)
    return model



def fitModel(fwf_arr, idx_arr, amplitude_threshold_to_ignore = 200,
                densify_factor = 4, fwhm_guess = 6, noise_floor_length = 5,
                noise_factor = 1, model_type = 'gaussian'):
    '''Fits Model to Signal and return Parameters, Signal Width, Signal Amplitude at Peak,
    Signal FWHM, Signal Center and Model Values.'''
    # SCALE DOWN THE SIZE OF THE ARRAY ACTUALLY WORKED ON FOR TESTING
    #fwf_arr = fwf_arr[0:100]
    #idx_arr = idx_arr[0:100]
    # Initial Setup
    #model_fit_list = []
    #model_c = []#np.empty( len(idx_arr) )
    model_fit_amplitude = [] #np.empty( len(idx_arr) )
    model_fit_center = [] #np.empty( len(idx_arr) )
    #model_fit_sigma = []#np.empty( len(idx_arr) )
    #model_fit_fwhm = []#np.empty( len(idx_arr) )
    #model_values = []
    I = len(fwf_arr)
    # Model Selection
    if model_type is 'gaussian':
        model = make_Gaussian_model
    elif model_type is 'exponential_gaussian':
        model = make_ExponentialGaussian_model
    # Model Application
    for i, (fwf, idx) in enumerate(zip(fwf_arr, idx_arr)):
        print('%d of %d'%(i, I))
        noise_floor_level = np.mean(fwf[0:noise_floor_length])
        amplitude_threshold_to_ignore = noise_floor_level * noise_factor
        s = densifyData(idx, factor = densify_factor)[0]
        usp = UnivariateSpline(idx, fwf)
        peaks, properties = find_peaks(usp(s), prominence = 2)
        # Amplitude Filtering
        # fpeaks = []
        # for j in range(len(peaks)): #only use peaks above amplitude threshold
        #     amplitude = usp(s)[peaks[j]]
        #     if amplitude > amplitude_threshold_to_ignore:
        #         fpeaks.append(peaks[j])
        # fpeaks = np.array(fpeaks)
        amplitude = usp(s)[peaks]
        fpeaks = peaks[np.where(amplitude > amplitude_threshold_to_ignore)[0]]
        # Building the Model
        model_convolution_full = None # initial condition
        for j in range(len(fpeaks)):
            amplitude_guess = usp(s)[fpeaks[j]]
            center_guess = s[fpeaks[j]]
            model_convolution_part = model(j, amplitude_guess, center_guess, fwhm_guess)
            if model_convolution_full is None:
                model_convolution_full = model_convolution_part
            else:
                model_convolution_full = model_convolution_full + model_convolution_part
        constantoffset = noise_floor_level # utilises the first 5 entries to get noise plateau
        offset = ConstantModel()
        offset.set_param_hint('c', value=constantoffset, min=np.percentile(fwf,2), max=constantoffset + (0.2 * constantoffset))
        model_convolution_full = offset + model_convolution_full
        # Fitting the model and data extraction
        try:
            model_fit = model_convolution_full.fit(fwf, x=idx)
            #num_peak_mod = (len(model_fit.var_names) - 1) / 3 # No of peaks in model
        except ValueError:
            model_fit = np.nan
        else:
            model_fit_amplitude_i = []
            model_fit_center_i = []
            for k, p in enumerate(fpeaks):
                model_fit_amplitude_i.append(model_fit.values['f%d_amplitude'%k])
                model_fit_center_i.append(model_fit.values['f%d_center'%k])
                # model_c[i] = np.array(model_fit.values['c'] ) #baseline
                # model_fit_amplitude[i] = np.array(model_fit.values['f0_amplitude'] )
                # model_fit_center[i] = np.array(model_fit.values['f0_center'] )
                # model_fit_sigma[i] = np.array(model_fit.values['f0_sigma'] )
                # model_fit_fwhm[i] = np.array(model_fit.values['f0_fwhm'] )
                # model_values.append( model_fit.best_fit )
            # ONLY RETURN AMPLITUDE AND POSITION
            model_fit_amplitude.append(np.rint(model_fit_amplitude_i).astype(np.int))
            model_fit_center.append(np.rint(model_fit_center_i).astype(np.int))
        #model_fit.conf_interval()
        #model_fit_list.append(model_fit) # in case more information is to be extracted
    return(np.array(model_fit_amplitude), np.array(model_fit_center))




# def removeNoise(sig, noise_threshold):
#     '''Removes noise from the FWF array'''
#     noise_floor = (np.min(sig) * noise_threshold).astype(np.int)
#     sig[sig <= noise_floor] = 0 # where values are smaller then noise
#     return(sig)

def modelNoise(fwf, sig_factor = 3, return_model = False):
    '''Calculates histogram (bin_size=1) for intensity values below the mean,
    fits skewed gaussian model (or other) to the histogram with initialised parameters.
    Calculates noise as center (mu) of the model plus x * sigma (default:3).'''
    intensity = np.hstack(fwf)
    bins = np.arange(np.min(intensity), np.mean(intensity), 1) # values over the mean cannot constitute to noise
    bins = np.append(bins, np.max(intensity)) # account for high values
    hist, xedges = np.histogram(intensity, bins = bins)
    hist_noise = hist[0:-1] # exclude the 'high' values to only look at noise
    idx = np.indices(hist_noise.shape)[0]
    peak_pos = np.where(hist_noise == np.max(hist_noise))[0][0]
    print('modelling noise (skewed gaussian)')
    model = make_SkewedGaussian_model(num = 0, amplitude_guess = np.max(hist_noise),
                            center_guess = peak_pos , fwhm_guess = 15)
    model_fit = model.fit(hist_noise, x=idx)
    amp = model_fit.values['f0_amplitude']
    cnt = model_fit.values['f0_center']
    std = model_fit.values['f0_sigma']
    noise = bins[np.round( cnt + sig_factor * std ).astype(int)] # extract noise value
    # try: # try logging: Noise Peak Amplitude to Mean Intensity Ratio
    #     amp_ratio = np.round(np.mean(hist_noise) / np.max(hist_noise), 3)
    #     logger.info("Itensity ratio - mean PDF to peak PDF: %s"%str(amp_ratio))
    #     PSNR = np.round( np.max(intensity) / noise, 3) # Peak signal-to-noise ratio
    #     logger.info("PSNR: %s"%str(PSNR))
    # except NameError: pass
    if return_model is True:
        return(hist_noise, xedges[:-1], model, model_fit, noise)
    else:
        return(noise)


def maskNoise(sig, noise_threshold):
    '''Returns a boolean array, masking out values below noise threshold'''
    noise_floor = (np.min(sig) * noise_threshold).astype(np.int)
    return(sig > noise_floor) # where values are larger than Noise


def maskNoiseStatic(sig, noise_floor):
    '''Returns a boolean array, masking out values below static noise floor'''
    return(sig > noise_floor)


def maskRemove(sig, mask):
    '''fits the provided mask to the signal, subfunction
    for use in map() function'''
    return(sig[mask])


def removeHeightOutlier(coords, fwf, low_per = 0.25, high_per = 99.75):
    '''Removes outliers in Z-Dimension based on percentile thresholds.
    These can occur in the data from multiple reflections at target.'''
    low_z_pos = np.where(coords[:,2] < np.percentile(coords[:,2], low_per))[0]
    high_z_pos =  np.where(coords[:,2] > np.percentile(coords[:,2], high_per))[0]
    outlier_pos = np.hstack((low_z_pos, high_z_pos))
    coords_no_outl = np.delete(coords, outlier_pos, axis = 0)
    fwf_no_outl = np.delete(fwf, outlier_pos)
    print('removed %d outlier'%len(outlier_pos))
    return(coords_no_outl, fwf_no_outl)


def returnNumberExtractor(inFile, returns = 'all'):
    '''Builds indexing array for specific returns within a .las file.'''
    ret_nr = inFile.get_return_num()
    if returns == 'all':
        ret_nr_pos = np.arange(0,len(inFile.x))
    elif returns == 'last':
        num_returns = inFile.get_num_returns()
        ret_nr_pos = np.where(ret_nr == num_returns)[0]
    elif returns == 'ground':
        num_returns = inFile.get_num_returns()
        ret_only_first = np.where(num_returns == 1)[0] # only where 1 return exists
        ret_last_return = 5
    else: # here returns need to be integer value
        ret_nr_pos = np.where(ret_nr == returns)[0]
    return(ret_nr_pos)



def extractFwfPts(file_path, wdp_path, inFile =None, extractor = None, byte_size = 2):
    '''Extracts fullwaveform arrays and the discrete points from .wdp through
    byte offset.'''
    ## SETUP
    if inFile is None: # in case no laspy file instance is parsed
        inFile = lasToMemory(file_path)
    if extractor is None: # in case no extractor is parsed
        extractor = np.arange(0,len(inFile.x))
    ## XYZ POINTS (PTS)
    pts = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    pts = pts[extractor]
    ## FWF SIGNAL
     # .wdp internal byte structure (number of bytes per observation), for waveform packet
    print('extracting FWF data from file: ' + wdp_path)
    wf_byt, wf_byt_off, wf_byt_len = extractWdp(file_path, wdp_path)
    fwf_map = map(extractWdpPulse, repeat(wf_byt), wf_byt_off[extractor], \
                wf_byt_len[extractor], repeat( byte_size ))
    fwf = np.array(list(fwf_map))
    return(fwf, pts)


# old name fwfToCoords
def fwfCoordsFromFile(file_path, wdp_path, returns= None, refractive_index= 1.00028, \
                        time_step_size= 1005, time_step_oom= (-12), densify = False,
                        densify_factor = 2, bin_size_oom= 2, model=None,
                        noise_factor=None, noise_val=None, rmv_outl = True):
    '''Extract both Coordinates and Intensity value from a .las and corresponding
    .wdp file. Can also extract a specific type of return (e.g. 'first', 'last').
    If time steps between laser pulses in the fullwaveform must be provided.
    If models are to be fitted, model name must be provided ('gaussian' and
    'exponential gaussian'). If only noise is to be removed a noise_factor must
    be provided which will be applied to the lowest recoded value.'''
    # SETUP
    #noise_val = 170
    inFile = lasToMemory(file_path)
    scale = inFile.header.scale
    offset = inFile.header.offset
    #returns = 'last' # can this be done to speed up the process and reduce the number of points??
    if returns is None:
        extractor = np.arange(0,inFile.header.point_records_count)
    else:
        extractor = returnNumberExtractor(inFile, returns = returns)
    # CONVERSION FACTORs FROM TRAVELED TIME TO DISTANCE
    #c_0 = 299792458 # speed of light [m/s]
    #c_m = (c_0 / refractive_index) # divided by refractive index of medium, default set to air at sea level
    #c_m = c_m * 10**(time_step_oom) # distance traveled in [m] at time step. time step order of magnitude default: pico seconds
    # PTS, FWF EXTRACTION
    fwf, pts = extractFwfPts(file_path, wdp_path, inFile = inFile)
    fwf = fwf[extractor]
    pts = pts[extractor]
    # CREATION OF INDEX ARRAY FOR TIME STEPS BETWEEN PULSES
    if densify is True: # a densify_factor of 2 doubles temporal resolution
        for pos, signal in enumerate(fwf):
            fwf[pos] = densifyData(signal, factor = densify_factor)[0]
    sig_idx = buildIndexArray(fwf, start = 0)
    sig_idx *= time_step_size # Adjusting timesteps if not the're not exactly 1*time_step_oom, value should be 1005 (stated in .lasinfo as timestep)
    # NOISE REMOVAL
    # default filters through a skewed gaussian model on the intensity histogram for all points
    if noise_factor is not None:
        print('removing noise below %d times of minimum value'%noise_factor)
        noise_map = map(maskNoise, fwf, repeat(noise_factor))
        #noise_mask = np.array(list(noise_map))
    elif noise_val is not None:
        print('removing noise below %d from data'%noise_val)
        noise_map = map(maskNoiseStatic, fwf, repeat(noise_val))
    else:
        noise_val = modelNoise(fwf, sig_factor = 3)
        print('removing noise below %d from data'%noise_val)
        noise_map = map(maskNoiseStatic, fwf, repeat(noise_val))
    noise_mask = np.array(list(noise_map))
    # MODEL FITTING
    if model is 'gaussian' or model is 'exponential_gaussian':
        if noise_factor is None:
            fwf, sig_idx = fitModel(fwf, sig_idx, model_type = model)
        else:
            fwf, sig_idx = fitModel(fwf, sig_idx, model_type = model, noise_factor = noise_factor)
    # REMOVE ENTRIES BELOW NOISE LEVEL
    #elif noise_factor is not None or noise_val is not None: # if models are fitted only peaks remain
    idx_noise_map = map(maskRemove, sig_idx, noise_mask)
    sig_idx_no_noise = np.array(list(idx_noise_map))
    fwf_noise_map = map(maskRemove, fwf, noise_mask)
    fwf_no_noise = np.array(list(fwf_noise_map))
    fwf_flat = np.hstack(fwf_no_noise)
    # GET OFFSET OF REGISTERED POINT TO FWF IN TIME
    #pos = np.round((inFile.return_point_waveform_loc)),2)
    pos = inFile.return_point_waveform_loc.astype(np.int) # resolution in full picoseconds
    pos = pos[extractor]
    # GET OFFSET FROM PEAKS IN PULSE (as time)
    #sig_off_t = np.subtract(sig_idx_no_noise ,pos) # time interval beetwen pulse steps
    sig_off_t = np.subtract(pos, sig_idx_no_noise)
    # CONVERSION FROM TIME TO DISTANCE not needed, done explicitly through vector
    #sig_off_d = np.multiply(sig_off_t, c_m) # distance interval between pulse steps
    #sig_off_d = sig_off_t
    # REVERSE, AS ORIGINAL VECTORS POINT TOWARDS THE SENSOR
    #sig_off_d = np.multiply(sig_off_d, -1) # not sure if vectors are already flipped when stored, investigate
    # DIRECTIONAL VECTORS
    vec_dir = np.vstack((inFile.x_t, inFile.y_t, inFile.z_t)).transpose()
    vec_dir = vec_dir[extractor]
    #vec_dir = vec_dir * 1/np.array(scale)
    ## COORDINATES FROM SIGNAL, TILING THE DATA into 10**bin_size_oom tiles of POINTS
    print('building coordinate array, tilesize = ' + str(10**bin_size_oom))
    pts_fwf, pts_fwf_map = blockBuildCrdArr(sig_off_t, vec_dir, pts, bsize= bin_size_oom)
    #EXCLUDE OUTLIER ALONG Z-AXIS
    if rmv_outl is True:
        pts_fwf, fwf_flat = removeHeightOutlier(pts_fwf, fwf_flat, low_per = 0.05, high_per = 99.95)
    return(pts_fwf, fwf_flat)


    # IF .LAS IS TO BE VISUALIZED
    # name convetion kept same for easy usage
    #pts = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    #fwf_flat = inFile.intensity
    #pts = pts - off

    # QUICK VISUALISZATION FOR IMAGES
    off = np.min(pts_fwf, axis=0)
    off = np.array([33362306.778377026, 5808423.2205186775, 34.1590444767815])
    pts = pts_fwf - off

    # VIEWER PROPERTIES
    import pptk
    v = pptk.viewer(pts)#[np.arange(pos-100,pos+100 +1)])
    v.attributes(fwf_flat)#[np.arange(pos-100,pos+100 +1)])
    v.set(point_size=0.005)
    #col_top = [255,255,255,1]
    #col_bottom = [255, 255, 255,1]
    #v.set(bg_color_top = np.array(col_top) / np.array([255,255,255,1]))
    #v.set(bg_color_bottom = np.array(col_bottom) / np.array([255,255,255,1]))
    v.color_map('jet', [150,500])

    # CAMERA PROERTIES AND SCREENSHOT
    v.set(show_info = False) # Show information text overlay
    v.set(show_axis = False) # Show axis / look-at cursor
    v.set(show_grid = False) # Show floor grid
    view_point = np.percentile(pts,50, axis=0)
    view_point[2] = view_point[2] - np.percentile(pts,10, axis=0)[2]
    view_point = np.array([5.89981328, 6.32748254, 8.94487342])
    v.set(lookat = view_point)
    v.set(r = 30) # camera distance
    v.set(theta = 0.0)
    v.set(phi = 0.0) # 0
    v.capture('/Users/Jordn/Desktop/pptk_view_y_axis.png')
    v.set(phi = 1.570796) # 90
    v.set(phi = 3.141593) # 180
    v.set(phi = 4.712389) # 270
    v.capture('/Users/Jordn/Desktop/pptk_view_x_axis.png')

    #
    v.get('lookat') # Camera look-at position
    v.get('eye') # Camera position
    v.get('r') # Camera distance to look-at point
    v.get('right') # Camera Right vector
    v.get('theta') # Camera elevation angle (radians)



    #IDENTIFY VERY LOW LYING POINTS
    #MUST BE DONE BEFORE OUTLIER FILTERING
    min_pos = np.where(pts_fwf[:,2] == np.min(pts_fwf[:,2]))[0][0]
    # FIND CORRESPPONDING FWF ARRAY
    i = 0
    j = 0
    for sig in fwf_no_noise:
        i += len(sig)
        if i >= min_pos:
            pos_in_sig = min_pos - i
            break
        j += 1 # index for FWF

    return_point_loc = inFile.return_point_waveform_loc.astype(np.int)[j]
    return_point_loc = return_point_loc/1000
    return_point_int = inFile.intensity[j]

    # VISUALIZE INDIVIDUAL WAVEFORMS
    from matplotlib import pyplot as plt
    #fwf_idx = j-1
    #plt.plot(np.arange( 0,len(fwf[fwf_idx]) ), fwf[fwf_idx])
    #plt.show()
    extract = np.arange(j-0,j+0 +1)
    extract = np.arange(0,100)
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlim([0,150])
    ax.set_ylim([150,1000])
    ax.set_xlabel('Range [nano sec]', fontsize=10)
    ax.set_ylabel('Amplitude ', fontsize='medium')

    for signal in fwf[extract]:
        #ax.fill(signal, alpha=0.1);
        ax.fill_between(np.arange(0, len(signal)),signal, 0, alpha =0.1)
        ax.plot(signal, linewidth = 0.2, alpha=0.5);
        #ax.plot(return_point_loc, return_point_int, marker='o', linestyle =' ', markersize=0.5, color="red")
    plt.show()


    # HISTOGRAM PLOT
    from matplotlib import pyplot as plt
    # Non FWF data of same File
    data = inFile.intensity
    # FWF data
    data = np.hstack(fwf_no_noise)
    data = np.hstack(fwf)

    #logbins = np.logspace(np.log10(np.nanmin(data)),np.log10(np.nanmax(data)) , 20)# np.unique(data).shape[0])
    #bins = np.unique(data)[ np.unique(data) < np.mean(data)]          # np.nanpercentile(data,99.9)] # EXCLUDING FREAK VALUES
    #bins = np.arange(noise_val,np.nanpercentile(data,99.9), 10)
    #bins = np.arange(0,np.max(data), 10)
    #bins = np.arange(400,1000,10)
    #bins = np.unique(data)
    #bins = np.round(np.arange(0,2.5,0.1),2)
    bins = np.arange(1,np.max(data), 1)
    fig = plt.figure(figsize=(10, 4))
    #plt.yscale('log', nonposy='clip')
    ax = plt.axes()
    col = "#7872B1"
    col = "#97acc1"
    col2 = "#261930"
    #colors = np.full(bins.shape, col)
    #colors[-1] = "#C41868"
    ax.hist(data , density = True, bins = bins, #cumulative=-1,
                edgecolor='black', linewidth=0.2, #histtype = 'step',
                color=col)
    #ax.hist(data, bins=bins, density=True, histtype='step', cumulative=True,
           #label='Reversed emp.', color = col2)


    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    #ax.set_yscale('log')
    #ax.set_xlabel('Backscatter [Intensity per sample]', fontsize=8)
    #ax.set_xlabel('Backscatter [Intensity per return]', fontsize=8)
    #ax.set_ylabel('Normalised Likelihood', fontsize=8)
    ax.set_xlabel('FWF samples per Voxel', fontsize=8)
    ax.set_ylabel('Likelihood', fontsize=8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    #ax.set_xticks(np.arange(min(data), max(data)+1, 2.0))
    plt.show()
