import sys
import numpy as np


def buildIndexArray(arr, start=0):
    '''Creates an array containing indexing position for an array
    with non uniform entry length.'''
    #idx_arr = arr.copy() # no override is desired, but shape is preseved
    idx_arr = np.full(arr.shape, None)
    for idx, sub_arr in enumerate(arr):
        #noise = noiseExtractor(sub_arr)
        sub_arr_range = np.arange(start, (sub_arr.size + start) ) # builds positional array for later x,y,z extraction
        #idx_arr[idx] = sub_arr_range[sub_arr > noise] # excludes entries below noise threshold
        idx_arr[idx] = sub_arr_range
        #arr[idx] = sub_arr[sub_arr > noise]
        #arr[idx] = sub_arr
    return(idx_arr)



def noiseExtractor(fwf_sig, amount = 10):
    '''Identifies a noise level for a specific array of fullwaveform intensity
    values. Uses the mean value of the x lowest values within the data,
    default is 20. Obviously depending on record length.'''
    #low_vals = np.sort(fwf_sig)[0:amount]
    #noise_level = np.nanmean(low_vals) # here other methods may find good use as well..
    #noise_level = np.percentile(fwf_sig, 50)
    noise_level = (np.min(sig) * noise_threshold).astype(np.int)
    return(np.rint(noise_level).astype(int))



def blockDivide(data, bsize = 3):
    '''bsize: block size of data chunks for raw comparission as a power of ten.
    Returns an array containing the beginning and end indices of individual blocks
    within the data'''
    extra = int(str(len(data))[-bsize:]) # gets the last x digits
    counter_max = int(len(data) - extra)
    counter_end = 10**bsize
    counter_start = 0
    out = np.array(([],[])).transpose() # create empty array for output
    while (counter_end <= counter_max):
        # grep the block out of the overall data
        block =  np.array((counter_start,counter_end))
        out =    np.vstack((out,     block ))
        # set up next step
        counter_start =  counter_end
        counter_end = counter_end + 10**bsize
        #print(str(counter_start) + '   ' + str(counter_end)) # validate the index position
        #
        # poor mans progress bar
        #print('block ' + str(int(counter_end / 10**bsize)) + ' of ' + str(int(counter_max / 10**bsize)))
        #sys.stdout.write("\033[F") #back to previous line
        #sys.stdout.write("\033[K") #clear line
        #
    # calculating the function for remaining entries
    if extra > 0:
        counter_end = counter_end + extra - 10**bsize # removing the last step while adding the remaining values
        block =  np.array((counter_start,counter_end))
        out =    np.vstack((out,     block ))
    return(np.array(out, dtype = np.int))



def calculateCrd(signal, vector, point):
    '''Conversion from signal position (time) to xyz coordinate (space)
    using a directional vector.'''
    coord =np.array(list(map(lambda pulses: ( (pulses*vector)+point) , signal)))
    return(coord)



def buildCrdArr(sig_off_d, vec_dir, pts):
    '''Calculates all x,y,z coordinates from signal position (time) and
    directional vectors.'''
    coords = np.array( ([],[],[]) ).transpose() # initialize array
    coords_map = np.array( ([],[]) ).transpose()
    k= 0
    #
    for i in np.arange(0, pts.shape[0]):
        if sig_off_d[i].size == 0:
            coords =     np.vstack((coords,     np.array( ([],[],[]) ).transpose() ))
            coords_map = np.vstack((coords_map, np.array( [coords.shape[0], 1])))
            #
        else:
            next_coords = calculateCrd(sig_off_d[i], vec_dir[i], pts[i])
            coords =     np.vstack((coords, next_coords))
            coords_map = np.vstack((coords_map, np.array([(coords.shape[0]), next_coords.shape[0]]) ))
    return(coords, coords_map)



def blockBuildCrdArr(sig_off_d, vec_dir, pts, bsize=3):
    '''Calculates all x,y,z coordinates from signal position (time) and directional
    vectors. Tiles the data into power of 10 sized blocks, e.g. bsize=3 means
    blocks have size 1000.'''
    if pts.shape[0] == sig_off_d.shape[0] and pts.shape[0] == vec_dir.shape[0]:
        blocks = blockDivide(pts, bsize=bsize) #
    else:
        return('input does not have same size.')
    #
    coords = np.array( ([],[],[]) ).transpose() # initialize array
    coords_map = np.array( ([],[]) ).transpose()
    #
    for i in blocks:
        block_coords, block_coords_map = buildCrdArr(sig_off_d[i[0]:i[1]], vec_dir[i[0]:i[1]], pts[i[0]:i[1]])
        coords =         np.vstack((coords, block_coords))
        coords_map =     np.vstack((coords_map, block_coords_map))
        # poor mans progress bar
        print('running block ' + str(int(i[0] / 10**bsize)) + ' of ' + str(int(pts.shape[0] / 10**bsize)))
        sys.stdout.write("\033[F") #back to previous line
        sys.stdout.write("\033[K") #clear line
    return(coords, coords_map)
