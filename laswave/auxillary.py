import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid


def getUnique(block):
    '''function written for a list of numpy arrays with different length
    to extract the unique values between both lists'''
    arr = np.full(1, -9999)
    # getting the return signal with most returns
    for pulse in block:
        pulse = np.sort(pulse)
        arr = arrayMerge(arr,pulse)
        arr = arr[1:] # dropping the first number -9999
        #mr = np.unique(arr, return_counts=True)
        #pos = np.where(mr[1] == max(mr[1]))[0][0] # finds where the maximum position is located
        #mr = mr[0][pos] # most returns
    return(arr)



def arrayMerge(a, b):
    '''Merging two sorted arrays of different shape.'''
    if len(a) < len(b):
        b, a = a, b
    c = np.empty(len(a) + len(b), dtype=a.dtype)
    b_indices = np.arange(len(b)) + np.searchsorted(a, b)
    a_indices = np.ones(len(c), dtype=bool)
    a_indices[b_indices] = False
    c[b_indices] = b
    c[a_indices] = a
    return c



def consecutive(data, stepsize=1):
    '''Returns consecutive elements of an array in form of X arrays'''
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)



def blockApply(data, func, bsize = 3):
    '''
    func: function which normally would take the dataframe
    bsize: block size of data chunks for raw comparission as a power of ten

    build a counter structure to split the sorting.'''
    extra = int(str(len(data))[-bsize:]) # gets the last x digits
    counter_max = int(len(data) - extra)
    counter_end = 10**bsize
    counter_start = 0
    out = [] # create empty list for output
    #
    while (counter_end <= counter_max):
        # grep the block out of the overall data
        block =  data[counter_start:counter_end]
        # apply function
        out.append(func(block))
        counter_start =  counter_end
        counter_end = counter_end + 10**bsize
        #print(str(counter_start) + '   ' + str(counter_end)) # validate the index position
        #
        # poor mans progress bar
        print('block ' + str(int(counter_end / 10**bsize)) + ' of ' + str(int(counter_max / 10**bsize)))
        sys.stdout.write("\033[F") #back to previous line
        sys.stdout.write("\033[K") #clear line
        #
    # calculating the function for remaining entries
    counter_end = counter_end + extra - 10**bsize # removing the last step while adding the remaining values
    block =  data[counter_start:counter_end]
    out.append(func(block))
    return(out)



def itemapply(func, arr):
    '''Tries to apply a function to a every element of a list.'''
    if isinstance(arr, list):
        x = np.zeros(len(arr))
    else:
        x = np.zeros(arr.shape)
    j = 0
    for i in arr:
        try:
            y = func(i)
            x[j] = y
        except:
            x[j] = np.nan
            print('entry ' + str(j) + ' non processable.')
        j += 1
    return(x)



def percentile_extractor(val, perc= 1, neg= False):
    '''returns extractor array for any value array. Either
    for the top or for the bottom percentage.'''
    # top and bottom X percent of the data is considered
    #val = var.copy()
    ## building an extractor to extract the relevant pules from the dataset
    if neg == False:
        extractor = np.where(val <= np.nanpercentile(val, perc))[0]
    elif neg == True:
        extractor = np.where(val >= np.nanpercentile(val, (100 - perc)))[0]
    #
    return(extractor)



## Colormapbuilder for creating your own colormap
# used in mapping special values to R,G,B in a .las

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    #
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    #
    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])
    #
    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        #
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    #
    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    #
    return newcmap



    # def varName(my_var):
    #     '''returns variable name as string.'''
    #     my_var_name = [ k for k,v in locals().iteritems() if v == my_var][0]
    #     return(my_var_name)



    # def pseudoGauss(x, y):
    #     '''Returns a almost accurate gaussian like fit for data that is close
    #     to normal distribution. Input must be two single dimensioned arrays.'''
    #     x0 = np.sum(x*y)/np.sum(y)
    #     s2 = np.sum((x-x0)*(x-x0)*y)/np.sum(y)
    #     return y.max() * np.exp(-0.5*(x-x0)*(x-x0)/s2)
