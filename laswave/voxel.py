####################################################################
########                    VOXEL                           ########
import numpy as np
import laswave.auxillary
from scipy.stats import skew, mode

class voxel:
    def __init__(self, origin, size):
        self.origin = origin
        # X Y Z coordinates of individual voxel
        #self.leftFrontBot = self.origin
        #self.leftFrontTop = self.origin + (np.array([0,1,0]) * size)
        #self.leftBackTop = self.origin + (np.array([0,1,1]) * size)
        #self.rightBackTop = self.origin + (np.array([1,1,1]) * size)
        #self.rightFrontTop = self.origin + (np.array([1,1,0]) * size)
        #self.leftBackBot = self.origin + (np.array([0,0,1]) * size)
        #self.rightBackBot = self.origin + (np.array([1,0,1]) * size)
        #self.rightFrontBot = self.origin + (np.array([1,0,0]) * size)
        self.size = size
        #self.center = origin + np.array([0.5 * size, 0.5 * size, 0.5 * size])
        self.content = None


    def __delete__(self, instance):
        # print ("deleted voxel" + str(self.origin))
        del self.origin
        del self.size
        del self.content


    def getVolume(self):
        return(self.size ** 3)


    def getCenter(self):
        # center = self.origin + np.array([0.5 * self.size,
        #                                 0.5 * self.size,
        #                                 0.5 * self.size] )
        # return(center.reshape((1,3)))
        center = self.origin + 0.5 * self.size
        return(center)


    def calculateNeighbours(self, dist = 1):
        '''defines all immediate neighbours coords of the voxel'''
        # denomination:
        # dimension 0: Left (LL), LeftRight (LR), Right (RR)
        # dimension 1: Under (UU), UnderOver (UO), Over (OO)
        # dimension 2: Front (FF), FrontBack (FB), Back (BB)
        self.LBUUFF = self.origin + [-1, -1, -1] * self.size #* dist
        self.LBUUFB = self.origin + [-1, -1, 0] * self.size #* dist
        self.LBUUBB = self.origin + [-1, -1, 1] * self.size #* dist
        self.LBUOFF = self.origin + [-1, 0, -1] * self.size #* dist
        self.LBUOFB = self.origin + [-1, 0, 0] * self.size #* dist
        self.LBUOBB = self.origin + [-1, 0, 1] * self.size #* dist
        self.LBOOFF = self.origin + [-1, 1, -1] * self.size #* dist
        self.LBOOFB = self.origin + [-1, 1, 0] * self.size #* dist
        self.LBOOBB = self.origin + [-1, 1, 1] * self.size #* dist
        self.LRUUFF = self.origin + [0, -1, -1] * self.size #* dist
        self.LRUUFB = self.origin + [0, -1, 0] * self.size #* dist
        self.LRUUBB = self.origin + [0, -1, 1] * self.size #* dist
        self.LRUOFF = self.origin + [0, 0, -1] * self.size #* dist
        #self.LRUOFB = self.origin + [0, 0, 0] * self.size #* dist # THIS IS ORIGIN
        self.LRUOBB = self.origin + [0, 0, 1] * self.size #* dist
        self.LROOFF = self.origin + [0, 1, -1] * self.size #* dist
        self.LROOFB = self.origin + [0, 1, 0] * self.size #* dist
        self.LROOBB = self.origin + [0, 1, 1] * self.size #* dist
        self.RRUUFF = self.origin + [1, -1, -1] * self.size #* dist
        self.RRUUFB = self.origin + [1, -1, 0] * self.size #* dist
        self.RRUUBB = self.origin + [1, -1, 1] * self.size #* dist
        self.RRUOFF = self.origin + [1, 0, -1] * self.size #* dist
        self.RRUOFB = self.origin + [1, 0, 0] * self.size #* dist
        self.RRUOBB = self.origin + [1, 0, 1] * self.size #* dist
        self.RROOFF = self.origin + [1, 1, -1] * self.size #* dist
        self.RROOFB = self.origin + [1, 1, 0] * self.size #* dist
        self.RROOBB = self.origin + [1, 1, 1] * self.size #* dist
        # validate that  all neighbours are within the voxelspaces bounds


    def setOrigin(self, xyz):
        '''used to point towards other pixels.'''
        self.origin = xyz


    def getOrigin(self):
        return(self.origin)


    def setSize(self, size):
        self.size = size


    def getSize(self):
        return(self.size)


    def initContent(self):
        self.content = voxelContent()
        self.content.origin = self.origin # this might be dodgy programmin

    def getContent(self):
        return(self.content)


    def setContent(self, content):
        self.content = content



    # def meanEuclideanDist(self):
    #     '''Returns the mean euclidean distance of all point to the center of
    #     the voxel. Not direct vector but absolute distance.'''
    #     coords = self.content.getCoords()
    #     dist = coords - self.getCenter()
    #     mean_dist = np.mean(dist, axis = 0)
    #     return(mean_dist)



    def extractMeanEuclideanVector(self, expos_type = None, weights = 'intensity'):
        '''Returns the mean euclidean vector, weighted by intensity or different
        value.'''
        if expos_type is not None:
            mask = extractMaskEuclideanExposition( expos_type = expos_type)
        else:
            mask = np.full(( len(self.content.dim_dist,), ), True )
        vec = self.content.dist[mask]
        if weights is 'intensity':
            weights = self.content.int / np.max(self.content.int)
        else:
            weights = None
        return(np.average(self.content.dim_dist, axis=0, weights = weights) )



    def buildEuclideanExposition(self, elliptoid_factor = 0.5):
        '''Parent function to build and extract euclidean exposition.'''
        self.content.dist, self.content.threshold_dist, self.content.dim_dist, self.content.threshold_dist_dim = self.euclideanEllipsoid(elliptoid_factor)
        self.content.expos_arr = self.euclideanEllipsoidExposition(self.content.dist, self.content.threshold_dist, self.content.dim_dist, self.content.threshold_dist_dim)



    # def euclideanEllipsoid(self, elliptoid_factor = 0.5):
    #     '''Calculates an epplisoid inside the voxel that fill out a voxel volume
    #     corresponding to the elliptoid_factor. Thus a factor of 0.5 represents
    #     an ellipsoid that fills out half the voxels volume.'''
    #     zz = self.getCenter() # voxelCenter, center of mass for ellipsoid
    #     dim_dist = self.content.coords - zz # distance from origin
    #     alpha = np.arctan( self.size[0] / self.size[1] ) # angle between x and y axis
    #     beta =  np.arctan( self.size[0] / self.size[2] ) # angle between x and z axis
    #     func = lambda x,y,z :   (np.sin(alpha) * x, \
    #                             np.cos(alpha) * y, \
    #                             np.sin(beta) * z)
    #     # 0.5 * self.size is the distance between zz and voxel edge
    #     threshold = func(self.size[0]* 0.5 * elliptoid_factor, \
    #                      self.size[1]* 0.5 * elliptoid_factor, \
    #                      self.size[2]* 0.5 * elliptoid_factor)
    #     # eucl_thresh = np.sum(threshold)
    #     eucl_dist = map( func, dim_dist[:,0], dim_dist[:,1], dim_dist[:,2] )
    #     eucl_dist = np.vstack(list(eucl_dist))
    #     # returns individual euclidean distance, as well as the euclidean threshold
    #     # both returned arrays are 3 dimensional
    #     return(eucl_dist, threshold)



    def euclideanEllipsoid(self, elliptoid_factor = 0.5):
        '''Calculates an epplisoid inside the voxel that fill out a voxel volume
        corresponding to the elliptoid_factor. Thus a factor of 0.5 represents
        an ellipsoid that fills out half the voxels volume.'''
        # zz gets minimal offset so no devide be zero occurs
        zz = self.getCenter() + np.array([0.001,0.001,0.001]) # voxelCenter, center of mass for ellipsoid
        dim_dist = self.content.coords - zz # elliptoid distance from origin for vector angle
        ellip_func = lambda x,y,z :   (np.cos( np.arctan(y/x) ) * self.size[0] * elliptoid_factor, \
                                       np.sin( np.arctan(y/x) ) * self.size[1] * elliptoid_factor, \
                                       np.cos( np.arctan(z/y) ) * self.size[2] * elliptoid_factor)
        threshold_dist_dim = map(ellip_func, dim_dist[:,0], dim_dist[:,1], dim_dist[:,2] )
        threshold_dist_dim = np.vstack(list(threshold_dist_dim))
        threshold_dist = np.sqrt( np.sum( (threshold_dist_dim * threshold_dist_dim), axis = 1) )
        #
        #dist_func = lambda x,y,z :   (np.cos( np.arctan(y/x) ) * x , \
        #                               np.sin( np.arctan(y/x) ) * y , \
        #                               np.cos( np.arctan(z/y) ) * z )
        #eucl_dist_dim = map( dist_func, dim_dist[:,0], dim_dist[:,1], dim_dist[:,2] )
        #eucl_dist_dim =  np.vstack(list(eucl_dist_dim))
        # np.sqrt( np.sum(eucl_dist_dim * eucl_dist_dim, axis = 1) )
        # np.sum(np.abs(eucl_dist_dim), axis = 1)
        dist_func = lambda x : np.sqrt(np.dot(x,x))
        dist = map( dist_func, dim_dist )
        dist = np.array(list(  dist  ))
        # returns individual euclidean distance, as well as the euclidean threshold
        # both returned arrays are 3 dimensional
        return(dist, threshold_dist, dim_dist, threshold_dist_dim)


    def euclideanEllipsoidExposition(self, dist, threshold_dist, dim_dist, threshold_dist_dim):
        '''Calculates mask representing which coordinate points lie within
        or outside the ellipsoid.'''
        # THIS CAN BE OPTIMISED FOR ONLY USING THE SUMS, DIMENSIONALITY HERE
        # IS MISPLACED AS THRESHOLD ONLY REPRESENTS THE VECTOR TOWARDS THE VOXEL CORNERS
        #dist_from_thresh = np.abs(dist) - np.abs(threshold)
        #is_inside = np.sum(dist_from_thresh, axis = 1) < 0
        #is_outside = np.sum(dist_from_thresh, axis = 1) > 0
        # NEW IMPLEMENTATION
        dist_from_thresh =  dist - threshold_dist
        is_inside = dist_from_thresh <= 0
        is_outside = dist_from_thresh > 0
        #
        # TRUE values represent steps on the axis into the positive direction (e.g. x =1, y=2.4)
        # FALSE values represent steps into the negativ direction (e.g z = -0.5) all referencing the center
        vec_dir = dim_dist > 0
        # building expositional array filled only with 'Z'
        # try statements in order to avoid problems if no or all values lie outside of ellipsoid
        expos_arr = np.full(vec_dir.shape, 8) #'Z')
        try: # X-dimension
            expos_arr[ is_outside * (vec_dir[:,0] == True), 0 ] = 1  # 'R' # (R)ight
        except:
            pass
        try:
            expos_arr[ is_outside * (vec_dir[:,0] == False), 0 ] = 0  #'L' # (L)eft
        except:
            pass
        try: # Y-dimension
            expos_arr[ is_outside * (vec_dir[:,1] == True), 1 ] = 2  #'B' # (B)ack
        except:
            pass
        try:
            expos_arr[ is_outside * (vec_dir[:,1] == False), 1 ] = 0  #'F' # (F)ront
        except:
            pass
        try: # Z-dimension
            expos_arr[ is_outside * (vec_dir[:,2] == True), 2 ] = 4 # 'U' # (U)p
        except:
            pass
        try:
            expos_arr[ is_outside * (vec_dir[:,2] == False), 2 ] = 0 # 'D' # (D)own
        except:
            pass
        #self.content.expos_arr = expos_arr
        return(expos_arr)



    # def euclideanEllipsoidExposition(self, eucl_dist, threshold):
    #     '''Calculates mask representing which coordinate points lie within
    #     or outside the ellipsoid.'''
    #     # THIS CAN BE OPTIMISED FOR ONLY USING THE SUMS, DIMENSIONALITY HERE
    #     # IS MISPLACED AS THRESHOLD ONLY REPRESENTS THE VECTOR TOWARDS THE VOXEL CORNERS
    #     #dist_from_thresh = np.abs(eucl_dist) - np.abs(threshold)
    #     #is_inside = np.sum(dist_from_thresh, axis = 1) < 0
    #     #is_outside = np.sum(dist_from_thresh, axis = 1) > 0
    #     # NEW IMPLEMENTATION
    #     dist_from_thresh =  np.sum( eucl_dist, axis =1 )
    #     is_inside = dist_from_thresh < 0
    #     is_outside = dist_from_thresh > 0
    #     #
    #     # TRUE values represent steps on the axis into the positive direction (e.g. x =1, y=2.4)
    #     # FALSE values represent steps into the negativ direction (e.g z = -0.5) all referencing the center
    #     vec_dir = eucl_dist > 0
    #     # building expositional array filled only with 'Z'
    #     # try statements in order to avoid problems if no or all values lie outside of ellipsoid
    #     expos_arr = np.full(vec_dir.shape, 8) #'Z')
    #     try: # X-dimension
    #         expos_arr[ is_outside * (vec_dir[:,0] == True), 0 ] = 1  # 'R' # (R)ight
    #     except:
    #         pass
    #     try:
    #         expos_arr[ is_outside * (vec_dir[:,0] == False), 0 ] = 0  #'L' # (L)eft
    #     except:
    #         pass
    #     try: # Y-dimension
    #         expos_arr[ is_outside * (vec_dir[:,1] == True), 1 ] = 2  #'B' # (B)ack
    #     except:
    #         pass
    #     try:
    #         expos_arr[ is_outside * (vec_dir[:,1] == False), 1 ] = 0  #'F' # (F)ront
    #     except:
    #         pass
    #     try: # Z-dimension
    #         expos_arr[ is_outside * (vec_dir[:,2] == True), 2 ] = 4 # 'U' # (U)p
    #     except:
    #         pass
    #     try:
    #         expos_arr[ is_outside * (vec_dir[:,2] == False), 2 ] = 0 # 'D' # (D)own
    #     except:
    #         pass
    #     #self.content.expos_arr = expos_arr
    #     return(expos_arr)



    exposition_dict = { # valuepairs: identifier, name, axis, las_classification
            'ZZZ':  ['voxelCenter',    24,   None,  64],
            'LFD':  ['leftFrontDown',   0,   None,  65],
            'LFU':  ['leftFrontUp',     4,   None,  66],
            'LBD':  ['leftBackDown',    2,   None,  67],
            'LBU':  ['leftBackUp',      6,   None,  68],
            'RFD':  ['rightFrontDown',  1,   None,  69],
            'RFU':  ['rightFrontUp',    5,   None,  70],
            'RBD':  ['rightBackDown',   3,   None,  71],
            'RBU':  ['rightBackUp',     7,   None,  72],
            'R':    ['right',           1,   0,     73],
            'L':    ['left',            0,   0,     74],
            'F':    ['front',           0,   1,     75],
            'B':    ['back',            2,   1,     76],
            'U':    ['up',              4,   2,     77],
            'D':    ['down',            0,   2,     78],
            'Z':    ['center',          8,   None,  79]
    }



    def extractMaskEuclideanExposition(self , expos_type,
                                            expos_dict = exposition_dict):
        '''extracts a set of points based on pre determined euclidean
        exposition.'''
        mask = np.full( (len(self.content.coords), ) , True )
        for type in expos_type:
            id = expos_dict[type][1]
            col = expos_dict[type][2]
            if col is None:
                mask *= ( np.sum(self.content.expos_arr, axis = 1) == id )
            else:
                mask *= ( self.content.expos_arr[:,col] == id  )
        return(mask)



    def extractColorEuclideanExposition(self):
        '''extracts an integer array representing the colors/ classification id
        for each point in voxel.voxelContent.coords '''
        return( np.sum( self.content.expos_arr, axis = 1 ))



    def extractModeEuclideanExposition(self):
        '''Extracts the most common exposition for data points within voxel.'''
        id = np.sum(self.content.expos_arr, axis = 1)
        return(mode(id)[0][0])


    def extractCountEuclideanExposition(self, expos_dict = exposition_dict):
        '''Extracts the count for all expositions'''
        id = np.sum(self.content.expos_arr, axis = 1)
        count = np.bincount(id)
        # mode_val = np.where(count == np.max(count))[0][0]
        return(count)


    # def intMean(self):
    #     '''Intensity mean'''
    #     if self.content.int is not None:
    #         return(np.nanmean(z_coords))
    #     else:
    #         return(None)
    #
    #
    # def heightMean(self):
    #     '''Height mean'''
    #     if self.content.coords is not None:
    #         z_coords = self.content.coords[:,2] - self.origin[2]
    #         return(np.nanmean(z_coords))
    #     else:
    #         return(None)
    #
    #
    # def intMode(self):
    #     '''Intensity mode'''
    #     if self.content.int is not None:
    #
    #     else:
    #         return(None)
    #
    #
    # def heightMode(self):
    #     '''Height mode'''
    #     if self.content.coords is not None:
    #         z_coords = self.content.coords[:,2] - self.origin[2]
    #     else:
    #         return(None)
    #
    #
    # def intVariance(self):
    #     '''Intensity variance'''
    #     if self.content.int is not None:
    #         return(np.var(self.int))
    #     else:
    #         return(None)
    #
    #
    # def heightVariance(self):
    #     '''Height variance'''
    #     if self.content.coords is not None:
    #         z_coords = self.content.coords[:,2] - self.origin[2]
    #         return(np.var(z_coords))
    #     else:
    #         return(None)
    #
    #
    # def intSkew(self, content = 'intensity'):
    #     '''Intensity skew'''
    #     if self.content.int is not None:
    #         return(skew(self.content.int))
    #     else:
    #         return(None)
    #
    #
    # def heightSkew(self):
    #     '''Height skew'''
    #     if self.content.coords is not None:
    #         return(skew(self.content.coords[:,2]))
    #     else:
    #         return(None)
    #
    #
    # def intensityDistFit(self):
    #     '''Fits a pseudo gaussian to all intenisty values in voxel.'''
    #     if self.content.coords is None or self.content.int is None:
    #         return(None)
    #     z_coords = self.content.coords[:,2] - self.origin[2] # normalize for origin height
    #     arr = np.column_stack((z_coords,self.content.int))
    #     arr = arr[arr[:,0].argsort()] # sorts array by the height
    #     return(pseudoGauss(arr[:,0],arr[:,1]))
    #
    #
    # def heightDistFit(self):
    #     '''Fits a pseudo gaussian to normalized height values in voxel.'''
    #     if self.content.coords is None:
    #         return(None)
    #     z_coords = self.content.coords[:,2] - self.origin[2]
    #     arr = np.column_stack((np.arange(0,len(z_coords)), z_coords))
    #     return(pseudoGauss(arr[:,0],arr[:,1]))

    def coordHist2d(self, dim1, dim2, bin_factor = 10):
        '''Builds a 2D histogram for the provided dimensions (axis). Bins are
        calculated through voxel.size/bin_factor. Results can be visualized through
        plt.pcolormesh .'''
        x = self.content.coords[:,dim1]
        y = self.content.coords[:,dim2]
        #
        x_min = self.origin[dim1]
        x_max = self.origin[dim1] + self.size[dim1]
        #
        y_min = self.origin[dim2]
        y_max = self.origin[dim2] + self.size[dim2]
        #
        x_step = self.size[dim1]/ bin_factor
        y_step = self.size[dim2]/ bin_factor
        #
        x_bin = np.arange(x_min, x_max + x_step , x_step )
        y_bin = np.arange(y_min, y_max + y_step , y_step )
        #
        H, xedges, yedges = np.histogram2d(x, y, bins= np.array([x_bin,y_bin]))
        X, Y = np.meshgrid(xedges, yedges)
        return(H, X, Y)
        #plt.pcolormesh(X, Y, H, cmap = 'PuBu')
        #plt.show()


    def coordHist(self, dim, bin_factor = 10):
        '''Builds a Histogram for the distribution of coordinates in the specified
        dimension. Bins are calculated through voxel.size/bin_factor.'''
        x_min = self.origin[dim]
        x_max = self.origin[dim] + self.size[dim]
        x_step = self.size[dim] / bin_factor
        x_bin = np.arange(x_min, x_max + x_step , x_step )
        H, xedges = np.histogram(self.content.coords[:,dim], bins = x_bin)
        return(H, xedges)


    def intHist(self, step_size = 50):
        '''Builds a Histogram for the distribution of the intensity values.
        Values are binned on a fixed scale from 200 to >1500 with steps specified
        though step_size.'''
        bins = np.arange(200,1500, step_size)
        bins = np.append(bins, 3000) # do account for high values
        H, xedges = np.histogram(self.content.int, bins = bins)
        return(H, xedges)



####################################################################
########                    VOXELCONTENT                    ########

class voxelContent():
    '''Class containing all data within a voxel.'''
    def __init__(self):
        self.coords = None
        self.coord_count = 0
        self.int = None
        self.int_count = 0


    def addCoords(self, coord, count = 1):
        if self.coords is None: # no coords exist yet
            #self.coords = [coord]
            self.coords = coord #np.array([coord])
        #elif self.coords is not None: # if coords already exists
            #self.coords.append(coord)
        else:
            #self.coords = np.concatenate((self.coords, [coord]), axis=1)
            self.coords = np.vstack((self.coords, coord))
        self.coord_count += count


    def addIntensity(self, intensity, count = 1):
        if self.int is None:
            self.int = [intensity]
        elif self.int is not None:
            self.int.append(intensity)
        self.int_count += count


    def rmvCoords(self, coord, all = False):
        if self.coords is None:
            pass
        # Removes all entries of coord
        if all is True :
            while (self.coords.count(coord)):
                    self.coords.remove(coord)
                    self.coord_count -= 1
        # Removes only first instance of coord
        elif all is False :
            if self.coords.count(coord) > 0: # checks for instance of these entries
                self.coords.remove(coord)
                self.coord_count -= 1
        # Makes sure to set coords to None if none are left in the list
        if len(self.coords) == 0:
            self.coords = None


    def rmvIntensity(self):
        if self.int is None:
            pass
        # Removes all entries of int
        if all is True :
            while (self.int.count(int)):
                    self.int.remove(int)
                    self.int_count -= 1
        # Removes only first instance of int
        elif all is False :
            if self.int.count(int) > 0: # checks for instance of these entries
                self.int.remove(int)
                self.int_count -= 1
        # Makes sure to set int to None if none are left in the list
        if len(self.int) == 0:
            self.int = None


    def getCoords(self):
        return(self.coords)


    def setCoords(self, coords):
        self.coords = coords


    def getCoordCount(self):
        return(self.coord_count)


    def setCoordCount(self, count):
        self.coord_count = count


    def getIntensity(self):
        return(self.int)


    def setIntensity(self, intensity):
        self.int = intensity


    def getIntensityCount(self):
        return(self.int_count)


    def setIntensityCount(self, count):
        self.int_count = count


    def setSource(self, id):
        self.source = id


    def getSource(self):
        if self.source is None:
            return(None)
        else:
            return(self.source)


    def intSum(self):
        if self.int is not None:
            return(np.sum(self.int))
        else:
            pass


    def intMean(self):
        '''Intensity mean'''
        if self.int is not None:
            return( int( ( np.sum(self.int) / self.int_count) ))
        else:
            return(None)


    def heightMean(self):
        '''Height mean'''
        if self.coords is not None:
            z_coords = self.coords[:,2] - self.origin[2]
            return(np.nanmean(z_coords))
        else:
            return(None)


    def intMode(self):
        '''Intensity mode'''
        if self.int is not None:
            return(mode(self.int)[0])
        else:
            return(None)


    def heightMode(self):
        '''Height mode'''
        if self.coords is not None:
            z_coords = self.coords[:,2] - self.origin[2]
            return(mode(z_coords))
        else:
            return(None)


    def intVar(self):
        '''Intensity variance'''
        if self.int is not None:
            return(np.var(self.int))
        else:
            return(None)


    def heightVar(self):
        '''Height variance'''
        if self.coords is not None:
            z_coords = self.coords[:,2] - self.origin[2]
            return(np.var(z_coords))
        else:
            return(None)


    def intSkew(self, content = 'intensity'):
        '''Intensity skew'''
        if self.int is not None:
            return(skew(self.int))
        else:
            return(None)


    def heightSkew(self):
        '''Height skew'''
        if self.coords is not None:
            return(skew(self.coords[:,2]))
        else:
            return(None)


    def coordStd(self):
        '''standart deviation for the coords in voxel'''
        if self.coords is not None:
            return(np.std(self.coords, axis = 0))
        else:
            return(None)

## THESE NEED TO BE IMPLEMENTED INTO ANOTHER FUNCTIO

    def zAxisHist(self):
        '''computes a histogram for the height values (z-axis) weighted by
        return intensity. Returns an array with two sub-arrays, the first
        containing the approximated count per bin and the second containing
        the bin individual edges. Bins are tipically not evenly spaced here.'''
        hist = np.histogram(self.coords[:,2], weights = self.int/np.nanmax(self.int))
        return(hist)


    def xAxisHist(self):
        return(np.histogram(self.coords[:,0], weights = self.int/np.nanmax(self.int)))


    def yAxisHist(self):
        return(np.histogram(self.coords[:,1], weights = self.int/np.nanmax(self.int)))



    def intDistFit(self):
        '''Fits a pseudo gaussian to all intenisty values in voxel.'''
        if self.coords is None or self.int is None:
            return(None)
        z_coords = self.coords[:,2] - self.origin[2] # normalize for origin height
        arr = np.column_stack((z_coords,self.int))
        arr = arr[arr[:,0].argsort()] # sorts array by the height
        return(pseudoGauss(arr[:,0],arr[:,1]))



    def heightDistFit(self):
        '''Fits a pseudo gaussian to normalized height values in voxel.'''
        if self.coords is None:
            return(None)
        z_coords = self.coords[:,2] - self.origin[2]
        arr = np.column_stack((np.arange(0,len(z_coords)), z_coords))
        return(pseudoGauss(arr[:,0],arr[:,1]))



    def fwhmInt(self):
        '''Calculates fwhm based on the window function fit.'''
        fit = self.indDistFit()
        fit_half_max = np.nanmax(fit) / 2



def pseudoGauss(x, y):
    '''Returns a almost accurate gaussian like fit for data that is close
    to normal distribution (window function).
    Input must be two single dimensioned arrays.'''
    # first normalising the values seem to bring better results
    # voxelContent.int - np.min(voxelConten.int) # dont know if this one is necessary
    # voxelContent.coords[:,2] - np.min(voxelContent.coords[:,2])
    x0 = np.sum(x*y)/np.sum(y) # mean of variable x normalised for y
    s2 = np.sum((x-x0)*(x-x0)*y)/np.sum(y) # mean squared distances of x from x0 normalised for y
    return y.max() * np.exp(-0.5*(x-x0)*(x-x0)/s2) # why the -0.5 ?


def interQuartileRange(data, axis):
    '''Returns interquartile range for given data set'''
    q1_x = np.percentile(data, 25, interpolation='midpoint', axis = axis)
    q3_x = np.percentile(data, 75, interpolation='midpoint', axis = axis)
    return(q3_x - q1_x)
