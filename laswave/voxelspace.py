## VOXELSPACE
import numpy as np
import laspy
#from voxel import *
import laswave.voxel
#from laswave_core_georef import *
import laswave.core_georef
import uuid
from scipy.stats import mode
#import copy
import logging
import matplotlib.cm as cm
import sys
from structure_tensor import eig_special_3d, structure_tensor_3d


# DICTIONARY FOR DIFFERENT VALUE TYPES THAT CAN BE EXTRACTED
data_type = {       # valuepairs: type, axis, func, dimension, aggregation
        'center':              ['voxelAttr',    0   , voxel.getCenter, 3, np.nanmean],
        'size':                ['voxelAttr',    0   , voxel.getSize, 3, np.nanmean], #should return the actual size since all voxel share the same size on levels
        'volume':              ['voxelAttr',    None, voxel.getVolume, 1, np.nanmean],
        'eucl_vec_mean':       ['voxelAttr',    0,    voxel.extractMeanEuclideanVector, 3, np.nanmean],
        'eucl_exp_mode':       ['voxelAttr',    None, voxel.extractModeEuclideanExposition, 1, mode],
        'coords':              ['voxelCont',    0   , voxelContent.getCoords, 3, np.nanmean],
        'coords_std_tile':     ['voxelCont',    0   , voxelContent.getCoords, 3, np.std],  # approach between coords_std and intensity_var is different
        # coords_std is returning std for tile/voxel level, whereas intensity_var is always calculated on the lowest level
        'coords_std_vox':      ['voxelCont',    0   , voxelContent.coordStd, 3, np.mean],
        'coords_iqr_tile':     ['voxelCont',    0   , voxelContent.getCoords, 3, interQuartileRange],
        'coord_count':         ['voxelCont',    None, voxelContent.getCoordCount, 1, np.nansum],
        'intensity':           ['voxelCont',    None, voxelContent.getIntensity, 1, np.nanmean],
        'intensity_sum':       ['voxelCont',    None, voxelContent.intSum, 1, np.nansum],
        'intensity_mean':      ['voxelCont',    None, voxelContent.intMean, 1, np.nanmean],
        'intensity_mode':      ['voxelCont',    None, voxelContent.intMode, 1, np.nanmean],
        'intensity_var':       ['voxelCont',    None, voxelContent.intVar, 1, np.nanmean],
        'intensity_std':       ['voxelCont',    None, voxelContent.getIntensity, 1, np.std],
        'intensity_iqr':       ['voxelCont',    None, voxelContent.getIntensity, 1, interQuartileRange],
        'intensity_skew':      ['voxelCont',    None, voxelContent.intSkew, 1, np.nanmean],
        'intensity_fit_gauss': ['voxelCont',    None, voxelContent.intDistFit, 1, np.nanmean],
        #'tensor_linear':       ['voxelCont',    None, voxelContent.tensorLinear, 3, np.nanmean],
        'mask':                ['voxel',        None, None,             1, None],
        'grid':                ['voxel',        0,    None,             3, None],
        'voxel':               ['voxel',        None, None,             1, None]
}



####################################################################
########                    VOXELSPACE                      ########

class voxelSpace:
    def __init__(self, width, depth, height, vsize, coordOrg = np.array([0,0,0]),
                init_voxel = True, space_type = 'voxelSpace' ):
        self.logger = logging.getLogger('__main__') # gets logger instance from parent script
        self.logger.setLevel(logging.INFO)
        #self.type = 'voxelSpace'
        self.width = width
        self.depth = depth
        self.height = height
        self.size = np.array([width, depth, height])
        self.vsize = vsize # voxelsize
        if '.' in str(vsize): # decimals to round data too
            self.precision = len(str(np.min(vsize)).split('.')[-1]) + 1
        else:
            self.precision = 2
        self.vspace = None
        self.coordOrg = coordOrg
        self.grid = voxelSpace.gridFromExtent(self)
        self.mask = voxelSpace.maskFromExtent(self)
        if init_voxel == True:
            self.x_id, self.y_id, self.z_id = voxelSpace.createScaleFactors(self)
            self.id = voxelSpace.createIdentifier(self)
            self.vspace = voxelSpace.vspaceFromGrid(self)
        elif init_voxel == 'empty':
            self.vspace = np.full( len(self.grid), object )
        elif init_voxel == False:
            pass # here to indicate that this is also a valid value
        self.las_path = None
        self.las_file = None
        self.data_size = None
        self.data_entry_pos = None
        if type(self) is voxelSpace:
            self.logger.info('created voxelSpace object with')
            self.logger.info('  width: %d', self.width)
            self.logger.info('  depth: %d', self.depth)
            self.logger.info('  height: %d', self.height)
        self.level = 0
        # self.initProgressTracker()
    # def logDebug(self, message):
    # def logInfo(self, message):
    # def logWarning(self, message):
    # def logError(self, message):
    # def logCritical(self, message):


    def getOrigin(self):
        '''return the lower left corner (origin) of the voxelSpace or
        voxelSpaceTile.'''
        return(self.coordOrg)



    def createFilePath(self, dir=None, value_name=None, format_name=None):
        '''Namestring generator for writing out files.'''
        # origin = str(self.getOrigin())
        # origin = origin.replace('[', '')
        # origin = origin.replace(']', '')
        # origin = origin.replace(' ', '_')
        x_org = str(self.getOrigin()[0])
        y_org = str(self.getOrigin()[1])
        z_org = str(self.getOrigin()[2])
        origin = x_org + '_' + y_org + '_' + z_org
        origin = origin.replace('.', '-')
        if isinstance(self, voxelSpace):
            inst_name = 'tile'
        elif isinstance(self, voxelSpaceTile):
            inst_name = 'space'
        if value_name == None:
            file_path = dir + '/' + inst_name + '_org_' + origin + '.' + format_name
        else:
            file_path = dir + '/' + value_name + '_' + inst_name + '_org_' + origin + '.' + format_name
        self.logger.info('out path: %s', file_path)
        return(file_path)



    def createScaleFactors(self, return_pos = False):
        '''Creates unique identifier factors for each dimension (x,y,z) in
        order to build unique identifier in createIdentifier().'''
        dim = np.copy(self.size)
        dim_sort = np.sort(self.size)
        dim_scaled = np.ones(3) # array for scaling factors
        #dim_scaled[0] = 1                        # scale factor 0 eqv: x = 1
        dim_scaled[1] = dim_sort[1]               # scale factor 1 eqv: x = x
        dim_scaled[2] = dim_sort[1] * dim_sort[2] # scale factor 2 eqv: x = x*y
        for count, d in enumerate(dim):
            pos = np.where(dim_sort == d)[0][0]
            dim[count] = dim_scaled[pos]
        # if return_pos = True:
        #     min_dim = np.where(dim == np.min(dim))[0][0]
        #     max_dim = np.where(dim == np.max(dim))[0][0]
        #     mean_dim = np.where( (dim < np.max(dim)) & (dim > np.min(dim)) )[0][0]
        #     return(min_dim, mean_dim, max_dim)
        return(dim[0], dim[1], dim[2])



    def createIdentifier(self):
        '''Creates a unique spatial identifier to place voxel instance housed
        in voxelSpace.vspace.'''
        a = self.grid[:,0] * self.x_id
        b = self.grid[:,1] * self.y_id
        c = self.grid[:,2] * self.z_id
        #id = np.prod((a,b,c), axis=0)
        id = np.sum((a,b,c), axis=0)
        return(id)



    def gridFromExtent(self, size = None, vsize = None):
        '''Creates a grid from x,y,z size of the voxelSpace (default) or a
        specified size. Step size, which is synonimous with voxel size can
        be specified through vsize flag.'''
        if size is None:
            size = self.size
        if vsize is None:
            vsize = self.vsize
        X,Y,Z = np.mgrid[   self.coordOrg[0]:(self.coordOrg[0]+size[0]):vsize[0],
                            self.coordOrg[1]:(self.coordOrg[1]+size[1]):vsize[1],
                            self.coordOrg[2]:(self.coordOrg[2]+size[2]):vsize[2]  ]
        X = np.round(X, self.precision)
        Y = np.round(Y, self.precision)
        Z = np.round(Z, self.precision)
        grid = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        if type(self) is voxelSpace:
            self.logger.debug('grid created. Size: %d for H: %d  W: %d  D: %d', len(grid), size[0], size[1], size[2] )
        return(grid)



    def vspaceFromGrid(self):
        '''Creates an array of voxel instances according to the number of
        elements in voxelSpace.grid '''
        space = []
        for i in self.grid:
            vox = voxel(i, self.vsize)
            space.append(vox)
        if type(self) is voxelSpace:
            self.logger.info('%d voxel instances created', len(space))
        return(np.array(space))



    def maskFromExtent(self, init_val = 0, grid ='voxel'):
        '''Mask to flag voxel instances that contain data. Value
        of 0 indicates empty voxel at position (default value if not
        specified in init_val).'''
        if grid is 'voxel':
            grid = self.grid
        if grid is 'tile':
            grid = self.tile_grid
        data_size = len(grid)
        if init_val == 1:
            mask = np.ones((data_size,), dtype=int)
        elif init_val == 0:
            mask = np.zeros((data_size,), dtype=int)
        if type(self) is voxelSpace:
            self.logger.debug('build mask, initialised with %d', init_val)
        return(mask)



    def maskEntry(self, pos, grid='voxel'):
        '''Changing the value of voxelSpace.vspace entry at position (pos)
        to its opposing value. Mask entries can represent either
        'contains data' (val = 1) or 'contains no data' (val = 0).'''
        if grid == 'voxel':
            val = abs( self.mask[pos] -1 )
            self.mask[pos] = val
        if grid == 'tile':
            val = abs( self.tile_mask[pos] -1 )
            self.tile_mask[pos] = val



    def getPositionFromCoords(self, xyz):
        '''Gets the position within voxelSpace.vspace based on a x,y,z position.
        Utilises the the fact that voxelSpace.grid and voxelSpace.vspace are
        build 'parallel' to each other and thus a position in grid corresponds
        to a position in vspace. Returns position of voxel instance.'''
        xyz = xyz - ( (xyz - self.coordOrg) % self.vsize )
        #xyz = np.round(xyz, self.precision) # TEST IF NECESSARY
        # subtract here to account for the offset of the coordOrg
        # offset discribes the starting position at which grid position are
        # beginning to be counted
        # e.g. coordOrg = [1,3,7], vsize = 0.3
        # then first position in the grid [1.3, 3.3, 7.3]
        # substracting normalizes this on onto [0,0,0] origin
        # this is necessary for the % operator to work properly
        try:
            vox_pos = np.where(np.all(self.grid == xyz, axis = 1))[0][0]
            return(vox_pos)
        except:
            # print('couldnt find grid position for: ' + str(xyz))
            return(None)



    def getVoxel(self, xyz, vox_pos = False):
        '''Return voxel instance corresponding to XYZ position or the position
        provided through vox_pos. Provided position must correspond to the
        position of that voxel instance in voxelSpace.vspace .'''
        # in case no voxel position could be determined in stack beforehand
        # but still is getting parsed
        if vox_pos == None:
            return
        if vox_pos == False:
            vox_pos = self.getPositionFromCoords(xyz)
        try:
            return(self.vspace[vox_pos]) # returns voxel instance
        except:
            print('couldnt find xyz Voxel: ' + str(xyz))



    # def getVoxelFromPositionId(self, pos):
    #     '''Gets the position within the ID array based on XYZ position.
    #     Utilises the unique identifiert number generated and stored
    #     in self.id. This is more robust as it is independent from
    #     voxelSpace.grid and voxelSpace.vspace having identical positions
    #     for voxel instances. Computes significally longer for large data.'''
    #     pos = np.trunc(pos)
    #     # further 'truncation' down to the actual voxel coordinate
    #     # somewhat like trimming down to the grid level
    #     pos = pos - ( (pos + self.coordOrg) % self.vsize ) # can I kick coordOrg?
    #     id = int( (pos[0] * self.x_id)
    #             + (pos[1] * self.y_id)
    #             + (pos[2] * self.z_id) )
    #     try:
    #         vox_pos = np.where(self.id == id)[0][0]
    #         return(vox_pos)
    #     except:
    #         print('couldnt find (x,y,z) ID for: ' + str(pos))



    def setVoxeltoVoxel(self, pos, new_vox):
        '''Replaces the voxel instance in voxelSpace.vspace
        at position pos with the voxel instance provided by new_vox.'''
        self.vspace[pos] = None
        self.vspace[pos] = new_vox
        self.logger.info('voxel at entry %d filled', pos)


    def copyVoxelToVoxel(self, xyz, newxyz):
        '''Copies voxel instance onto the position of another
        voxel instance. Overrides the old instance in turn.'''
        vox = self.getVoxel(xyz)
        copy_to_pos = self.getPositionFromCoords(newxyz)
        self.logger.info('voxel at entry %d copied', copy_to_pos)
        self.setVoxeltoVoxel(vox, copy_to_pos)




    def fillVoxel(self, xyz, content='coords', intensity=None, \
                            source=None, pos=None):
        '''Pours data (coords, intensity) into voxel instances, the data is
        stored in the voxel subclass voxelContent. Initiates voxelContent
        instance if none exists, updates Mask to indicate voxel has been filled.
        The voxel can either be identified through a x,y,z coordinate or a
        position for voxelSpace.vspace . If intensity values are to be filled
        coordinates will be filled simultaneously to reduce computing time. To
        add intensity values later use flag content = 'intensity_add' .'''
        if pos == None:
            pos = self.getPositionFromCoords(xyz)
        vox = self.getVoxel(xyz = xyz, vox_pos = pos)
        if vox is object:
            vox = voxel(self.grid[pos], self.vsize) # create new voxel
            self.vspace[pos] = vox
        elif vox is None: # in case no voxel could be identified
            if pos is not None:
                self.logger.warning('tried pouring into voxel at pos %s , failed.' , str(pos))
            else:
                self.logger.debug('could not place point %s in space' , str(xyz))
            return()
        if vox.content is None:
            vox.initContent()
            self.maskEntry(pos)
        if content == 'coords':
            vox.content.addCoords(xyz)
        elif content == 'intensity':
            vox.content.addCoords(xyz)
            vox.content.addIntensity(intensity)
        elif content == 'intensity_add':
            vox.content.addIntensity(intensity)
        elif content == 'source':
            vox.content.setSource(source)



    def getVoxelContent(self, xyz, content ='coords'):
        '''Returns voxelContent subclass of voxel instance.'''
        vox = self.getVoxel(xyz)
        if vox.content is None:
            return(None)
        if content == 'coords':
            return( vox.content.getCoords() )
        if content == 'intensity':
            return( vox.content.getIntensity() )
        if content == 'source':
            return( vox.content.getSource() )



    def checkVoxelContent(self, xyz):
        '''Returns if a voxelcontent subclass exist for a voxel instance'''
        vox = self.getVoxel(xyz)
        if vox.content is None:
            return(False)
        if vox.content is not None:
            return(True)



    def maskArray(self, arr = 'id'):
        '''Creates masked array for voxelSpace.grid or voxelSpace.id which
        excludes masked entries. Should only be run after all data has been
        poured into the voxelSpace.vspace .'''
        import numpy.ma as ma
        if arr == 'id':
            return(ma.masked_array(self.id, mask= self.mask))
        elif arr == 'grid':
            return(ma.masked_array(self.grid,
                    mask= np.column_stack((self.mask, self.mask, self.mask)) ))



    def removeMaskedVoxels(self, rmv_entry = False):
        '''Deletes all voxel instances from voxelSpace.vspace for which
        voxelSpace.mask holds value = 0 i.e. 'empty' voxels.'''
        self.vspace = np.delete(self.vspace, np.where(self.mask == 0)[0], axis=0)
        if rmv_entry is True:
            self.id = self.maskArray('id')
            self.grid = self.maskArray('grid')



    def maskReset(self, content = None):
        '''Resets the masks for all tile levels. All voxels that contain coords
        will be flagged as True .'''
        if self.vspace is not None:
            if content == None:
                # set init_vale = 1 to enable function to work on existing masks as well
                self.mask = self.maskFromExtent(init_val = 1) # create new and full mask
                self.maskReset(content = 'no_content')
                self.maskReset(content = 'no_coords')
            elif content == 'no_content':
                for pos, voxel in enumerate(self.vspace):
                    if voxel.content is None:
                        #pos = np.where( (self.grid == voxel.getOrigin()).all(axis=1) )[0][0]
                        self.maskEntry(pos)
            elif content == 'no_coords':
                non_empty_pos = np.where(self.mask == 1)[0]
                for pos, voxel in enumerate(self.vspace[self.mask.astype(np.bool)]):
                    if voxel.content.getCoords() is None:
                        #pos = np.where( (self.grid == voxel.getOrigin()).all(axis=1) )[0][0]
                        self.maskEntry(non_empty_pos[pos])
            return(np.nansum(self.mask))
        elif self.tspace is not None:
            self.tile_mask = self.maskFromExtent(init_val = 1)
            for pos, tile in enumerate(self.tspace):
                mask_sum = tile.maskReset() # recursive penetration
                if mask_sum == 0: # mask out if lower level mask is empty
                    self.maskEntry(pos, grid = 'tile')
            return(np.nansum(self.tile_mask)) # upwards propagation



    def maskBelowThreshold(self, content = None, value = None, reset_mask = False,
                            depth = 0, data_type = data_type):
        '''Creates mask for voxels instances that do not fall under the specified
        criterias. Filters are additive, so numerous can be combined. Reset the
        mask through maskBelowThreshold(reset_mask=True) .'''
        # LOWEST LEVEL MASKING (End Note)
        if depth == 0 and self.vspace is not None:
            # CONDITIONAL MASK
            if data_type[content][0] == 'voxelCont':
                func = data_type[content][2]
                removed_entries = 0
                for voxel in self.vspace[self.mask.astype(np.bool)]:
                    if func(voxel.content) < value:
                        pos = np.where( (self.grid == voxel.getOrigin()).all(axis=1) )[0][0]
                        self.maskEntry(pos)
                        removed_entries += 1
                if np.nansum(self.mask) == 0:
                    print('tile ' + str(self.coordOrg) + 'excluded as empty')
                print('removed ' + str(removed_entries) + ' entries')
                print('correspond to ' + str((removed_entries/np.nansum(self.mask))*100) + ' percent')
            return(np.nansum(self.mask)) # upper call stack propagation
        # RECURSIVE PENETRATION
        elif self.tspace is not None and depth > 0:
            non_empty_pos = np.where(self.tile_mask == 1)[0] # non empty (value=1) positions in tile_mask
            for pos, tile in enumerate(self.tspace[self.tile_mask.astype(np.bool)]):
                mask_sum = tile.maskBelowThreshold(content, value, reset_mask, (depth-1) )
                if mask_sum == 0: # mask out if lower level mask is empty
                    self.maskEntry(non_empty_pos[pos], grid = 'tile')
            return(np.nansum(self.tile_mask))
        # HIGHER LEVEL MASKING (End Node)
        elif self.tspace is not None and depth == 0:
            non_empty_pos = np.where(self.tile_mask == 1)[0]
            agg_func = data_type[content][4] # aggregation function
            for pos, tile in enumerate(self.tspace[self.tile_mask.astype(np.bool)]):
                data = tile.dataFromAllVoxels(content = content) # extract data within tile
                aggregate_data = agg_func(data, axis = data_type[content][1]) # aggregate data into metrci
                if aggregate_data < value:
                    self.maskEntry(non_empty_pos[pos], grid = 'tile') # exclude entries
            if np.nansum(self.tile_mask) == 0:
                print('tile ' + str(self.coordOrg) + ' excluded')
            return(np.nansum(self.tile_mask)) # upper call stack propagation
        else:
            self.logger.warning('No voxels found at depth level %d', depth)



    # def updateMaskUpwards(self):
    #     '''Propagates the changes at voxel mask upwards through voxelSpaceTile
    #     and VoxelSpace level.'''
    #     if self.vspace is None and np.nansum(self.tile_mask) > 0:
    #         for count, tile in enumerate(self.tspace[self.tile_mask.astype(np.bool)]):
    #             remaining_vals = tile.updateMaskUpwards()
    #             if remaining_vals == 0:
    #                 self.self.tspace[self.tile_mask.astype(np.bool)][count] = 0
    #         return(np.nansum(self.tile_mask))
    #     elif self.vspace is not None:
    #         return(np.nansum(self.mask))



    def getDepth(self):
        '''returns the depth level of a voxelSpace instance.'''
        if self.vspace is None and self.tspace is not None:
            depth_val = self.tspace[0].getDepth()
            return(depth_val + 1)
        elif self.vspace is not None:
            return(0) # lowest depth level
        else:
            self.logger.warning('No Depth in VoxelSpace')


    # def dataFromAllVoxels(self, content, data_type = data_type):
    #     '''Extracts one type of attribute from voxel class iterrated over all
    #     voxel instances within voxelSpace.vspace . Types can be found in dict
    #     data_type. Returns array containing all data.'''
    #     if np.nansum(self.mask) > 0:
    #         axis = data_type[content][1]
    #         func = data_type[content][2]
    #         if data_type[content][0] == 'voxelCont':
    #             for count, voxel in enumerate(self.vspace[self.mask.astype(np.bool)]):
    #                 vox_content = voxel.getContent()
    #                 if vox_content is not None: # in case of being used on vspace where voxels contain no content
    #                     data = func(vox_content)
    #                     if data is not None: # verifies that content has been initiated
    #                         if 'arr' in locals():
    #                             arr = np.append(arr, data, axis= axis)
    #                         else:
    #                             arr = data
    #         elif data_type[content][0] == 'voxelAttr':
    #             for count, voxel in enumerate(self.vspace[self.mask.astype(np.bool)]):
    #                 data = func(voxel)
    #                 if 'arr' in locals():
    #                     arr = np.append(arr, [data], axis= axis)
    #                 else:
    #                     arr = [data]
    #         return(arr)
    #     else:
    #         print('No data in tile ' + str(self.coordOrg) + ' to ' + str(self.coordOrg + self.size))
    #         return(None)



    def dataFromAllVoxels(self, content, data_type = data_type):
        '''Extracts one type of attribute from voxel class iterrated over all
        voxel instances within voxelSpace.vspace . Types can be found in dict
        data_type. Returns array containing all data.'''
        if np.nansum(self.mask) > 0:
            axis = data_type[content][1]
            func = data_type[content][2]
            if data_type[content][0] == 'voxelCont':
                arr = [] #initiation array
                for count, voxel in enumerate(self.vspace[self.mask.astype(np.bool)]):
                    vox_content = voxel.getContent()
                    if vox_content is not None: # in case of being used on vspace where voxels contain no content
                        data = func(vox_content)
                        arr.append(data)
                if axis == 0:
                    arr = np.vstack(arr)
                elif axis == None:
                    arr = np.hstack(arr)
            elif data_type[content][0] == 'voxelAttr':
                arr = []
                for count, voxel in enumerate(self.vspace[self.mask.astype(np.bool)]):
                    data = func(voxel)
                    arr.append(data)
                if axis == 0:
                    arr = np.vstack(arr)
                elif axis == None:
                    arr = np.hstack(arr)
            elif data_type[content][0] == 'voxel':
                arr = self.vspace[self.mask.astype(np.bool)]
            return(arr)
        else:
            self.logger.warning('No data in space ' + str(self.coordOrg) + ' to ' + str(self.coordOrg + self.size))
            return(None)



    def tileVoxelSpace(self, tile_size, tile_vsize = None, depth_level = 0, depth_size_factor = 2):
        '''Divides the space into equally sized sub-voxelSpaces Tiles. These sub-spaces
        can in themself contain sub-spaces. The depth of how many times spaces
        are to be divided recursively is specified through the flag, depth_level.
        The flag depth_size_factor specifies the factor by which the size of
        the sub-space and its internal voxels varries from the mother-voxelSpace.
        Initial sub-space size must be provided as a three dimensional array
        through the flas tile_size. Initial sub-space voxel size can be provided
        as a three dimensional array through the flag tile_vsize .'''
        if tile_vsize is None:
            tile_vsize = self.vsize
        self.tile_size = tile_size
        self.tile_grid = self.gridFromExtent(size = self.size, vsize = tile_size)
        self.tile_mask = self.maskFromExtent(grid = 'tile')
        self.tspace = []
        self.level = depth_level # just for reference, not utilised in navigation
        if depth_level > 0: # upper recursion level
            for grid_pos in self.tile_grid:
                tile = voxelSpaceTile(tile_coord_org = grid_pos,
                                        size = tile_size,
                                        vsize = tile_vsize,
                                        voxelSpace_grid = self.grid,
                                        init_voxel = False)
#                   self.logger.debug('creating Tile: '+str(tile.coordOrg)+' with size '+str(tile.size))
                tile.tileVoxelSpace(tile_size = (tile.size/depth_size_factor),
                                    tile_vsize = (tile.vsize/depth_size_factor),
                                    depth_level = (depth_level - 1),
                                    depth_size_factor = depth_size_factor )
                self.tspace.append(tile)
        elif depth_level == 0: # lowest recursion level.
            for grid_pos in self.tile_grid:
                tile = voxelSpaceTile(tile_coord_org = grid_pos,
                                        size = tile_size,
                                        vsize = tile_vsize,
                                        voxelSpace_grid = self.grid,
                                        init_voxel = True)
                self.tspace.append(tile)
        self.tspace = np.array(self.tspace)
        if type(self) is voxelSpace:
            self.logger.info('tiled voxelSpace with tilesize %s resulting in %d tiles',
                                str(self.tile_size), len(self.tspace))



    # def initProgressTracker(self):
    #     if self.level > 0:
    #         #vox_count = len(self.tspace)* self.tile_vsize^(3 * self.depth)
    #         vox_count = (len(self.tspace) * 1/(self.tile_vsize^3)) ^ (3*self.depth)
    #     else:
    #         vox_count = len(self.vspace)
    #     self.vox_count = vox_count
    #     self.vox_progress = 0
    #
    #
    #
    # def incrementProgressTracker(self):
    #     self.vox_progress += 1
    #
    #
    #
    # def callProgressTracker(self, block_size = 1000):
    #     if (self.vox_count % block_size) == 0:
    #         perc = np.round((self.vox_progress / self.vox_count)*100,1)
    #         print(str(perc)+ '% filled of voxelSpace ' + str(self.coordOrg) + ' with size ' + str(self.size))
    #         sys.stdout.write("\033[F") #back to previous line
    #         sys.stdout.write("\033[K") #clear line



    def callProgressTracker(self, data_idx, data_len):
        perc = np.round((data_idx / data_len)*100,1)
        print(str(perc)+ '% filled of voxelSpace ' + str(self.coordOrg) + ' with size ' + str(self.size))
        sys.stdout.write("\033[F") #back to previous line
        sys.stdout.write("\033[K") #clear line


    def quickFillVoxelSpace(self, xyz_arr_full, int_arr_full, box_size = 10):
        '''quicker version in filling voxelSpace as it utilises numpy to
        pre chunk the data. Only works for untiled voxelSpaces.
        Must provide both coordinate and intensity data. VoxelSpace
        must be initiated with empty voxels. box_size is giving the
        block processing window in map units.
        No checks and varifications, use at own risk! '''
        #if self.vspace is not None: # make sure voxels have been initiated
        #print('start')
        #blocks = blockDivide(xyz_arr_full, bsize = 4)
        #for block in blocks:
            #xyz_arr = xyz_arr_full[block[0]:block[1]]
            #int_arr = int_arr_full[block[0]:block[1]]
        step_num = box_size / self.vsize
        block_size = self.vsize * step_num
        #indices_for_block =  np.prod(self.grid % block_size == 0, axis =1)
        #grid = self.grid[indices_for_block.astype(np.bool)]
        #if np.all(grid[-1] != self.grid[-1]):
        #    grid = np.vstack((grid, self.grid[-1]))
        grid_block = voxelSpace.gridFromExtent(self, size = self.size, vsize = block_size)
        for grid_block_pos, grid_block_entry in enumerate(grid_block):
            # Creation of data block
            mask =  np.all(xyz_arr_full >= grid_block_entry, axis=1) * \
                    np.all(xyz_arr_full < (grid_block_entry + block_size), axis=1)
            if np.sum(mask) > 0:
                xyz_arr = xyz_arr_full[mask]
                int_arr = int_arr_full[mask]
                # Creation of indices to loop over in voxelSpace grid
                mask =  np.all(self.grid >= grid_block_entry, axis=1) * \
                        np.all(self.grid < (grid_block_entry + block_size), axis=1)
                sub_grid = self.grid[mask]
                #print(np.nanmax(sub_grid, axis = 0))
                # Running through block
                for pos, grid_entry in enumerate(sub_grid):
                    sub_mask = np.all(xyz_arr >= grid_entry, axis=1) * \
                            np.all(xyz_arr < (grid_entry + self.vsize), axis=1)
                    count = np.sum(sub_mask)
                    if count > 0: #np.sum(sub_mask) > 0:
                        baseline = np.where( np.all(self.grid == grid_entry, axis = 1) == True )[0][0]
                        #print(baseline)
                        xyz = xyz_arr[sub_mask]
                        intensity = int_arr[sub_mask]
                        vox = voxel(self.grid[grid_block_pos + pos], self.vsize) # create new voxel
                        self.vspace[baseline] = vox
                        vox.initContent()
                        self.maskEntry(baseline)
                        vox.content.addCoords(xyz, count = count)
                        vox.content.addIntensity(intensity, count = count)
                        #xyz_arr = xyz_arr[~sub_mask]
                        #int_arr = int_arr[~sub_mask]
                    else:
                        pass
            #if (grid_block_pos % 100) == 0:
            self.callProgressTracker(grid_block_pos, len(grid_block))
        # for pos, grid_entry in enumerate(self.grid):
        #     mask =  np.all(xyz_arr >= grid_entry, axis=1) * \
        #             np.all(xyz_arr < (grid_entry + self.vsize), axis=1)
        #     if np.sum(mask) > 0:
        #         xyz = xyz_arr[mask]
        #         intensity = int_arr[mask]
        #         vox = voxel(self.grid[pos], self.vsize) # create new voxel
        #         self.vspace[pos] = vox
        #         vox.initContent()
        #         self.maskEntry(pos)
        #         vox.content.addCoords(xyz)
        #         vox.content.addIntensity(intensity)
        #         xyz_arr = xyz_arr[~mask]
        #         int_arr = int_arr[~mask]
        #         if (pos % 100) == 0:
        #             self.callProgressTracker(pos, len(self.grid))
        #     else:
        #         pass
        #print('done!')


    def fillVoxelSpace(self, xyz_arr, content_type = 'coords', int_arr = None):
        '''Fills all voxel instances of a voxelSpace with content corresponding
        to the content_type flag. Tiles are filled downwards recursively in
        itteration down to a depth level where voxels are encountered.'''
        if self.vspace is not None: # filling data into vspace if already on lowest level
            self.logger.debug('filling %s', str(self.coordOrg))
            #print('filling space: ' + str(self.coordOrg) ,' to ', str(self.coordOrg + self.size))
            if content_type == 'coords':
                for data_idx, xyz in enumerate(xyz_arr):
                    self.fillVoxel(xyz, content = 'coords')
                    if (data_idx % 1000) == 0:
                        self.callProgressTracker(data_idx, len(xyz_arr))
            elif content_type == 'intensity' and int_arr is not None:
                for data_idx, (xyz, int) in enumerate(zip(xyz_arr, int_arr)):
                    self.fillVoxel(xyz, content = 'intensity', intensity = int) # fills both coords and intensity at the same time
                    if (data_idx % 1000) == 0:
                        self.callProgressTracker(data_idx, len(xyz_arr))
        elif self.vspace is None and self.tspace is not None: # recursive penetration into lower levels
            for pos, lower_level_tile in enumerate(self.tspace):
                mask = lower_level_tile.extractTileData(xyz_arr, return_mask=True)
                if mask is not None and np.nansum(mask) > 0: # cope with empty masks
                    if content_type == 'coords':
                        lower_level_tile.fillVoxelSpace( xyz_arr[mask])
                        self.maskEntry(pos, grid='tile')
                    elif content_type == 'intensity':
                        lower_level_tile.fillVoxelSpace( xyz_arr[mask], content_type, int_arr[mask])
                        self.maskEntry(pos, grid='tile')
                        int_arr = int_arr[np.invert(mask)] # invert mask, subtract already used entries of xyz_arr from xyz_arr.
                    xyz_arr = xyz_arr[np.invert(mask)] # this reduces the number of entries that need to be searched



    def extract(self, content, depth = None):
        '''Parent function to extract data according to attribute content from
        any voxelSpace.'''
        if depth is not None: # on lowest level
            data = self.extractFromDepth(depth_threshold= depth, content=content)
        elif self.vspace is None and self.tspace is not None:
            data = self.extractDataFromTileSpace(content= content)
        elif self.vspace is not None:
            data = self.dataFromAllVoxels(content= content)
        else:
            return('Error trying to extract data %s from voxelSpace'%content)
        return(data)


    def funcVoxelSpace(self, func, **kwargs):
        '''Runs Voxel function for all voxels in Voxelspace. Functions should
        not produce output but rather store their information within the voxels.
        Afterwards data can be extracted through extractDataFromTilespace.
        Tiles are processed downwards recursively in itteration down to a depth
        level where voxels are encountered. This data will NOT be extractable
        through voxelSpaceToVoxel and thus not for higher depth levels.'''
        if self.vspace is not None: # filling data into vspace if already on lowest level
            for vox in self.vspace[self.mask.astype(np.bool)]:
                func(vox, **kwargs)
        elif self.vspace is None and self.tspace is not None: # recursive penetration into lower levels
            for pos, lower_level_tile in enumerate(self.tspace[self.tile_mask.astype(np.bool)]):
                lower_level_tile.funcVoxelSpace( func, **kwargs )



    def voxelSpaceToVoxelSpace(self, depth_threshold):
        '''Creates a new instance of a voxelSpace with different resolution'''
        self.logger.debug('re-resolving voxelSpace')
        vspace = self.extractFromDepth(depth_threshold = depth_threshold, content = 'voxel')
        vsize = vspace[0].size
        # add vsize to account for smaller steps resolved in sub tiles
        # could also be done by extracting the exact dimension from tiles,
        # but that would be more time consuming and probably will have the same effect
        grid = []
        for vox in vspace:
            grid.append(vox.origin)
        grid = np.vstack(grid)
        #
        newSpace = voxelSpace(width = (self.width + vsize[0]),
                                depth = (self.depth + vsize[1]),
                                height = (self.height + vsize[2]),
                            vsize = vsize, coordOrg = self.coordOrg,
                            init_voxel = False)
        newSpace.vspace = vspace
        newSpace.grid = grid
        #newSpace.mask = newSpace.maskFromExtent(init_val = 1)
        # newSpace.maskBelowThreshold(content = 'no_content') # should not be happening
        #newSpace.maskBelowThreshold(content = 'no_coords')
        newSpace.mask = np.ones(len(vspace)) # since only filled voxels remain
        self.logger.debug('re-resolved voxelSpace to former depth level %d', depth_threshold)
        return(newSpace)


    def extractVoxelsFromTileSpace(self):
        '''Extracting voxel instance array (vspace) out of tiled voxelSpace'''
        if self.vspace is not None: # lowest level
            arr = self.vspace[self.mask.astype(np.bool)]
        elif self.vspace is None and self.tspace is not None: # upper level
            arr = []
            for tile in self.tspace[self.tile_mask.astype(np.bool)]:
                arr_tile = tile.extractVoxelsFromTileSpace()
                arr.extend(arr_tile)
        self.logger.debug('extracted voxels {grid}', grid = self.grid[self.mask.astype(np.bool)])
        return(arr)



    def extractDataFromTileSpace(self, content, data_type = data_type):
        '''Extracting data array out of tiled voxelSpace corresponding to the
        content flag. Tiled voxelSpace is recursively and iterratively penetrated
        until a vspace array of voxels is encountered from which data can be
        obtained.'''
        axis = data_type[content][1]
        if self.vspace is not None: # case when on lowest level
            arr = self.dataFromAllVoxels(content)
        elif self.vspace is None and self.tspace is not None and np.nansum(self.tile_mask) > 0:  # case when on higher level
            axis = data_type[content][1]
            arr = []
            for tile in self.tspace[self.tile_mask.astype(np.bool)]:
                arr_tile = tile.extractDataFromTileSpace(content = content)
                if arr_tile is not None:
                    arr.append(arr_tile)
            if axis is None:
                arr = np.hstack(arr) # integer values can be appended
            elif axis == 0:
                arr = np.vstack(arr)
        else:
            arr = None # No data found in tilespace/ tile
            self.logger.warning('No data found in tile {coordOrg} to {coordEnd}',
                                    coordOrg = self.coordOrg,
                                    coordEnd = self.coordOrg + self.size)
        return(arr)



    def extractFromDepth(self, depth_threshold, depth_level = 0, content = None,
                            data_type = data_type):
        '''Extracts data or instances for specific depth in the voxelSpace
        structure and unifies them for the depth level resolution.
        Voxels can also be extracted, if this is the case a list of
        voxel instances will be returned that are equivalent in size to voxels
        that are found on this tiles depth level. Since the lowest level
        houses only voxels but no tiles the maximum of value for
        depth_threshold is equal to total depth levels - 1. For extraction of
        data on the lowest level use extractDataFromTileSpace .'''
        axis = data_type[content][1]
        arr = np.zeros((1,data_type[content][3])) # initial array to append onto
        if depth_level < depth_threshold:
            for tile in self.tspace[self.tile_mask.astype(np.bool)]:
                tile_arr = tile.extractFromDepth(depth_threshold = depth_threshold,
                                            depth_level = depth_level + 1,
                                            content = content)
                if tile_arr is not None:
                    if axis is None:
                        arr = np.append(arr, tile_arr, axis= axis) # integer values can be appended
                    else:
                        arr = np.vstack((arr, tile_arr)) # arrays will be stacked
        else:
            if np.nansum(self.tile_mask) > 0:
                if content == 'voxel':
                    arr = np.array(voxel([0,0,0], 0)) # initiating voxel array to append to
                    for tile in self.tspace[self.tile_mask.astype(np.bool)]:
                        vspace = tile.extractVoxelsFromTileSpace()
                        if len(vspace) > 0:
                            # creating temporary voxelSpace to pour voxels into
                            temp_space = voxelSpace(tile.size[0], tile.size[1], tile.size[2],
                                                    tile.vsize, tile.coordOrg, init_voxel = False)
                            temp_space.vspace = np.array(vspace) # transform to array so the mask can function properly
                            temp_space.mask = np.ones(len(vspace), dtype = np.int8) # set up mask to indicate all voxels are filled
                            # resolve temporary voxelspace into single voxel
                            vox = temp_space.voxelSpaceToVoxel()
                            arr = np.append(arr, vox)
                elif content == 'grid':
                    arr = np.append(arr, self.tile_grid, axis = 0)
                elif content == 'mask':
                    arr = np.append(arr, self.tile_mask)
                elif content in data_type: # checks if the data that is searched for is part of voxel internal data
                    for tile in self.tspace[self.tile_mask.astype(np.bool)]:
                        data = tile.extractDataFromTileSpace(content)
                        if data is not None: #and len(data) > 0:
                            data_round = np.round(data_type[content][4](data, axis = axis), self.precision)  # HOW SHOULD DATA BE TREATED FOR AVERIGING?
                            if axis is None:
                                arr = np.append(arr, data_round, axis= axis) # integer values can be appended
                            else:
                                arr = np.vstack((arr, data_round)) # arrays need to be stacked
            else:
                self.logger.warning('Tilespace empty / No data found at %s'%str(self.coordOrg))
                return(None) # CASE IN WHICH THERE IS NO DATA IN ANY TILES
        arr = arr[1:] # removal of initialisation entry
        return(arr)



    # def extractDataFromDepth(self, depth, content, depth_level = 0):
    #     '''Extracts data from a specific depth level within a tiled voxelSpace.
    #     utilises 'extractFromDepth' . The depth level at which data is to
    #     be extracted and averaged is specified through the flag 'depth'. Data
    #     type to be extracted is specified through the flag 'content' .'''
    #     arr = []
    #     for tile in self.tspace:
    #         if depth == depth_level:
    #             data = tile.dataFromAllVoxels(content)
    #             if data is not None:
    #                 arr.extend(data)
    #         else:
    #             data = tile.extractDataFromDepth(depth = depth,
    #                                             content = content,
    #                                             depth_level = (depth_level + 1))
    #             arr.extend(data)
    #     return(np.array(arr))



    def tileSpaceToVoxelSpace(self):
        '''Converts voxelSpaceTiles, which contain voxels, into single
        voxelSpace. Utilises voxelSpaceToVoxel to convert voxelSpace instance
        to voxel instance. For functionality vspace must be initiated and
        data poured into voxelSpace.'''
        self.logger.debug('converting tiles into voxels for tile %d', self.coordOrg)
        self.vspace = []
        self.vsize = self.tile_size
        self.grid = self.tile_grid
        self.mask = self.maskFromExtent(init_val = 0)
        for pos, tile in enumerate(self.tspace):
            if tile.vspace is None:
                tile.tileSpaceToVoxelSpace()
            vox = tile.voxelSpaceToVoxel()
            self.vspace.append(vox)
            self.maskEntry(pos)
        self.vspace = np.array(self.vspace)
        self.maskBelowThreshold(content = 'no_content') # masks no content values
        self.maskBelowThreshold(content = 'no_coords') # masks no coords values
        del(self.tspace, self.tile_size)
        self.logger.info('converted tiles into voxels for tile %d', self.coordOrg)



    def voxelSpaceToVoxel(self):
        '''converts all instances of voxels within the voxelspace into a
        single voxel with corresponding voxelContent which contains the data of
        the input voxelSpace.vspace . Attributes of the vspace will be averaged
        over all voxels before.'''
        # extracting data
        if self.mask is None or sum(self.mask) > 0: # only the case if data exists in vspace/ tspace
            if len(self.vspace) > 0: # ensures no empty voxelSpaces (from removeMaskedVoxels) are evaluated
                coords = self.dataFromAllVoxels( content = 'coords')
                coord_count = len(coords)
            try:
                intensity = self.dataFromAllVoxels( content = 'intensity')
                intensity_count = len(intensity)
            except:
                intensity = None
                intensity_count = 0
                self.logger.warning('could not load intensity for voxelSpace origin %d', str(self.coordOrg))
        else:
            coords = None
            coord_count = 0
            intensity = None
            intensity_count = 0
            self.logger.warning('could not load intensity for voxelSpace origin %d', str(self.coordOrg))
        # building the replacing voxel
        vox = voxel(self.coordOrg, self.size) #np.max(self.size)) # USES THE LARGEST SIDE FOR THE VOXEL SIZE
        vox.initContent()
        content = vox.getContent()
        content.setCoords(coords)
        content.setCoordCount(coord_count)
        content.setIntensityCount(intensity_count)
        content.setIntensity(intensity)
        vox.setContent(content)
        return(vox)



    def structureTensor(self, dim, sigma = 1, rho = 5, perc = 0, thresh= None):
        '''Returns the structural tensor object for a voxelSpace instance.
        Does not function for tiled voxelSpaces'''
        # initial parameters
        #sigma = 1.0 # noise scale, values smaller then sigma will be removed
        #rho = 5 # window of neighbours to consider, integration scale
        # Data extraction
        #coord_count = self.dataFromAllVoxels(content = 'coord_count')
        coord_count = self.dataFromAllVoxels(content = 'intensity_sum')
        #Shape for array must be Z, Y, X ?
        count_arr = np.zeros(len(self.grid))
        count_arr[self.mask.astype(np.bool)] = coord_count
        count_arr = count_arr.reshape(np.ceil(self.size / self.vsize).astype(np.int)) # reshape into 3 dimensions with voxel resolution for extend
        # NEW APPROAC
        #coord_count = self.extract(content = 'intensity_sum', depth=1)
        #mask = self.extract(content = 'mask', depth = 1)
        #grid = self.extract(content = 'grid', depth = 1)
        ## Shape for array must be Z, Y, X ?
        #count_arr = np.zeros(len(grid))
        #count_arr[mask.astype(np.bool)] = coord_count
        #grid_size = np.nanmax(grid, axis= 0) - np.nanmin(grid, axis= 0)
        ## might be squichy for values that are only located in 1 entry per dimension throughout the voxelspace
        #x_step = np.unique(grid[mask.astype(np.bool),0])[1] - np.unique(grid[mask.astype(np.bool),0])[0]
        #y_step = np.unique(grid[mask.astype(np.bool),1])[1] - np.unique(grid[mask.astype(np.bool),1])[0]
        #z_step = np.unique(grid[mask.astype(np.bool),2])[1] - np.unique(grid[mask.astype(np.bool),2])[0]
        #step_size = np.array([x_step, y_step, z_step])
        #count_arr = count_arr.reshape(np.ceil(grid_size / step_size).astype(np.int)) # reshape into 3 dimensions with voxel resolution for extend
        # Tensor calculation

        #count = count.astype('float')
        #count /= count.max()
        trc = structure_tensor_3d(count_arr, sigma, rho)
        val, vec = eig_special_3d(trc)
        # calculating Eigenvalues
        zi, yi, xi = np.nonzero(count_arr)
        # l0 < l1 < l2
        l0 = val[0, zi, yi, xi] + 1e-14 # making sure not equal to 0
        l1 = val[1, zi, yi, xi] + 2e-14
        l2 = val[2, zi, yi, xi] + 3e-14

        if dim == [1,2,3] or dim == 'all':
            t_lin = (l2 - l1) / l2
            t_pla = (l1 - l0) / l2
            t_iso = l0 / l2
            return(t_lin, t_pla, t_iso)
        # how linear is the tensor
        elif dim == [1,2] or dim == "linear":
            tensor_val = (l2 - l1) / l2 # comparing if the two bigger Eigenvalues are similar, normalised to the biggest Eigenvalue
        # how flat is the tensor
        elif dim == [0,1] or dim == "flat":
            tensor_val = (l1 - l0) / l2 # comparing if the two smaller Eigenvalues are similar, normalised to the biggest Eigenvalue
        # how isoropic is the tensor
        elif dim == [0] or dim == "isotropic":
            tensor_val = l0 / l2 # comparing the smallest to the largest Eigenvalue, result should be close to one for isomorphy
        #
        if thresh is not None and thresh > 0:
            thresh_val = thresh
        else:
            thresh_val = np.percentile(tensor_val, perc)
        #thresh_val = 0.5
        #tensor_val = tensor_val[tensor_val < thresh_val]
        tensor_val[tensor_val < thresh_val] = np.nan
        return(tensor_val)

        # Approach - first reshape, then fill
        #count_arr = np.zeros(self.size.astype(np.int)) # creating a 3D formed array to pour the data into
        #count_arr_mask = self.mask.reshape(count_arr.shape) # reshaping mask in order
        non_empty_pos = np.where(self.mask == 1)[0]


    def intHist(self, depth = None, step_size = 50, int_high_val = 1500):
        '''Builds a histogram for the intensity values contained by voxels/
        voxelTiles within. The Flag int_high_val specifies until where intensity
        shall be effectively bined.'''
        #vspace = self.extractVoxelsFromTileSpace()
        if self.vspace is None and depth is not None:
            intensity = self.extractFromDepth(depth_threshold = depth, content = 'intensity') # 'intensity_mean')
        elif self.vspace is None:
            intensity = self.extractDataFromTileSpace(content = 'intensity') # 'intensity_mean')
        else:
            intensity = self.dataFromAllVoxels(content = 'intensity') # 'intensity_mean')
        bins = np.arange(200,int_high_val, step_size)
        bins = np.append(bins, 3000) # do account for high values
        H, xedges = np.histogram(intensity, bins = bins)
        return(H, xedges)



    def coordHist(self, dim, depth = None, bin_factor = 10, step = 0.1,
                    content = 'coords', cluster_flag = True):
        '''Builds a histogram for the coordinates for a specific dimension.
        Dim needs to be specified as integer (0,1,2). Resolution can be set
        through the 'res' flag (unit meter or CRS).'''
        if self.vspace is None and depth is not None:
            coords = self.extractFromDepth(depth_threshold = depth, content = 'center')
            data = self.extractFromDepth(depth_threshold = depth, content = content)
            x_step = self.extractFromDepth(depth_threshold = depth, content = 'size')[0][dim]
        elif self.vspace is None:
            coords = self.extractDataFromTileSpace(content = 'center')
            data = self.extractDataFromTileSpace(content = content)
            x_step = self.extractDataFromTileSpace(content = 'size')[0][dim]
        else:
            coords = self.dataFromAllVoxels(content = 'center')
            data = self.dataFromAllVoxels(content = content)
            x_step = self.dataFromAllVoxels(content = 'size')[0][dim]
        x_min = self.coordOrg[dim]
        x_max = self.coordOrg[dim] + self.size[dim]
        #x_step = self.size[dim] / bin_factor
        if step is not None:
            x_step = step # 30cm bins
        x_bin = np.arange(x_min, x_max + x_step , x_step )
        H, xedges = np.histogram(coords[:,dim], bins = x_bin)
        ## calculation of clustering in x,y dimension in bin
        if cluster_flag is True:
            dist_arr = []
            for i, pos in enumerate(xedges[1:]):
                min_z = xedges[i]
                max_z = pos
                if H[i] > 1:
                    coords_in_bin = coords[ (coords[:,2] > min_z) & (coords[:,2] < max_z) ]
                    coords_mean = np.mean(coords_in_bin[:,:2], axis = 0)
                    #cluster_dist = np.round(np.mean(abs(coords_in_bin[:,:2] - coords_mean) / self.size[0] ), 3 )
                    cluster_dist = np.round(np.mean(abs(coords_in_bin[:,:2] - coords_mean) ), 3 )
                else:
                    cluster_dist = 0 #None
                dist_arr.append(cluster_dist)
            dist_arr = np.array(dist_arr)
        return(H, xedges, x_step, dist_arr) # returns step so it can be used in plot


    def exposHist(self, depth = None):
        '''Creats a histogram for the specified content.'''
        if self.vspace is None and depth is not None:
            exp = self.extractFromDepth(depth_threshold = depth, content = 'eucl_exp_mode')
        elif self.vspace is None:
            exp = self.extractDataFromTileSpace(content = 'eucl_exp_mode')
        else:
            exp = self.dataFromAllVoxels(content = 'eucl_exp_mode')
        exp[np.where(exp == 24)] = 9
        x_min = 0 # scale ranges from 0 to 8 and 24
        x_max = 9 # 24 needs therefore to be mapped to 9
        x_bin = np.arange(x_min, x_max + 1 , 1)
        H, xedges = np.histogram(exp, bins = x_bin)
        return(H, xedges) # returns step so it can be used in plot



    def footPrint(self):
        '''Creates a grid for the X,Y dimension of the voxels or tiles,
        depicting the corners of individual substructures.'''
        if self.vspace is None:
            step_size = self.tile_size
        else:
            step_size = self.vsize
        xedges = np.arange(self.coordOrg[0], (self.coordOrg[0] + self.size[0] +
                            step_size[0]), step_size[0])
        yedges = np.arange(self.coordOrg[1], (self.coordOrg[1] + self.size[1] +
                            step_size[1]), step_size[1])
        C = np.zeros( (len(xedges), len(yedges)) )
        X, Y = np.meshgrid(xedges, yedges)
        return(X,Y,C)



    def coordHist2d(self, dim1, dim2, depth= None, res = 0.1, bin_factor = None,
                    data_type = data_type, hist_val = 'coords', val_dim = None):
        '''Builds a 2D histogram for coordinates for two provided dimensions.
        Dim needs to be specified as integer (0,1,2). Resolution can be set
        through the 'res' flag (unit meter or CRS). Results can be visualized
        through plt.pcolormesh .'''
        if self.vspace is None and depth is not None:
            coords = self.extractFromDepth(depth_threshold = depth, content = 'coords')
        elif self.vspace is None:
            coords = self.extractDataFromTileSpace(content = 'coords')
        else:
            coords = self.dataFromAllVoxels(content = 'coords')
        # case when different variable is to be displayed as 'histogram'
        if hist_val is not 'coords':
            if self.vspace is None and depth is not None:
                val_arr = self.extractFromDepth(depth_threshold = depth, content = hist_val)
            elif self.vspace is None:
                val_arr = self.extractDataFromTileSpace(content = hist_val)
            else:
                val_arr = self.dataFromAllVoxels(content = hist_val)
        #
        x_min = self.coordOrg[dim1]
        x_max = self.coordOrg[dim1] + self.size[dim1]
        #
        y_min = self.coordOrg[dim2]
        y_max = self.coordOrg[dim2] + self.size[dim2]
        #
        if bin_factor is not None: # factor how the size will be upscaled for each bin
            x_step = self.size[dim1]/ bin_factor[dim1]
            y_step = self.size[dim2]/ bin_factor[dim2]
            pos_arr = np.floor( ((coords)/ (self.size / bin_factor)) ).astype(np.int)
            # self.grid[self.mask.astype(np.bool)]
        else:
            x_step = res
            y_step = res
            pos_arr = np.floor ( ((coords)/ res) ).astype(np.int)
        #
        x_bin = np.arange(x_min, x_max + x_step , x_step )
        y_bin = np.arange(y_min, y_max + y_step , y_step )
        #
        # if the count of coordinates are to be displayed in the histogram
        H, xedges, yedges = np.histogram2d(coords[:,dim1], coords[:,dim2],
                                            bins= np.array([x_bin,y_bin]))
        H = H.astype(np.int)
        X, Y = np.meshgrid(xedges, yedges)
        if hist_val in data_type: # case where special value is to be displayed on the histogram
            # build empty 2d histogram like array
            H_mask = H > 0 # determine where data is located in original array
            #x_extend = (x_max - x_min) / (self.size[dim1] / bin_factor[dim1])
            #y_extend = (y_max - y_min) / (self.size[dim2] / bin_factor[dim2])
            x_extend = np.ceil(bin_factor[dim1])
            y_extend = np.ceil(bin_factor[dim2])
            H = np.zeros( ( ( x_extend ).astype(np.int), ( y_extend ).astype(np.int) ) )
            # build corresponding count array
            H_c = np.ones( ( ( x_extend ).astype(np.int), ( y_extend ).astype(np.int) ) )
            #
            # because np.unique returns sorted values by axis, the index is matching with the val_arr/ original voxel index
            for pos, val in zip(np.unique(pos_arr, axis = 0), val_arr):
                H_c[pos[dim1], pos[dim2]] += 1
                H[pos[dim1], pos[dim2]] += np.sum(val[val_dim,])
            return((H / H_c).transpose(), X, Y)# rotation only for cosmetic purposes on the plot
        return(H.transpose(), X, Y)
        #plt.pcolormesh(X, Y, H, cmap = 'PuBu')
        #plt.show()



    def create3dGrid(self, depth = 0):
        '''Creates a 3 dimensional grid for the voxelSpace.vspace representing
        all filled voxel as True values, entries are spaced by the corresponding
        vsize/ tsize.'''
        # if self.vspace is None:
        #     step_size = self.tile_size
        #     mask = self.tile_mask.astype(np.bool)
        #     grid = self.tile_grid[mask]
        # else:
        #     step_size = self.vsize
        #     mask = self.mask.astype(np.bool)
        #     grid = self.grid[mask]
        grid, mask, step_size = self.getGridFromDepth(depth = depth)
        #creates empty space - distance between integer indices corresponds to step_size
        grid3d = np.zeros(np.ceil(self.size /step_size).astype(np.int))
        #index for position of voxels that contain data
        idx =( (grid - self.coordOrg) / step_size ).astype(np.int)
        #fill voxels
        for i in idx:
            grid3d[i[0]][i[1]][i[2]] = True
        # build correct labels for grid
        idx_size = np.ceil(self.size / step_size).astype(np.int)
        x,y,z = np.indices( (idx_size[0]+1, idx_size[1]+1, idx_size[2]+1 )  )
        x = x * step_size[0] + self.coordOrg[0]
        y = y * step_size[1] + self.coordOrg[1]
        z = z * step_size[2] + self.coordOrg[2]
        return(grid3d, x, y, z, idx, idx_size)



    def getVoxelSizeFromDepth(self, depth, depth_level = 0):
        '''Returns the size of voxels/ tiles at specified depth'''
        if depth_level < depth:
            try:
                tile = self.tspace[self.tile_mask.astype(np.bool)][0]
            except:
                self.logger.error("no tiles/ voxels found at depth level %d", depth_level)
                raise RecursionError('specified depth exceeds depth of voxelSpace')
                #
            size_at_depth = tile.getVoxelSizeFromDepth(depth = depth,
                                    depth_level = depth_level + 1)
        else:
            if self.vspace is None:
                tile = self.tspace[self.tile_mask.astype(np.bool)][0]
                size_at_depth = tile.size
            else:
                vox = self.vspace[self.mask.astype(np.bool)][0]
                size_at_depth = vox.size
        return(size_at_depth)



    def getGridFromDepth(self, depth = 0):
        '''Returns a grid, the corresponding mask from a certain depth.
        Depth cannot be greater than depth within voxelSpace.'''
        if depth is None or depth is 0:
            if self.vspace is None:
                step_size = self.tile_size
                mask = self.tile_mask.astype(np.bool)
                grid = self.tile_grid[mask]
            else:
                step_size = self.vsize
                mask = self.mask.astype(np.bool)
                grid = self.grid[mask]
        elif depth > 0 :
            grid = np.zeros((1,3)) # initial array to append onto
            mask = np.array([True])
            for tile in self.tspace:
                grid_i, mask_i, step_size = tile.getGridFromDepth(depth = depth - 1)
                grid = np.vstack((grid, grid_i)) # integer values can be appended
                mask = np.append(mask, mask_i, axis = 0) # arrays need to be stacked
            grid =grid[1:] # removal of initialisation entry
            mask = mask[1:]
        return(grid, mask, step_size)




    # def plot3d(self, draw_vox = True, draw_pts = True, pts ='coords',
    #             pts_col = None, prt_to_scr = True, data_type = data_type,
    #             vox_face_col = '#555efc', vox_edge_col = '#151738', vox_alpha = 0.1,
    #             pts_face_col = '#2d0370', pts_alpha = 0.6,  pts_size = 0.3,
    #             fig = None, subplot_nr = None):
    #     '''Build a plot-instance for voxels and or x,y,z points within the space.
    #     Only voxels / points that are marked in self.mask/ self.tile_mask are
    #     drawn. Figure can be parsed to another function through 'pts_to_scr' = False.
    #     '''
    #     from mpl_toolkits.mplot3d import Axes3D
    #     if fig is None:
    #         fig = plt.figure()
    #         ax = fig.gca(projection='3d')
    #     elif fig is not None and subplot_nr is not None: # compatability with plotting functions
    #         ax = fig.add_subplot(subplot_nr, projection='3d')
    #     #ax.pbaspect = [tile.size[0], tile.size[1], tile.size[2]]
    #     if draw_vox is True: # voxel plot for voxels
    #         space3d, x, y, z = self.create3dGrid()
    #         ax.voxels( x, y, z,
    #                 filled = space3d,
    #                 facecolors= vox_face_col ,
    #                 edgecolors= vox_edge_col, #'#bbbeea',  # brighter
    #                 linewidth=0.2,
    #                 alpha = vox_alpha)
    #     #
    #     if draw_pts is True: # scatterplots for points
    #         if pts_col in data_type:
    #             pts_face_col = self.extractDataFromTileSpace(content = pts_col)
    #         if draw_vox is False: # need to draw correct axis dimensions, non visible points
    #             ext_pts = np.vstack((self.coordOrg, (self.coordOrg + self.size) ))
    #             ax.scatter(ext_pts[:,0], ext_pts[:,1], ext_pts[:,2], alpha = 0)
    #         plt_coords = self.extractDataFromTileSpace(content = pts)
    #         ax.scatter(plt_coords[:,0], plt_coords[:,1], plt_coords[:,2],
    #                     marker = 'o' , alpha = pts_alpha, c=pts_face_col,
    #                     s = pts_size, linewidths = 0)
    #     #
    #     #ax.set(xlabel='x', ylabel='y', zlabel='z')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('z')
    #     ax.grid(linewidth = 0.1, alpha = 0.5)
    #     if prt_to_scr is True:
    #         plt.show()
    #     else:
    #         return(ax) # for further use in other plot function



    def create3dColors(self, vox_val, grid3d, idx, cmap_name = 'viridis', cmap_steps = 10,
                        map_to_cmap = True, min_val = None, max_val = None, normalize = True):
        '''Creates the 3d array containing the RGB values for 3D voxel plots.'''
        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        # Data Adjustment
        if min_val is None:
            min_val = np.nanmin(vox_val)
        if max_val is None:
            max_val = np.nanmax(vox_val)
        if normalize is True:
            vox_val = (vox_val - min_val) / (max_val - min_val) # adjust min and max value of data
        # Color Map
        if map_to_cmap is True:
            cmap = cm.get_cmap(cmap_name, cmap_steps)
            cmap_vox_val = cmap(vox_val) # translate values to RGB
        else:
            cmap_vox_val = vox_val
            #
            #log_val = np.log10(vox_val*1000)
            #cmap_vox_val = np.round(log_val / np.max(log_val), 2)
            #
            #cmap = cm.get_cmap('hsv', cmap_steps)
            #cmap_vox_val = cmap(vox_val)
            #cmap_gray = rgb2gray(vox_val)
            #cmap_vox_val = cmap(cmap_gray)
        # Color Array
        colors3d = np.full(grid3d.shape + (3,), np.nan)
        for i,dim in enumerate(idx):
            colors3d[dim[0]][dim[1]][dim[2]][0] = cmap_vox_val[i,0] # inserting red channel
            colors3d[dim[0]][dim[1]][dim[2]][1] = cmap_vox_val[i,1] # green channel
            colors3d[dim[0]][dim[1]][dim[2]][2] = cmap_vox_val[i,2] # blue channel
        return(colors3d)
        # ##################
        # max_val =
        # min_val = 0
        # vox_face_val = vox_face_val - min_val
        # vox_face_val = vox_face_val / (max_val-min_val) #np.max(vox_face_val)
        #
        # col_bar = cm.get_cmap('viridis', 30)
        # vox_face_col = col_bar(vox_face_val)
        #
        # # create colors for voxel:
        # #colors = np.copy(space3d)
        # #colors = colors[..., np.newaxis]
        # colors = np.zeros(space3d.shape + (3,))
        # colors = np.full(space3d.shape + (3,), np.nan)
        # for ii,i in enumerate(idx):
        #     colors[i[0]][i[1]][i[2]][0] = vox_face_col[ii,0]
        #     colors[i[0]][i[1]][i[2]][1] = vox_face_col[ii,1]
        #     colors[i[0]][i[1]][i[2]][2] = vox_face_col[ii,2]
        # #colors[..., 0] = vox_face_col[:,0]
        # #colors[..., 1] = vox_face_col[:,1]
        # #colors[..., 2] = vox_face_col[:,2]


    def plot3d(self, draw_vox = True, draw_pts = True, pts ='coords',
                pts_col = None, prt_to_scr = True, data_type = data_type,
                vox_face_col = '#555efc', vox_edge_col = '#151738', vox_alpha = 0.1,
                vox_line_width = 0.1, vox_line_style = ':', ax_equal = True,
                pts_face_col = '#ffffff', pts_alpha = 0.6,  pts_size = 0.1,
                fig = None, subplot_nr = None, vox_depth = None, pts_depth = None,
                tick_nr = 8):
        '''Build a plot-instance for voxels and or x,y,z points within the space.
        Only voxels / points that are marked in self.mask/ self.tile_mask are
        drawn. Figure can be parsed to another function through 'pts_to_scr' = False.
        '''
        from mpl_toolkits.mplot3d import Axes3D
        if fig is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        elif fig is not None and subplot_nr is not None: # compatability with plotting functions
            ax = fig.add_subplot(subplot_nr, projection='3d')
            # ax.set_xlim(0,(10 * 1.5))
            # ax.set_ylim(0,(10 * 1.15))
            # ax.set_zlim(0,(10)) # / 1.5))
        #ax.pbaspect = [tile.size[0], tile.size[1], tile.size[2]]
        # SCALING OF AXIS (mimiking ax.set_aspect('equal'))
        if ax_equal == True:
            x_scale= self.size[0] / np.max(self.size)
            y_scale= self.size[1] / np.max(self.size)
            z_scale= self.size[2] / np.max(self.size)
            scale=np.diag([x_scale, y_scale, z_scale, 1.0])
            scale=scale*(1.0/scale.max())
            scale[3,3]=1.0
            def short_proj():
              return np.dot(Axes3D.get_proj(ax), scale)
            ax.get_proj=short_proj
            ax.set_xticks( np.linspace(0, self.size[0], tick_nr, dtype=int) )
            ax.set_yticks( np.linspace(0, self.size[1], tick_nr, dtype=int) )
            ax.set_zticks( np.linspace(0, self.size[2], tick_nr, dtype=int) )
        #
        if draw_vox is True: # voxel plot for voxels
            space3d, x, y, z, idx, idx_size = self.create3dGrid(depth = vox_depth)
            if len(vox_face_col.shape) > 1:
                #pass
                #mask = np.isnan(vox_face_col) # adjust if values are exluced in prior steps (np.nan)
                vox_face_col = self.create3dColors(vox_face_col, space3d, idx, map_to_cmap = False, normalize = False)
            elif type(vox_face_col) is not str:
                mask = np.isnan(vox_face_col) # adjust if values are exluced in prior steps (np.nan)
                vox_face_col = self.create3dColors(vox_face_col[~mask], space3d, idx[~mask])
                mask3d = np.isnan(vox_face_col) #which entries need to be excluded
                mask3d = mask3d[:,:,:,0] # exclude dimensionality information about
                space3d[mask3d] = 0.0
                #mask = numpy.full(space3d.shape, False, dtype=bool)
            if type(vox_edge_col) is not str:
                #idx = idx[~np.isnan(vox_edge_col)]
                #vox_edge_col = self.create3dColors(vox_edge_col[~mask], space3d, idx[~mask])
                #vox_edge_col = vox_edge_col * 0.6
                vox_edge_col = vox_face_col * 0.6
            ax.voxels( x, y, z,
                    filled = space3d,
                    facecolors = vox_face_col, #vox_face_col ,
                    edgecolors = vox_edge_col, #'#bbbeea',  # brighter
                    linewidth = vox_line_width,
                    alpha = vox_alpha,
                    linestyle= vox_line_style,
                    shade = False)
        #
        if draw_pts is True: # scatterplots for points
            if pts_col in data_type:
                if vox_depth is not None:
                    pts_face_col = self.extractFromDepth(content = pts_col) # depth_threshold = vox_depth
                else:
                    pts_face_col = self.extractDataFromTileSpace(content = pts_col)
            if draw_vox is False: # need to draw correct axis dimensions, non visible points
                ext_pts = np.vstack((self.coordOrg, (self.coordOrg + self.size) ))
                ax.scatter(ext_pts[:,0], ext_pts[:,1], ext_pts[:,2], alpha = 0)
            if pts_depth is not None:
                plt_coords = self.extractFromDepth(content = pts, depth_threshold = pts_depth)
            else:
                plt_coords = self.extractDataFromTileSpace(content = pts)
            # Plot
            ax.scatter(plt_coords[:,0], plt_coords[:,1], plt_coords[:,2],
                        marker = '.' , alpha = pts_alpha, c=pts_face_col,
                        s = pts_size, linewidths = 0, cmap = 'viridis') #,
                        #vmax = np.nanpercentile(pts_face_col, 97.5),
                        #vmin = np.nanpercentile(pts_face_col, 2.5) )
            #
        # if draw_data is True: # scatterplots for data aggregated for voxel
        #     if data_col in data_type:
        #         if vox_depth is not None:
        #             data_face_col = self.extractFromDepth(content = data_col) # depth_threshold = vox_depth
        #         else:
        #             data_face_col = self.extractDataFromTileSpace(content = data_col)
        #     if draw_vox is False: # need to draw correct axis dimensions, non visible points
        #         ext_pts = np.vstack((self.coordOrg, (self.coordOrg + self.size) ))
        #         ax.scatter(ext_pts[:,0], ext_pts[:,1], ext_pts[:,2], alpha = 0)
        #     if pts_depth is not None:
        #         plt_coords = self.extractFromDepth(content = 'center', depth_threshold = data_depth)
        #     else:
        #         plt_coords = self.extractDataFromTileSpace(content = 'center')
        #     ax.scatter(plt_coords[:,0], plt_coords[:,1], plt_coords[:,2],
        #                 marker = '.' , alpha = data_alpha, c=data_face_col,
        #                 s = pts_size, linewidths = 0, cmap = 'Greys_r')

        #
        #ax.set(xlabel='x', ylabel='y', zlabel='z')
        #ax.set_zticks([0, 5, self.size[2]])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # tick size
        ax.xaxis._axinfo['tick']['inward_factor'] = 0.05
        ax.yaxis._axinfo['tick']['inward_factor'] = 0.05
        ax.zaxis._axinfo['tick']['inward_factor'] = 0.05
        # grid line width
        ax.xaxis._axinfo['grid']['linewidth'] = 0.05
        ax.yaxis._axinfo['grid']['linewidth'] = 0.05
        ax.zaxis._axinfo['grid']['linewidth'] = 0.05
        #ax.zaxis._axinfo['tick']['color']='r'
        #ax.zaxis._axinfo['tick']['line_wdith'] = 0.2
        #ax.tick_params(outward_factor = 0.05)
        if prt_to_scr is True:
            plt.show()
        else:
            return(ax) # for further use in other plot function



    def newTile(self, tile_coord_org, size, tile_vsize = None, init_voxel = True):
        '''Creates a voxelSpaceTile instance.'''
        if tile_vsize is None:
            tile_vsize = self.vsize
        tile = voxelSpaceTile(tile_coord_org, size, tile_vsize,
                            voxelSpace_grid = self.grid, init_voxel = init_voxel)
        return(tile)



    def useHeaderLas(self, path):
        '''Extract reusable header data from .las file'''
        import copy
        lasFile = laspy.file.File(path, mode = 'r')
        header = copy.copy(lasFile.header)
        lasFile.close()
        return(header)



    def createHeaderLas(self, scale = np.array([0.01, 0.01, 0.01]), \
                        format = 1.3, data_format_id = 4):
        '''Builds a laspy.header instance with multiple user defined entry
        slots. Or copies the header from the provided file (path) for the newly created
        file into which voxel data will be poured.'''
        header = laspy.header.Header(file_version = format, point_format = data_format_id)
        #header.format = format
        #header.data_format_id = data_format_id # number/ type of user-defined header records in the header
        header.scale = scale # https://pythonhosted.org/laspy/header.html#laspy.header.HeaderManager.scale
        header.offset = ([self.coordOrg[0], self.coordOrg[1], self.coordOrg[2]])
        # header format 6 works only in las version 1.4
        return(header)



    def getHeaderInfoLas(self, path=None):
        '''Returns a list of header attributes for the file provided at
        self.las_file or at file path.'''
        if path is None:
            f = self.las_file
        else:
            f = laspy.file.File(path, mode = 'r')
        name = []
        values = []
        for func_name in dir(f.header):
            if 'get_' in func_name: # returns only attributes for which a get method exists
                func = getattr(laspy.header.HeaderManager, func_name)
                try:
                    val = func(f)
                    name.append(func_name)
                    values.append(val)
                except:
                    pass
        results = []
        for iter in list(zip(name, values)):
            results.append(iter)
        return(results)



    def createLas(self, path,   # scale = np.array([0,0,0]), #
                    header = None, header_file_path = None, vlrs = None,
                    close = True, data_size = None ):
        '''Creates a new .las file for the created path. utilises a header from a
        specified file if path is provided'''
        # Create File
        if header == None:
            if header_file_path == None:
                header = self.createHeaderLas()
            else:
                header = self.useHeaderLas(header_file_path)
        lasFile = laspy.file.File(path, mode = 'w', header = header, vlrs = vlrs) # creates the blanco file
        # Adjust Header Properties
        header = lasFile.header
        header.set_systemid('LasWave by Phil Jordan\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') # sets systems identifiyer
        header.set_softwareid('LasWave V-0.01 (unlicensed)\x00\x00\x00\x00\x00')
        header.set_filesourceid(0) # equal to ID has not been assigned
        if float(header.get_version()) < 1.3:
            header.set_version(1.3)
        # Store in memory or close
        if close == True:
            lasFile.close()
        else:
            self.las_file = lasFile



    def openLas(self, path):
        '''Opens or if not existend creates a .las file to pour voxel data into.'''
        import os.path
        if os.path.isfile(path):
            #header = self.useHeaderLas(path)
            self.las_file = laspy.file.File( path, mode = "rw")# , header = header)
            self.las_path = path
        else:
            print ("No file at " + path)
            # print('creating new .las file')
            # self.createLas(path)
            # header = self.useHeaderLas(path)
            # self.las_file = laspy.file.File( path, mode = "w", header = header)
            # self.las_path = path



    def checkOpenLas(self, path = None, openLas = False):
        '''Verifies that a laspy.file.File object exits and is open within
        the voxelSpace object. Open file at self.las_path if openLas
        is flagged 'True'. '''
        if isinstance(self.las_file, laspy.file.File):
            return(True)
        else:
            if openLas == False:
                return(False)
            elif openLas == True:
                if path == None:
                    if self.las_path == None:
                        print('no path for lasfile')
                        return(False)
                self.openLas(path = path)
                if self.checkOpenLas(): # if fileinstance can now be verified
                    return(True)
                else:
                    print('lasfile could not be opened')
                    return(False)



    def closeLas(self):
        '''Closes the open .las instance. Empty .las cannot be written out.'''
        self.las_file.close()



    def buildDimensionSpecLas(self, dim_name, dim_byte_type, dim_byte_size):
        '''Creates the laspy Dimensions specs for use in the VLR body.'''
        extra_dim_spec = laspy.header.ExtraBytesStruct(name = dim_name,
                                                       data_type = dim_byte_type)
        return(extra_dim_spec, dim_byte_size)
        # NEEDS ALSO TO RETURN THE OVERALL DIM_BYTE_SIZE



    def buildAllDimensionSpecLas(self, content, byte_flag = True):
        '''Creates all dimensions specs from a list of variables. Returns a
        List of dimension specs or one bytestring to be used in
        writeDimensionsIntoLas.'''
        vlr_body = []
        dim_byte_size = 0
        for count, dim_name in enumerate(content):
            if dim_name == 'center':
                dim = self.buildDimensionSpecLas(dim_name = 'center', dim_byte_type = 30, dim_byte_size = 24) # Double[3]
                #dim_1 = self.buildDimensionLas(dim_name = 'center_x', dim_byte_type = , dim_byte_size =)
                #dim_2 = self.buildDimensionLas(dim_name = 'center_y', dim_byte_type = , dim_byte_size =)
                #dim_3 = self.buildDimensionLas(dim_name = 'center_z', dim_byte_type = , dim_byte_size =)
            elif dim_name == 'coord_count':
                dim = self.buildDimensionSpecLas(dim_name = 'coord_count', dim_byte_type = 3, dim_byte_size = 2) # Short
            elif dim_name == 'intensity_sum':
                dim = self.buildDimensionSpecLas(dim_name = 'intensity_sum', dim_byte_type = 6, dim_byte_size = 4) # Long
            elif dim_name == 'intensity_mean':
                dim = self.buildDimensionSpecLas(dim_name = 'intensity_mean', dim_byte_type = 3, dim_byte_size = 2) # Short
            elif dim_name == 'intensity_var':
                dim = self.buildDimensionSpecLas(dim_name = 'intensity_var', dim_byte_type = 3, dim_byte_size = 2) # Short
            elif dim_name == 'intensity_mode':
                dim = self.buildDimensionSpecLas(dim_name = 'intensity_mode', dim_byte_type = 3, dim_byte_size = 2) # Short
            elif dim_name == 'intensity_skew':
                dim = self.buildDimensionSpecLas(dim_name = 'intensity_skew', dim_byte_type = 3, dim_byte_size = 2) # Short
            elif dim_name == 'volume':
                dim = self.buildDimensionSpecLas(dim_name = 'volume', dim_byte_type = 3, dim_byte_size = 2) # Short
            else:
                continue
            if byte_flag == True:
                if count == 0:
                    vlr_body = dim[0].to_byte_string()
                else:
                    vlr_body = vlr_body + dim[0].to_byte_string()
            elif byte_flag == False:
                vlr_body.append([dim[0]]) # will return instances of dimensionspec objects in a list
            dim_byte_size = (dim_byte_size + dim[1])
        return(vlr_body, dim_byte_size)



    def createExtraDimensionLas(self , path, content, header_path, data_size = None, close = True):
        '''Dimension / VLR builder for extra dimensions. This needs to be done
        before points are added to the file in order to set up the point_format
        correctly.
        Creates a new .las file with added dimeions, creates a blanco
        header instance for the file.'''
        #dim_new_header = self.createHeaderLas(format = 1.4, data_format_id = 6)
        dim_header = self.useHeaderLas(path = header_path)
        vlr_body, dim_byte_size = self.buildAllDimensionSpecLas(content = content,
                                                                byte_flag = True)
        extra_dim_vlr = laspy.header.VLR(user_id = "LASF_Spec",
                                 record_id = 4,
                                 description = "houses extra dimensions",
                                 VLR_body = vlr_body)
        dim_header.data_record_length += dim_byte_size
        # extra_dim_vlr.extr_dimensions[0].set_options = 2
        #las_file =
        self.createLas(path = path, header = dim_header,
                        vlrs = [extra_dim_vlr], data_size = data_size,
                        close = close)
        #return(las_file)
        # dir(extra_dim_vlr.extra_dimensions[0])



    # def extractDataForLas(self, lasPath = None, content = ['center', 'coord_count']):
    #     '''Extracts all data from voxels contained within and prepares it
    #     so it can be written into a .las. Returns the center points of voxels
    #     and the corresponding data.'''
    #     if isinstance(self.las_file, laspy.file.File):
    #         pass
    #     else:
    #         self.openLas(lasPath)
    #         data = self.getFromAllVoxels(content = 'center')
    #         data = self.getFromAllVoxelContents(content = 'coord_count')
    #         data = self.getFromAllVoxelContents(content = 'intensity_sum')
    #         data = self.getFromAllVoxelContents(content = 'intensity_mean')



    def writeToDisk(self, dir, content, header_path, project_id = None, format ='las', depth=0):
        '''Writes file attributes to disk. Preferably into a .las file (.laz
        currently not implemented). Project ID can be adjusted by providing an
        uuid.UUID object through project_id. Intensity value is set to the
        intensity_mean to enable voxel representation on higher depth levels.'''
        # build path for out_file
        path = self.createFilePath(dir = dir, value_name = None, format_name = format)
        # build empty .las
        self.createExtraDimensionLas(path = path, header_path = header_path, content = content, close = False)
        offset = self.las_file.header.get_offset()
        scale = self.las_file.header.get_scale()
        if isinstance(project_id, uuid.UUID):
            self.las_file.header.set_guid(project_id)
        for count, type in enumerate(content):
            # extract relevant data
            if depth == 0 or depth == None:
                data = self.extractDataFromTileSpace(content = type)
            else:
                data = self.extractFromDepth(depth_threshold = depth, content = type)
            # write data into file
            if type == 'center':
                    self.las_file.set_x_scaled(data[:,0] + offset[0]),
                    self.las_file.set_y_scaled(data[:,1] + offset[1]),
                    self.las_file.set_z_scaled(data[:,2] + offset[2])
                    #self.las_file.set_x(data[:,0]),
                    #self.las_file.set_y(data[:,1]),
                    #self.las_file.set_z(data[:,2])
            elif type == 'intensity_mean':
                    self.las_file.set_intensity(data.astype(int))
            else:
                    self.las_file._writer.set_dimension(type, data)
        # save changes to disk
        #self.las_file.header.update_min_max()
        self.las_file.close()



    def writeTilesToDisk(self, tile_depth, resolution_depth, dir, content,
                            header_path = None, project_id = None, ):
        '''Write out all Tile Data to Disk at specific depth level (tile_depth).
        Voxel size of data that is to be written corresponds (i.e. is set) ot the
        voxel size at 'resolution_depth' .'''
        if tile_depth > 0:
            for tile in self.tspace[self.tile_mask.astype(np.bool)]:
                tile.writeTilesToDisk(tile_depth = (tile_depth-1),
                                    resolution_depth = (resolution_depth-1),
                                    dir= dir, content = content,
                                    header_path = header_path,
                                    project_id = project_id )
        elif tile_depth == 0:
            self.writeToDisk(dir = dir, content = content,
                            header_path = header_path,
                            project_id = project_id, depth= resolution_depth)



    # Legacy name visualizeFileLas
    def visualizeData(self, pts_var, *col_var, path = None, update_col = False,
                            pts_size = 0.05 , col_map = 'hot', scale = None):
        '''Visualizes the lasfile data through pptk package. Requires a xyz point
        array and one or multiple color array instances. Either these instance
        arrays can be provided, extracted from the voxelSpace or extracted from
        a .las file.'''
        import pptk
        # get data from voxelspace
        if path is 'voxelSpace':
            pts = self.dataFromAllVoxels(pts_var)
            col = self.dataFromAllVoxels(col_var)
        # load file to memory
        elif path is not None:
            if path is not 'self.las_file':
                self.openLas(path)
                # get data from lasfile object in memory
            x = self.las_file.get_x_scaled()
            y = self.las_file.get_y_scaled()
            z = self.las_file.get_z_scaled()
            pts = np.vstack((x,y,z)).T
            col = self.las_file._writer.get_dimension(col_var)
        # if no path is specified instances of both points and color
        # *args must be provided
        else:
            pts = pts_var
        v = pptk.viewer(pts)
        v.attributes(*col_var)
        if scale == None:
            v.color_map(col_map)
        else:
            v.color_map(col_map, scale)
        v.set(point_size=pts_size)
        v.wait() #v.clear() #v.close()































class voxelSpaceTile(voxelSpace):
    '''Subclass of voxelspace. Used to chunk up large data sizes.'''

    def __init__(self, tile_coord_org, size, vsize, voxelSpace_grid = None, init_voxel=True):
        super().__init__(width=size[0], depth=size[1],
                        height=size[2], vsize= vsize,
                        coordOrg = tile_coord_org, init_voxel=init_voxel )
        self.voxelSpace_grid = voxelSpace_grid
        #del(self.size) # just to avoid confusion which size is which
        #self.size = size
        #self.type = 'voxelSpaceTile'
        #self.grid = self.buildTileGrid()
        #self.id = voxelSpace.createIdentifier(self)
        #self.vspace = voxelSpace.vspaceFromGrid(self)


    def getVolume(self):
        return(np.prod(self.size))



    def buildTileGrid(self):
        '''Creates a grid for the tile, depending on its extent. Grid steps are
        based on grid steps of superclass voxelSpace.'''
        X,Y,Z = np.mgrid[self.coordOrg[0]:self.coordOrg[0]+self.size[0]:self.vsize[0],
                        self.coordOrg[1]:self.coordOrg[1]+self.size[1]:self.vsize[1],
                        self.coordOrg[2]:self.coordOrg[2]+self.size[2]:self.vsize[2],    ]
        grid = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        return(grid)



    def extractTileData(self, in_data, return_mask = False):
        '''returns the subdata contained within a specific tile.
        Currently only works with coordinates. For Intensity values pls
        use return_mask option.'''
        # in_bounds = self.assertTileInBounds()
        # if in_bounds == True:
        #     pass
        # elif in_bounds == False:
        #     self.bringTileInBounds()
        #     print('reshaping tile ' + str(self.coordOrg) + ' to ' + str(self.size))
        # elif in_bounds == None:
        #     print('could not place tile origin ' + str(self.coordOrg) + ' in grid' )
        #     return None
            #return(np.array([]))
        if len(in_data.shape) == 2: # input data with shape (X, Y)
            mask =  np.all(in_data >= self.coordOrg, axis=1) * \
                    np.all(in_data < (self.coordOrg + self.size), axis=1)
        else:
            print('input data has wrong shape.')
            return None
        #
        if np.sum(mask) == 0:
            print('No data found for Tile ' + str(self.coordOrg))
            return(np.array([]))
        if return_mask == True:
            return(mask)
        else:
            return(in_data[mask])



    def assertTileInBounds(self, modify_size = None):
        '''Checks if the tile is within the bounds of the grid, checks for
        the coordinate origin and an additional size (default is tile size)'''
        if modify_size == None:
            size = self.size
        else:
            size = modify_size
        if np.any(self.coordOrg > self.voxelSpace_grid[-1]):
            print('Tile out of bound for position (inner Position) ' + str(self.coordOrg) )
            return(None)
        elif np.any((self.coordOrg + self.size) > self.voxelSpace_grid[-1]):
            return(False)
        elif np.all((self.coordOrg + self.size) <= self.voxelSpace_grid[-1]):
            return(True)



    def bringTileInBounds(self):
        '''Adjusts the borders for a tile so it will sit on the edge of the grid.'''
        for pos, dim in enumerate( np.where(self.coordOrg < self.voxelSpace_grid[-1])[0] ): # verefies tile coordOrg lies within boundary of voxelspace grid
            while (self.coordOrg[pos] + self.size[pos]) > self.voxelSpace_grid[-1][pos] :
                #if self.coordOrg[pos] < self.voxelSpace_grid[-1][pos]:
                self.size[pos] -= self.vsize[pos] #[pos] currently only has one dimension
            #while self.assertTileInBounds(self.coordOrg, modify_size = self.vsize) == False:



    def maskForTile(self):
        '''returns the extend of the tile as a grid and as a voxelspace'''
        #lower = np.where(np.all(self.voxelSpace_grid == self.coordOrg, axis=1))[0][0]
        #lower = self.voxelSpace_grid[lower]
        #upper = np.where(np.all(self.voxelSpace_grid == (self.coordOrg + self.size), axis = 1 ))[0][0]
        #upper = self.voxelSpace_grid[upper]
        lower = self.coordOrg
        upper = self.coordOrg + self.size
        mask = np.all(self.voxelSpace_grid >= lower, axis=1) * \
                np.all(self.voxelSpace_grid < upper, axis=1)
        #sub_grid = self.voxelSpace_grid[mask]
        #sub_vspace = self.vspace[mask]
        return(mask)
