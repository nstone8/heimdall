import numpy as np
import skimage.draw
import shapely.geometry
import heimdall.improcessing as imp
import scipy
from matplotlib import pyplot as plt
from skimage.morphology import disk

class RidgeSpec:
    '''Class representing the geometry of devices consisting of regularly spaced, identical angled ridges'''
    def __init__(self,ridge_height:float,ridge_width:float,ridge_spacing:float,ridge_angle:float):
        '''Create a new RidgeSpec object

        -----Parameters-----
        ridge_height: The height of the ridges measured perpendicular to the channel
        ridge_width: The width of the ridges measured parallel to the channel
        ridge_spacing: The pitch of the ridges measured parallel to the channel
        ridge_angle: The angle of the ridges, measured relative to the long axis of the channel

        -----Returns-----
        ridge: A new RidgeSpec object'''
        self.height=ridge_height
        self.width=ridge_width
        self.spacing=ridge_spacing
        self.angle=ridge_angle

    def get_mask(self,num_ridge:int,scale:float,tilt:float,offset_y:float,offset_x:float,im_shape:tuple,im_dtype=np.bool):
        '''Get a binary mask showing the ridge geometry transformed into image coordinates

        -----Parameters-----
        num_ridge: Number of ridges
        scale: Microns to pixels conversion factor
        tilt: Angle to rotate device geometry by
        offset_y: Y offset in pixels
        offset_x: X offset in pixels
        im_shape: Tuple giving the desired shape of the returned mask
        im_dtype: Desired dtype of the returned mask

        -----Returns-----
        mask: a binary image with the ridge geometry represented by one pixel wide positive lines'''

        mask=np.zeros(im_shape,im_dtype)
        all_coords=self.get_ridge_coords(num_ridge,scale,tilt,offset_y,offset_x)
        #generate mask
        for ridge in all_coords:
            ridge_x=[r[0] for r in ridge]
            ridge_y=[r[1] for r in ridge]
            rr,cc=skimage.draw.polygon_perimeter(ridge_y,ridge_x,im_shape)
            mask[rr,cc]=True

        return mask

    def get_ridge_coords(self,num_ridge:int,scale:float,tilt:float,offset_y:float,offset_x:float):
        '''Get the coordinates of the ridge vertices transformed into image coordinates

        -----Parameters-----
        num_ridge: Number of ridges
        scale: Microns to pixels conversion factor
        tilt: Angle to rotate device geometry by
        offset_y: Y offset in pixels
        offset_x: X offset in pixels

        -----Returns-----
        all_coords: A list where each entry is a list of tuple coordinates for the vertices of each ridge
        '''

        starting_coords=((0,0),(-self.width*scale,0),((-1*scale*self.height/np.tan(self.angle))-self.width*scale,self.height*scale),(-1*scale*self.height/np.tan(self.angle),self.height*scale))
        all_coords_no_tilt_no_trans=[]
        for i in range(num_ridge):
            all_coords_no_tilt_no_trans.append([(x-(i*self.spacing*scale),y) for x,y in starting_coords])

        all_coords_no_trans=[[rotate_point(x,y,tilt) for x,y in ridge] for ridge in all_coords_no_tilt_no_trans]

        all_coords=[[(x+offset_x,y+offset_y) for x,y in ridge] for ridge in all_coords_no_trans]
        return all_coords

    def detect_ridges(self,background:np.ndarray,obj_percentile=100,debug:bool=False):
        device=self
        '''Detect ridge location in an image
        -----Parameters-----
        background: Image to detect ridges in. Usually produced by heimdall.improcessing.get_background()
        device: A heimdall.device.RidgeSpec defining the geometry of the device pictured in background

        -----Returns-----
        cal_device: A heimdall.device.CalibratedDevice representing the detected transformation between image and device coordinates
        '''
        sobel=skimage.filters.sobel(background)
        thresh=skimage.filters.threshold_otsu(sobel)
        ridges=sobel>thresh
        #get y,x coordinates of corners
        ridges_closed=skimage.morphology.binary_closing(ridges,disk(2))
        ridges_skeleton=skimage.morphology.skeletonize(ridges_closed)
        seg=skimage.measure.label(ridges_closed+1,connectivity=1)
        #remove all objects touching edges
        edge_obj=[]
        edge_obj.extend(seg[0,:])
        edge_obj.extend(seg[:,0])
        edge_obj.extend(seg[-1,:])
        edge_obj.extend(seg[:,-1])

        for obj in set(edge_obj):
            seg[seg==obj]=0

        seg=skimage.measure.label(seg>0,connectivity=1)
        seg=remove_edge_objs(seg)

        filter_small_objs(seg,obj_percentile)
        skimage.io.imshow(skimage.color.label2rgb(seg))
        skimage.io.show()
        optimized_params,corners=find_ridges_in_seg_im(seg,device,ridges_skeleton)

        if debug:
            device_mask=device.get_mask(*optimized_params)
            fig,ax=plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
            ax=[a for row in ax for a in row]
            bg_overlay=background.copy()
            bg_overlay[device_mask>0]=0
            ax[0].imshow(bg_overlay)
            ax[1].imshow(ridges_skeleton)
            ax[2].imshow(ridges_closed)
            ax[3].imshow(skimage.color.label2rgb(seg))
            corners_y=[p[0] for pair in corners for p in pair]
            corners_x=[p[1] for pair in corners for p in pair]
            ax[3].plot(corners_x,corners_y,'k+',markersize=15)
            plt.show()

        return CalibratedDevice(device,*optimized_params[0:-1])

class RidgeSpecSemiGutter(RidgeSpec):
    def detect_ridges(self,background:np.ndarray,obj_percentile=100,debug:bool=False):
        device=self
        '''Detect ridge location in an image
        -----Parameters-----
        background: Image to detect ridges in. Usually produced by heimdall.improcessing.get_background()
        device: A heimdall.device.RidgeSpec defining the geometry of the device pictured in background

        -----Returns-----
        cal_device: A heimdall.device.CalibratedDevice representing the detected transformation between image and device coordinates
        '''
        sobel=skimage.filters.sobel(background)
        thresh=skimage.filters.threshold_otsu(sobel)
        ridges=sobel>thresh
        #get y,x coordinates of corners
        ridges_closed=skimage.morphology.binary_closing(ridges,disk(2))
        ridges_skeleton=skimage.morphology.skeletonize(ridges_closed)
        seg=skimage.measure.label(ridges_closed+1,connectivity=1)
        #Find object with max perimeter/area ratio out of objects with perimeters comparable to the image size, should be our ridges

        seg_props=skimage.measure.regionprops(seg)
        max_ratio=-1
        max_ratio_label=-1
        for props in seg_props:
            if props.perimeter<max(seg.shape):
                continue
            this_ratio=props.perimeter/props.area
            if this_ratio>max_ratio:
                max_ratio=this_ratio
                max_ratio_label=props.label
        bounds=skimage.segmentation.find_boundaries(seg==max_ratio_label)
        #find the channel area
        bounds_seg=skimage.measure.label(bounds+1,connectivity=1)
        bounds_seg_props=skimage.measure.regionprops(bounds_seg)
        max_area=-1
        max_area_label=-1
        for props in bounds_seg_props:
            if props.area>max_area:
                max_area=props.area
                max_area_label=props.label
        channel=bounds_seg==max_area_label

        #now we want to work on the boundary of this image
        final_bounds=skimage.segmentation.find_boundaries(channel)

        #now test all possible lines drawn through left and right side of the image to find the one which overlaps final_bounds the best
        best_sum=-1
        best_left=-1
        best_right=-1
        for left in range(final_bounds.shape[0]):
            for right in range(final_bounds.shape[0]):
                #get coordinates on this line
                rr,cc=skimage.draw.line(left,0,right,final_bounds.shape[1]-1)
                this_sum=np.sum(final_bounds[rr,cc])
                if this_sum>best_sum:
                    best_sum=this_sum
                    best_left=left
                    best_right=right
        rr,cc=skimage.draw.line(best_left,0,best_right,final_bounds.shape[1]-1)
        final_bounds[rr,cc]=1

        seg=skimage.measure.label(final_bounds+1,connectivity=1)
        seg=remove_edge_objs(seg)

        filter_small_objs(seg,obj_percentile)
        skimage.io.imshow(skimage.color.label2rgb(seg))
        skimage.io.show()

        optimized_params,corners=find_ridges_in_seg_im(seg,device,ridges_skeleton)

        if debug:
            device_mask=device.get_mask(*optimized_params)
            fig,ax=plt.subplots(nrows=2,ncols=3,sharex=True,sharey=True)
            ax=[a for row in ax for a in row]
            bg_overlay=background.copy()
            bg_overlay[device_mask>0]=0
            ax[0].imshow(bg_overlay)
            ax[1].imshow(ridges_skeleton)
            ax[2].imshow(final_bounds)
            ax[3].imshow(ridges_closed)
            ax[4].imshow(skimage.color.label2rgb(seg))
            ax[5].axis('off')
            corners_y=[p[0] for pair in corners for p in pair]
            corners_x=[p[1] for pair in corners for p in pair]
            ax[4].plot(corners_x,corners_y,'k+',markersize=15)
            plt.show()

        return CalibratedDevice(device,*optimized_params[0:-1])

    def get_mask(self,num_ridge:int,scale:float,tilt:float,offset_y:float,offset_x:float,im_shape:tuple,im_dtype=np.bool):
        '''Get a binary mask showing the ridge geometry transformed into image coordinates

        -----Parameters-----
        num_ridge: Number of ridges
        scale: Microns to pixels conversion factor
        tilt: Angle to rotate device geometry by
        offset_y: Y offset in pixels
        offset_x: X offset in pixels
        im_shape: Tuple giving the desired shape of the returned mask
        im_dtype: Desired dtype of the returned mask

        -----Returns-----
        mask: a binary image with the ridge geometry represented by one pixel wide positive lines'''

        mask=np.zeros(im_shape,im_dtype)
        all_coords=self.get_ridge_coords(num_ridge,scale,tilt,offset_y,offset_x)
        #generate mask
        for ridge in all_coords:
            ridge_x=[int(round(r[0])) for r in ridge]
            ridge_y=[int(round(r[1])) for r in ridge]
            lines_coords=[]
            lines_coords.append(skimage.draw.line(ridge_y[0],ridge_x[0],ridge_y[1],ridge_x[1]))
            lines_coords.append(skimage.draw.line(ridge_y[1],ridge_x[1],ridge_y[2],ridge_x[2]))
            lines_coords.append(skimage.draw.line(ridge_y[3],ridge_x[3],ridge_y[0],ridge_x[0]))
            #trim parts of ridge that fall off the edge of the mask
            rrr=[]
            ccc=[]
            for rr,cc in lines_coords:
                for r,c in zip(rr,cc):
                    if not ((r>=im_shape[0]) or (c>=im_shape[1])):
                        rrr.extend([r])
                        ccc.extend([c])

            mask[rrr,ccc]=True

        return mask

class CalibratedDevice:
    '''A class representing a sorting device calibrated to a specific video'''
    def __init__(self,ridge_spec:'RidgeSpec',num_ridge:int,scale:float,tilt:float,offset_y:float,offset_x:float):
        '''Create a new CalibratedDevice object

        -----Parameters-----
        ridge_spec: RidgeSpec object representing the type of device present in this video
        scale: Microns to pixels conversion factor for this video
        tilt: Angle to rotate device geometry by for this video
        offset_y: Y offset in pixels for this video
        offset_x: X offset in pixels for this video

        -----Returns-----
        cal_device: A new CalibratedDevice object'''

        print('optimized tilt:',tilt)
        self.num_ridge=num_ridge
        self.scale=scale
        self.tilt=tilt
        self.offset_y=offset_y
        self.offset_x=offset_x
        self.ridge_spec=ridge_spec
        ridge_coords=ridge_spec.get_ridge_coords(num_ridge,1,0,0,0)
        #flip y axis to go from image to real coordinates
        for ridge in range(len(ridge_coords)):
            ridge_coords[ridge]=[(x,-y) for x,y in ridge_coords[ridge]]
        self.ridge_coords=ridge_coords
        self.ridge_poly=[shapely.geometry.Polygon(points) for points in self.ridge_coords]

    def get_ridge_sep(self):
        '''Get the ridge separation along the channel axis

        -----Parameters-----
        None

        -----Returns-----
        The ridge separation for this device along the channel axis'''

        return self.ridge_poly[0].distance(self.ridge_poly[1])

    def point_under_ridge(self,point:tuple):
        '''Test if a location is under a ridge

        -----Parameters-----
        point: Tuple containing (y,x) coordinates to test for being under a ridge

        -----Returns-----
        ridge: The index of the ridge this point is under or None if this point is not under a ridge'''

        point_point=shapely.geometry.Point([point[1],point[0]])
        for i in range(len(self.ridge_poly)):
            if self.ridge_poly[i].contains(point_point):
                return i
        return None

    def dist_to_next_ridge(self,point:tuple)->('dist','ridge_no'):
        '''Get the distance from a point to the next ridge and the identity of that ridge

        -----Parameters-----
        point: Tuple containing (y,x) coordinates

        -----Returns-----
        dist,ridge_no: Distance dist to the next ridge ridge_no'''

        point_point=shapely.geometry.Point([point[1],point[0]])
        ridge_dists=[r.distance(point_point) for r in self.ridge_poly]
        ridge_rings=[ridge.exterior for ridge in self.ridge_poly]
        ridge_closest_points=[ring.interpolate(ring.project(point_point)).coords[0] for ring in ridge_rings]
        num_ridges=len(self.ridge_coords)
        ridges_ahead=[]
        for i in range(num_ridges):
            if self.ridge_poly[i].contains(point_point):
                if i==(num_ridges-1):
                    return None,'after_last_ridge'
                #if we're under a ridge, the next ridge is us+1
                else:
                    return ridge_dists[i+1],i+1
            elif (ridge_closest_points[i][0]-point[1])<0:
                #if the vector to the closest point on a ring has a negative x component, it is in front of us
                ridges_ahead.append(i)
        min_dist_ahead=float('inf')
        next_ridge=None
        for j in ridges_ahead:
            if ridge_dists[j]<min_dist_ahead:
                min_dist_ahead=ridge_dists[j]
                next_ridge=j
        if next_ridge==None:
            return None,'after_last_ridge'
        else:
            return ridge_dists[next_ridge],next_ridge

    def point_in_gutter(self,point:tuple):
        '''Test if a point is in the gutter

        -----Parameters-----
        point: Tuple containing (y,x) coordinates

        -----Returns-----
        under_ridge: True if point is in the gutter, False otherwise'''

        if (point[0]>self.ridge_coords[0][0][1]) or (point[0]<self.ridge_coords[0][2][1]):
            return True
        else:
            return False

    def get_ridge_offsets(self,offset:float):
        '''Get shapely.LineStrings offset displaced upstream and downstream from the ridge

        -----Parameters-----
        offset: Distance by which to offset the returned shapely.LineStrings from the ridge geometry

        -----Returns-----
        out: shapely.LineStrings identical to the long edges of the ridge translated a distance offset upstream from the leading edge and offset downstream from the trailing edge'''

        out=[]
        for ridge in self.ridge_coords:
            upstream_vertices=ridge[0:1]+ridge[3:]
            downstream_vertices=ridge[1:3]
            upstream=shapely.geometry.LineString([[x+offset,y] for x,y in upstream_vertices])
            downstream=shapely.geometry.LineString([[x-offset,y] for x,y in downstream_vertices])
            out.append([upstream,downstream])
        return out

def rotate_point(x,y,angle)->(float,float):
    '''find the location of point (x,y) after undergoing a rotation of angle radians around (0,0)

    -----Parameters-----
    x: X coordinate of the point of interest
    y: Y coordinate of the point of interest
    angle: Amount of rotation in radians about (0,0) to subject (x,y) to

    -----Returns-----
    new_x,new_y: Coordinates of point of interest after rotation'''

    r=np.sqrt((x**2)+(y**2))
    start_angle=np.arctan2(y,x)
    new_x=r*np.cos(start_angle+angle)
    new_y=r*np.sin(start_angle+angle)
    return new_x,new_y

def maximize_overlap(ridge_mask:np.ndarray,device:RidgeSpec,initial_guess:list):
    '''Maximize the overlap between device.get_mask(*args) and ridge_mask

    -----Parameters-----
    ridge_mask: a numpy.ndarray containing the ridge skeleton mask
    device: a heimdall.device.RidgeSpec representing the device geometry being aligned
    initial_guess: list of args for device.get_mask() constituting a guess for correct alignment

    -----Returns-----
    res: a scipy.optimize.OptimizeResult object representing optimized alignment between device and ridge_mask
    '''
    def total_dist(source_mask:np.ndarray,device_mask:np.ndarray):
        '''Get the sum of squares of the distance between each positive point on device_mask to the (orthagonally) closest positive point on source_mask

        -----Parameters-----
        source_mask: A numpy.ndarray representing the 'target' geometry we are aligning to
        device_mask: A numpy.ndarray representing a guess for aligned geometry

        -----Returns-----
        total_distance: The sum of squares of the distance between each positive point on device_mask to the (orthagonally) closest positive point on source_mask
        '''
        total_distance=0
        #for each point on the device mask see how far (orthagonally) the nearest point on the source_mask is
        x_arr,y_arr=np.nonzero(device_mask)
        for point in zip(x_arr,y_arr):
            this_dist=-1
            while True:
                this_dist+=1
                try:
                    above=source_mask[point[0],point[1]+this_dist]
                    right=source_mask[point[0]+this_dist,point[1]]
                    below=source_mask[point[0],point[1]-this_dist]
                    left=source_mask[point[0]-this_dist,point[1]]
                except IndexError:
                    #we're trying to index off the end of the image
                    break
                if any((above,right,below,left)):
                    break
            total_distance+=this_dist**2
        return total_distance
    #optimize parameters other than num_ridges (initial_guess[0])
    res=scipy.optimize.minimize(lambda x: total_dist(ridge_mask,device.get_mask(initial_guess[0],*x,initial_guess[-1])),initial_guess[1:-1],method='Nelder-Mead')
    return res

def remove_edge_objs(im):
    #remove all objects touching edges
    edge_obj=[]
    edge_obj.extend(im[0,:])
    edge_obj.extend(im[:,0])
    edge_obj.extend(im[-1,:])
    edge_obj.extend(im[:,-1])

    for obj in set(edge_obj):
        im[im==obj]=0

    seg=skimage.measure.label(im>0,connectivity=1)
    return im

def filter_small_objs(seg,obj_percentile):
    #filter out objects smaller than the ridges
    obj_sizes=[(obj,seg[seg==obj].size) for obj in set(seg.ravel()) if obj>0]
    max_size=np.percentile([obj_s[1] for obj_s in obj_sizes],obj_percentile)
    for obj,size in obj_sizes:
        if size<0.8*max_size:
            #this object is smaller than the ridges, remove from seg
            seg[seg==obj]=0
def find_ridges_in_seg_im(seg,device,ridges_skeleton):
    #for each ridge, find the two points that are furthest apart, these will be our 'corners'
    #Would probably be faster to first find the point furthest from the centroid, then the point furthest from that point

    corners={}
    for ridge in set(seg[seg>0].ravel()):
        #I think the data is in (y,x) order
        this_ridge=np.zeros(seg.shape,seg.dtype)
        y_coords,x_coords=np.where(seg==ridge)
        this_ridge[y_coords,x_coords]=1
        ridge_centroid=imp.get_centroid(this_ridge)
        max_dist=0
        first_corner=None
        for a in zip(y_coords,x_coords):
            #find furthest point from centroid
            this_dist=np.sqrt(((ridge_centroid[0]-a[0])**2)+((ridge_centroid[1]-a[1])**2))
            if this_dist>max_dist:
                first_corner=a
                max_dist=this_dist

        max_dist=0
        second_corner=None
        for b in zip(y_coords,x_coords):
            #find furthest point from first_corner
            this_dist=np.sqrt(((first_corner[0]-b[0])**2)+((first_corner[1]-b[1])**2))
            if this_dist>max_dist:
                second_corner=b
                max_dist=this_dist

        corners[ridge]=[first_corner,second_corner]
    new_corners={}
    #make the 'top' corner first
    for r in corners:
        p1,p2=corners[r]
        new_corners[r]=([p1 if p1[0]<p2[0] else p2, p2 if p2[0]>p1[0] else p1])
    corners=new_corners

    #calculate ridge slope and reject objects that don't match the ridges
    abs_slopes=[abs((corners[r][0][0]-corners[r][1][0])/(corners[r][0][1]-corners[r][1][1])) for r in corners]
    med_slope=np.median(abs_slopes)
    new_corners={}
    for r,slope in zip(corners,abs_slopes):
        #if the slope is off by more than 20%, can it
        if not ((slope<(0.8*med_slope)) or (slope>(1.2*med_slope))):
            new_corners[r]=corners[r]
    corners=new_corners

    dists=[dist_corners(corners[r][0],corners[r][1]) for r in corners]
    #find minimum distance between top corners
    top_corners=[c[0] for c in corners.values()]
    min_dist=float('inf')
    for c1 in top_corners:
        for c2 in top_corners:
            if c1 is c2:
                continue
            if dist_corners(c1,c2)<min_dist:
                min_dist=dist_corners(c1,c2)
    #stitch together broken ridges
    close_ridges={}
    for r1 in corners:
        these_close_ridges=[]
        for r2 in corners:
            if r1==r2:
                continue
            this_dist=dist_corners(corners[r1][1],corners[r2][0])
            if this_dist<0.5*np.max(dists):
                #these corners might be the same ridge, check if they're on the same line
                if dist_line_to_point(corners[r1],corners[r2][0])<(0.5*min_dist):
                    these_close_ridges.append(r2)
        if these_close_ridges:
            close_ridges[r1]=these_close_ridges

    #for each entry with close corners find the closest and join the ridges
    ridges_to_join=[]
    for r1 in close_ridges:
        closest_ridge=-1
        closest_dist=float('inf')
        these_close_ridges=close_ridges[r1]
        for friend in these_close_ridges:
            this_dist=dist_corners(corners[r1][1],corners[friend][0])
            if this_dist<closest_dist:
                closest_dist=this_dist
                closest_ridge=friend
        ridges_to_join.append([r1,closest_ridge])

    #look for 'chains' of ridges to join and consolidate
    all_chains=set()
    for to_join in ridges_to_join:
        this_set=set(to_join)
        go=True
        while go:
            go=False
            new_this_set=set(this_set)
            for ridge in this_set:
                for to_join_2 in ridges_to_join:
                    if ridge in to_join_2:
                        new_this_set|=set(to_join_2)
            if new_this_set != this_set:
                go=True
            this_set=new_this_set
        all_chains|=set([frozenset(this_set)])
    in_chain={j for i in all_chains for j in i}
    #populate new_new_corners with ridges that are not about to be joined
    new_new_corners=[]
    for r in corners:
        if not r in in_chain:
            new_new_corners.append(corners[r])
    for chain in all_chains:
        chain=list(chain)
        #join up ridges, make them all the same ridge, point with max y is bottom corner, point with min y is top
        this_ridge=chain[0]
        this_corner=corners[this_ridge]
        for other_ridge in chain[1:]:
            seg[seg==other_ridge]=this_ridge
            if corners[other_ridge][0][0]<corners[this_ridge][0][0]:
                this_corner[0]=corners[other_ridge][0]
            if corners[other_ridge][1][0]>corners[this_ridge][1][0]:
                this_corner[1]=corners[other_ridge][1]
        new_new_corners.append(this_corner)
    corners=new_new_corners

    #Get 'top' corners and use them to estimate tilt, scale and number of ridges
    top_corners=[c[0] for c in corners]
    #fit a line to top corners to estimate tilt
    popt,pcov=scipy.optimize.curve_fit(lambda x,m,b:[m*this_x+b for this_x in x],[p[1] for p in top_corners],[p[0] for p in top_corners],[0,top_corners[0][0]])
    detected_slope=popt[0]
    detected_intercept=popt[1]
    detected_tilt=np.arctan(detected_slope)
    print('detected_tilt:',detected_tilt)

    #find closest distance between top corners to estimate ridge spacing and max distance to estimate number of ridges
    min_dist=float('inf')
    max_dist=-float('inf')
    for c1 in top_corners:
        for c2 in top_corners:
            this_dist=np.sqrt(((c1[0]-c2[0])**2)+((c1[1]-c2[1])**2))
            if this_dist>0:
                if this_dist>max_dist:
                    max_dist=this_dist
                elif this_dist<min_dist:
                    min_dist=this_dist

    detected_ridge_spacing=min_dist
    detected_num_ridges=int(np.round(max_dist/detected_ridge_spacing)+1)

    #find coordinate of top corner with largest x component to get estimate of x offset, get y from linear fit
    detected_x_offset=max([p[1] for p in top_corners])
    detected_y_offset=int(np.round(detected_slope*detected_x_offset+detected_intercept))

    #estimate scale factor from known device measurement
    detected_scale=detected_ridge_spacing/device.spacing
    optimized_result=maximize_overlap(ridges_skeleton,device,[detected_num_ridges,detected_scale,detected_tilt,detected_y_offset,detected_x_offset,ridges_skeleton.shape])
    #need to add in detected_num_ridges and background.shape as they were not considered in optimization
    optimized_params=[detected_num_ridges,*optimized_result.x,ridges_skeleton.shape]
    return optimized_params,corners

def dist_corners(c1,c2):
    return np.sqrt(((c1[0]-c2[0])**2)+((c1[1]-c2[1])**2))

def dist_line_to_point(points_defining_line,point):
    #find equation of line passing through points_defining_line
    (y1,x1),(y2,x2)=points_defining_line
    m=(y1-y2)/(x1-x2)
    b=y1-m*x1
    #now define a shapely.linestring that goes from the ridge to the y value of the point
    yline1=0
    xline1=(yline1-b)/m
    yline2=point[0]
    xline2=(yline2-b)/m
    line=shapely.geometry.LineString(((xline1,yline1),(xline2,yline2)))
    return line.distance(shapely.geometry.Point((point[1],point[0])))
