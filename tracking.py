import heimdall.improcessing as imp
import heimdall.device
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import skimage.draw

def calibrated_tracks_from_path(vid_path,device,cell_size,min_cell_size=None,num_frames=None,vid_flow_direction='left',num_bg=None,time_factor=10,max_dist_percentile=99,mem=None,debug=False):
    '''Get calibrated tracks from a video path

    -----Parameters-----
    vid_path: File path of the video to be processed
    device: RidgeSpec object representing the geometry of the device in the video
    cell_size: Nominal size of cells in the video, in microns
    min_cell_size: Minimum size of cells in the video, in microns. If None, this value defaults to 0.5*cell_size
    num_frames: Number of frames of the video stored at vid_path to process. If None, defaults to processing the entire video
    vid_flow_direction: Direction of cell flow in the video stored at vid_path. Must be one of 'up' or 'left' (the default)
    num_bg: Number of frames to use for backgound extraction. If None, defaults to the entire video.
    time_factor: Multiplier for converting from temporal to spatial units of distance. Values that are too high will make algorithm tend to link distant points that occurred at the same time. Values that are too small will make the algorithm tend to link close points that occurred at very different times.
    max_dist_percentile: Percentile of nearest neighbor distances to define as too distant to link. Smaller values require that points be closer together to link
    mem: Temporal distance (number of frames * time_factor) to search for links in path. If None, all points are considered.
    debug: Whether to produce debugging output during processing. Default is False.

    -----Returns-----
    cal_paths: A heimdall.tracking.CalibratedPaths object representing the paths detected in the video, transformed to device coordinates'''
    
    vid=imp.VidIterable(vid_path,num_frames=num_frames,vid_flow_direction=vid_flow_direction)
    bg=imp.get_background(vid,num_bg)
    cal_device=imp.detect_ridges(bg,device,debug)
    cell_size*=cal_device.scale
    if min_cell_size:
        min_cell_size*=cal_device.scale
    movers=imp.gen_movers(vid,bg,cell_size,min_cell_size,debug)
    tracks=get_tracks(movers,time_factor,max_dist_percentile,mem,debug)
    cal_paths=calibrate_paths(tracks,cal_device)
    return cal_paths

def get_tracks(movers:'Iterator',time_factor:int=10,max_dist_percentile:float=99,mem:float=None,debug:bool=False):
    '''Get tracks from an iterator of labelled masks

    -----Parameters-----
    movers: An iterator of labeled object masks. Usually created using heimdall.improcessing.gen_movers()
    time_factor: Multiplier for converting from temporal to spatial units of distance. Values that are too high will make algorithm tend to link distant points that occurred at the same time. Values that are too small will make the algorithm tend to link close points that occurred at very different times.
    max_dist_percentile: Percentile of nearest neighbor distances to define as too distant to link. Smaller values require that points be closer together to link
    mem: Temporal distance (number of frames * time_factor) to search for links in path. If None, all points are considered.
    debug: Whether to produce debugging output during processing. Default is False.

    -----Returns-----
    A list of heimdall.tracking.Path objects representing the object trajectories in movers'''
    
    time=0
    coords=[]
    for frame in movers:
        objs=[o for o in set(frame.ravel()) if o]
        for obj in objs:
            this_obj_mask=frame==obj
            this_obj_location=imp.get_centroid(this_obj_mask)
            this_obj_coords=(*this_obj_location,time*time_factor)
            coords.append(this_obj_coords)
        time+=1

    if debug:
        y_coords=[c[0] for c in coords]
        x_coords=[c[1] for c in coords]
        t_coords=[c[2] for c in coords]
        fig=plt.figure()
        ax=fig.add_subplot('111',projection='3d')
        ax.scatter(x_coords,y_coords,t_coords)
        plt.show()

    closest_points=[]
    #find closest point forward and backward in time for each point
    print('Linking paths')
    points_unprocessed=[Point(c) for c in coords]
    points_unlinked_forward=list(points_unprocessed)
    points_unlinked_backward=list(points_unprocessed)
    points=[]
    while points_unprocessed:
        print('Points to link:',len(points_unprocessed))
        reprocess=False
        p=points_unprocessed.pop(0)
        if not p.forward:
            points_after=[point for point in points_unlinked_backward if p.is_before(point,mem)]
            closest_forward=p.get_closest(points_after)
            #Check that the point before and point after agree with p that they should be buddies
            if closest_forward:
                before_closest_forward=[point for point in points_unlinked_forward if closest_forward.is_after(point,200)]
                if closest_forward.get_closest(before_closest_forward) is p:
                    #We agree!
                    p.set_forward(closest_forward)
                    closest_forward.set_backward(p)
                    closest_points.append(p.get_dist(closest_forward))
                    #mark these guys as together forever
                    del points_unlinked_forward[points_unlinked_forward.index(p)]
                    del points_unlinked_backward[points_unlinked_backward.index(closest_forward)]
                else:
                    #We don't agree, try again later
                    reprocess=True

        if not p.backward:
            points_before=[point for point in points_unlinked_forward if p.is_after(point,mem)]
            closest_backward=p.get_closest(points_before)
            #Check that the point before and point after agree with p that they should be buddies
            if closest_backward:
                after_closest_backward=[point for point in points_unlinked_backward if closest_backward.is_before(point,mem)]
                if closest_backward.get_closest(after_closest_backward) is p:
                    p.set_backward(closest_backward)
                    closest_backward.set_forward(p)
                    closest_points.append(p.get_dist(closest_backward))
                    del points_unlinked_backward[points_unlinked_backward.index(p)]
                    del points_unlinked_forward[points_unlinked_forward.index(closest_backward)]
                else:
                    reprocess=True

        if not reprocess:
            points.append(p)
        else:
            points_unprocessed.append(p)

    print('Trimming paths')
    max_dist=np.percentile([c for c in closest_points if c],max_dist_percentile)
    print('max_dist:',max_dist)
    #remove links with dist greater than max_dist
    for p in points:
        if p.forward:
            if p.get_dist(p.forward)>max_dist:
                p.clear_forward()
        if p.backward:
            if p.get_dist(p.backward)>max_dist:
                p.clear_backward()

    #check that each point only has one point registering it as a neighbor in each direction. if we have extras, cut the pair with the highest distance
    #this section can now probably be deleted (double-pairing can no longer happen during linking)
    print('Fixing intersecting paths')
    for p in points:
        claims_p_before=[point for point in points if point.forward is p]
        claims_p_after=[point for point in points if point.backward is p]
        #determine longest 'extra' link and cut it

        if len(claims_p_before)>1:
            raise Exception('Intersecting Paths!')
            dists=[p.get_dist(b) for b in claims_p_before]
            indices=list(range(len(dists)))
            #keep shortest link
            del indices[dists.index(min(dists))]
            for i in indices:
                claims_p_before[i].clear_forward()

        if len(claims_p_after)>1:
            raise Exception('Intersecting Paths!')
            dists=[p.get_dist(a) for a in claims_p_after]
            indices=list(range(len(dists)))
            #keep shortest link
            del indices[dists.index(min(dists))]
            for i in indices:
                claims_p_after[i].clear_backward()

    #join points into paths
    print('Finalizing')
    paths=[]
    while points:
        this_path=Path()
        this_point=points[0]
        #rewind to earliest point on path
        while this_point.backward:
            this_point=this_point.backward
        #now slurp up all linked points onto this path and remove them from points
        while this_point:
            this_path.add_point(this_point)
            try:
                del points[points.index(this_point)]
            except ValueError:
                print('this point not in points')
                return points,this_point
            this_point=this_point.forward
        paths.append(this_path)
    if debug:
        plt.imshow(draw_paths(paths,frame.shape))
        plt.show()
    #divide off time factor
    for path in paths:
        for p in path.points:
            new_coords=(p.coords[0],p.coords[1],p.coords[2]/time_factor)
            p.coords=new_coords
    return paths

def calibrate_paths(paths,cal_device):

    '''Get cell trajectories transformed to device coordinates from un-calibrated trajectories

    -----Parameters-----
    paths: A list of heimdall.tracking.Path objects representing the un-calibrated trajectories. Usually produced by heimdall.tracking.get_tracks()
    cal_device: A heimdall.device.CalibratedDevice object representing the alignment of the device to the video. Usually produced by heimdall.improcessing.detect_ridges()

    -----Returns-----
    A heimdall.tracking.CalibratedPaths object representing the cell trajectories transformed to device coordinates'''
    
    #transform paths into a list of lists of coordinates
    paths_arr=[list(p.points) for p in paths]
    for path in paths_arr:
        for i in range(len(path)):
            path[i]=list(path[i].coords)
    for points in paths_arr:
        for p in points:
            #first subtract off offset
            p[0]=p[0]-cal_device.offset_y
            p[1]=p[1]-cal_device.offset_x
            #now rotate points
            tilt_angle=np.arctan(cal_device.tilt)
            new_x,new_y=heimdall.device.rotate_point(p[1],p[0],-tilt_angle)
            p[0]=new_y
            p[1]=new_x
            #now scale them
            p[0]/=cal_device.scale
            p[1]/=cal_device.scale
            #now flip y axis to go from image to real coordinates
            p[0]*=-1
    return CalibratedPaths(paths_arr,cal_device)

class Point:
    '''A class for representing the position of a detected object'''
    def __init__(self,coords):
        '''Create a new Point object

        -----Parameters-----
        coords: the (y,x,t) coordinates of this object

        -----Returns-----
        A new Point object'''
        
        self.coords=coords
        self.forward=None
        self.backward=None

    def is_before(self,p:'Point',max_time=None):
        '''Test if this point occurs before another point

        -----Parameters-----
        p: The Point object to compare to
        max_time: The maximum time lag between points, default is None

        -----Returns-----
        before: True if this Point occured before p, False otherwise. If max_time is not None, this will return False if the points occured more than max_time apart'''
        if max_time:
            if abs(self.coords[2]-p.coords[2])>max_time:
                return False
        return self.coords[2]<p.coords[2]

    def is_after(self,p:'Point',max_time=None):
        '''Test if this point occurs after another point

        -----Parameters-----
        p: The Point object to compare to
        max_time: The maximum time lag between points, default is None

        -----Returns-----
        after: True if this Point occured after p, False otherwise. If max_time is not None, this will return False if the points occured more than max_time apart'''
        
        if max_time:
            if abs(self.coords[2]-p.coords[2])>max_time:
                return False
        return self.coords[2]>p.coords[2]

    def set_backward(self,p:'Point'):
        '''Set the nearest neighbor before this point
        
        -----Parameters-----
        p: The point that is putatively the previous point on the same path

        -----Returns-----
        None'''
        self.backward=p

    def set_forward(self,p:'Point'):
        '''Set the nearest neighbor after this point
        
        -----Parameters-----
        p: The point that is putatively the next point on the same path

        -----Returns-----
        None'''
        self.forward=p

    def clear_backward(self):
        '''Clear the nearest neighbor before this point

        -----Parameters-----
        None

        -----Returns-----
        None'''
        
        self.backward=None

    def clear_forward(self):
        '''Clear the nearest neighbor after this point

        -----Parameters-----
        None

        -----Returns-----
        None'''
        
        self.forward=None

    def get_dist(self,point):
        dist_spatial=np.sqrt(((self.coords[0]-point.coords[0])**2)+((self.coords[1]-point.coords[1])**2))
        total_dist=np.sqrt(((self.coords[2]-point.coords[2])**2)+(dist_spatial**2))
        return total_dist

    def get_closest(self,points:list):
        min_dist=float('inf')
        min_dist_point=None

        for point in points:
            this_dist=self.get_dist(point)
            if this_dist<min_dist:
                min_dist=this_dist
                min_dist_point=point

        return min_dist_point


class Path:
    def __init__(self):
        self.points=[]
    def add_point(self,p:'Point'):
        self.points.append(p)
    def get_path_coords(self)->(tuple,tuple):
        y_index=[]
        x_index=[]
        last_point=self.points[0]
        for point in self.points[1:]:
            new_y,new_x=skimage.draw.line(*last_point.coords[0:2],*point.coords[0:2])
            y_index.extend(new_y)
            x_index.extend(new_x)
            last_point=point
        return y_index,x_index

class CalibratedPaths:
    def __init__(self,paths,cal_device):
        self.paths=paths
        self.cal_device=cal_device
    def to_df(self)->pd.DataFrame:
        path_frames=[]
        path_no=0
        for p in self.paths:
            x=[point[1] for point in p]
            y=[point[0] for point in p]
            t=[point[2] for point in p]
            dist_and_next_ridge=[self.cal_device.dist_to_next_ridge(point) for point in p]
            dist_next=[d[0] for d in dist_and_next_ridge]
            next_ridge=[d[1] for d in dist_and_next_ridge]
            under_ridge=[self.cal_device.point_under_ridge(point) for point in p]
            in_gutter=[self.cal_device.point_in_gutter(point) for point in p]
            path_frames.append(pd.DataFrame(dict(path=[path_no]*len(x),x=x,y=y,t=t,dist_next_ridge=dist_next,next_ridge=next_ridge,under_ridge=under_ridge,in_gutter=in_gutter)))
            path_no+=1
        return pd.concat(path_frames,ignore_index=True)

def draw_paths(paths:tuple,img_dim):
    img=np.zeros(img_dim,np.bool)
    for path in paths:
        y_coords,x_coords=path.get_path_coords()
        img[y_coords,x_coords]=True
    return img

def plot_paths(*cal_paths,colors=('b', 'g', 'r', 'c', 'm', 'y', 'k')):
    colors=list(colors)
    for i in range(len(colors)):
        colors[i]+='-'
    len_ratio=int(np.ceil(len(cal_paths)/len(colors)))
    colors*=len_ratio
    #Check for the same device
    device_num_ridges=([len(p.cal_device.ridge_coords) for p in cal_paths])
    num_same=min(device_num_ridges)
    dev=cal_paths[0].cal_device
    fig,ax=plt.subplots()
    for ridge in dev.ridge_coords[0:num_same]:
        ridge_x=[r[0] for r in ridge+ridge[0:1]]
        ridge_y=[r[1] for r in ridge+ridge[0:1]]
        ax.plot(ridge_x,ridge_y,'k-')
    #only plot up to num_sameth ridge
    for cal_path_group,color in zip(cal_paths,colors):
        for path in cal_path_group.paths:
            path_x=[p[1] for p in path]
            path_y=[p[0] for p in path]
            ax.plot(path_x,path_y,color)
    plt.show()
