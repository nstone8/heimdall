import heimdall.improcessing as imp
import numpy as np
from matplotlib import pyplot as plt
import skimage

def get_tracks(vid_path:str,num_bg,cell_size,min_cell_size=None,velocity=None,memory=4,debug=False):
    '''track cells in a video if min_cell_size==None, it will be set to 50% of cell_size, if velocity==None,
    it will be set to 150% cell_size, all args with length units are in pixels'''
    if min_cell_size==None:
        min_cell_size=0.5*cell_size
    if velocity==None:
        velocity=1.5*cell_size
    movers=imp.gen_movers(vid_path,num_bg,cell_size,min_cell_size)
    frame_num=0
    paths=[]
    for frame in movers:
        if debug:
            plt.imshow(frame)
            plt.show()
            response=input('Enter for next image, q to stop, anything else to continue without output\n').strip()
            if response:
                if response=='q':
                    raise Exception('Stop')
                else:
                    debug=False

        objs=[o for o in set(frame.ravel()) if o]
        for obj in objs:
            this_obj_mask=frame==obj
            this_obj_location=imp.get_centroid(this_obj_mask)
            this_point=Point(frame_num,this_obj_location,velocity,memory)
            closest_path=None
            closest_path_dist=float('inf')
            for path in paths:
                this_path_dist=path.closest_point_on_same_path_dist(this_point)
                if (this_path_dist!=None) and (this_path_dist<closest_path_dist):
                    closest_path_dist=this_path_dist
                    closest_path=path
            if closest_path:
                closest_path.add_point(this_point)

            else:
                new_path=Path()
                new_path.add_point(this_point)
                paths.append(new_path)
        frame_num+=1
    path_image=draw_paths(paths,frame.shape)
    plt.imshow(path_image)
    plt.show()
    return paths


class Path:
    def __init__(self):
        self.points=[]
    def add_point(self,p:'Point'):
        self.points.append(p)
    def closest_point_on_same_path_dist(self,p:'Point'):
        if not self.points:
            return None
        else:
            dists=[]
            for point in self.points:
                if p.on_same_path(point):
                    dists.append(p.distance_between(point))
            if dists:
                return min(dists)
            else:
                return None
    def get_path_coords(self)->(tuple,tuple):
        y_index=[]
        x_index=[]
        last_point=self.points[0]
        for point in self.points[1:]:
            new_y,new_x=skimage.draw.line(*last_point.location,*point.location)
            y_index.extend(new_y)
            x_index.extend(new_x)
            last_point=point
        return y_index,x_index

class Point:
    def __init__(self,time:int,location:(int,int),velocity:float,memory:int):
        self.time=time
        self.location=location
        self.velocity=velocity
        self.memory=memory

    def on_same_path(self,p:'Point'):
        if abs(self.time-p.time)<(self.memory+1) and self.distance_between(p)<self.velocity:
            return True
        else:
            return False

    def distance_between(self,p:'Point'):
        '''calculate distance between self and p'''
        return np.sqrt(((self.location[0]-p.location[0])**2)+((self.location[1]-p.location[1])**2))

def draw_paths(paths:tuple,img_dim):
    img=np.zeros(img_dim,np.bool)
    for path in paths:
        y_coords,x_coords=path.get_path_coords()
        img[y_coords,x_coords]=True
    return img

def plot_paths(*cal_paths):
    dev=cal_paths[0].cal_device
    fig,ax=plt.subplots
    for ridge in dev.ridge_coords:
        ridge_x=[r[0] for r in ridge]
        ridge_y=[r[1] for r in ridge]
        ax.plot(ridge_x,ridge_y,'k-')

    for cal_path_group in cal_paths:
        for path in cal_path_group:
            path_x=[p[1] for p in path]
            path_y=[p[0] for p in path]
            ax.plot(path_x,path_y,'b-')
    plt.show()
