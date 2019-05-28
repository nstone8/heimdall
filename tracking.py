import heimdall.improcessing as imp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import skimage.draw

def get_tracks(movers,time_factor:int=10,max_dist_percentile=99.9,mem=None,debug=False):
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
    print('Fixing intersecting paths')
    for p in points:
        claims_p_before=[point for point in points if point.forward is p]
        claims_p_after=[point for point in points if point.backward is p]
        #determine longest 'extra' link and cut it

        if len(claims_p_before)>1:
            print('Intersecting Paths!')
            dists=[p.get_dist(b) for b in claims_p_before]
            indices=list(range(len(dists)))
            #keep shortest link
            del indices[dists.index(min(dists))]
            for i in indices:
                claims_p_before[i].clear_forward()

        if len(claims_p_after)>1:
            print('Intersecting Paths!')
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
    return paths

class Point:
    def __init__(self,coords):
        self.coords=coords
        self.forward=None
        self.backward=None

    def is_before(self,p:'Point',max_time=None):
        if max_time:
            if abs(self.coords[2]-p.coords[2])>max_time:
                return False
        return self.coords[2]<p.coords[2]

    def is_after(self,p:'Point',max_time=None):
        if max_time:
            if abs(self.coords[2]-p.coords[2])>max_time:
                return False
        return self.coords[2]>p.coords[2]

    def set_backward(self,p:'Point'):
        self.backward=p

    def set_forward(self,p:'Point'):
        self.forward=p

    def clear_backward(self):
        self.backward=None

    def clear_forward(self):
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

def draw_paths(paths:tuple,img_dim):
    img=np.zeros(img_dim,np.bool)
    for path in paths:
        y_coords,x_coords=path.get_path_coords()
        img[y_coords,x_coords]=True
    return img
