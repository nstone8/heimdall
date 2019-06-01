import numpy as np
import skimage.draw
import shapely.geometry

class RidgeSpec:
    def __init__(self,ridge_height:float,ridge_width:float,ridge_spacing:float,ridge_angle:float):
        self.height=ridge_height
        self.width=ridge_width
        self.spacing=ridge_spacing
        self.angle=ridge_angle

    def get_mask(self,num_ridge,scale,tilt,offset_y,offset_x,im_shape,im_dtype=np.bool):
        mask=np.zeros(im_shape,im_dtype)
        all_coords=self.get_ridge_coords(num_ridge,scale,tilt,offset_y,offset_x)
        #generate mask
        for ridge in all_coords:
            ridge_x=[r[0] for r in ridge]
            ridge_y=[r[1] for r in ridge]
            rr,cc=skimage.draw.polygon_perimeter(ridge_y,ridge_x,im_shape)
            mask[rr,cc]=True

        return mask
    def get_ridge_coords(self,num_ridge,scale,tilt,offset_y,offset_x):
        starting_coords=((0,0),(-self.width*scale,0),((-1*scale*self.height/np.tan(self.angle))-self.width*scale,self.height*scale),(-1*scale*self.height/np.tan(self.angle),self.height*scale))
        all_coords_no_tilt_no_trans=[]
        for i in range(num_ridge):
            all_coords_no_tilt_no_trans.append([(x-(i*self.spacing*scale),y) for x,y in starting_coords])

        all_coords_no_trans=[[rotate_point(x,y,tilt) for x,y in ridge] for ridge in all_coords_no_tilt_no_trans]

        all_coords=[[(x+offset_x,y+offset_y) for x,y in ridge] for ridge in all_coords_no_trans]
        return all_coords

class CalibratedDevice:
    def __init__(self,ridge_spec,num_ridge,scale,tilt,offset_y,offset_x):
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
        return self.ridge_poly[0].distance(self.ridge_poly[1])

    def point_under_ridge(self,point:tuple):
        '''returns None or the index of the ridge this point is under'''
        point_point=shapely.geometry.Point([point[1],point[0]])
        for i in range(len(self.ridge_poly)):
            if self.ridge_poly[i].contains(point_point):
                return i
        return None

    def dist_to_next_ridge(self,point:tuple)->('dist','ridge_no'):
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
        
    def point_in_gutter(self,point):
        if (point[0]>self.ridge_coords[0][0][1]) or (point[0]<self.ridge_coords[0][2][1]):
            return True
        else:
            return False


def rotate_point(x,y,angle)->(float,float):
    r=np.sqrt((x**2)+(y**2))
    if r==0:
        return 0,0
    elif x==0:
        if y>0:
            start_angle=np.pi/2
        else:
            start_angle=3*np.pi/2
    else:
        start_angle=np.pi-np.arctan(y/x)
    new_x=r*np.cos(start_angle+angle)
    new_y=r*np.sin(start_angle+angle)
    return x,y
