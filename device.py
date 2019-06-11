import numpy as np
import skimage.draw
import shapely.geometry

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

    def get_ridge_offsets(self,offset:float):
        '''get shapely.LineStrings offset displaced upstream and downstream from the ridge'''
        out=[]
        for ridge in self.ridge_coords:
            upstream_vertices=ridge[0:1]+ridge[3:]
            downstream_vertices=ridge[1:3]
            upstream=shapely.geometry.LineString([[x+offset,y] for x,y in upstream_vertices])
            downstream=shapely.geometry.LineString([[x-offset,y] for x,y in downstream_vertices])
            out.append([upstream,downstream])
        return out

def rotate_point(x,y,angle)->(float,float):
    r=np.sqrt((x**2)+(y**2))
    start_angle=np.arctan2(y,x)
    new_x=r*np.cos(start_angle+angle)
    new_y=r*np.sin(start_angle+angle)
    return new_x,new_y
