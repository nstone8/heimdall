import numpy as np
import skimage.draw

class RidgeSpec:
    def __init__(self,ridge_height:float,ridge_width:float,ridge_spacing:float,ridge_angle:float):
        self.height=ridge_height
        self.width=ridge_width
        self.spacing=ridge_spacing
        self.angle=ridge_angle

    def get_mask(self,num_ridge,scale,tilt,offset_y,offset_x,im_shape,im_dtype=np.bool):
        mask=np.zeros(im_shape,im_dtype)
        starting_coords=((0,0),(-self.width*scale,0),((-1*scale*self.height/np.tan(self.angle))-self.width*scale,self.height*scale),(-1*scale*self.height/np.tan(self.angle),self.height*scale))

        all_coords_no_tilt_no_trans=[]
        for i in range(num_ridge):
            all_coords_no_tilt_no_trans.append([(x-(i*self.spacing*scale),y) for x,y in starting_coords])
            
        all_coords_no_trans=[[_rotate_point(x,y,tilt) for x,y in ridge] for ridge in all_coords_no_tilt_no_trans]

        all_coords=[[(x+offset_x,y+offset_y) for x,y in ridge] for ridge in all_coords_no_trans]
        
        #generate mask
        for ridge in all_coords:
            ridge_x=[r[0] for r in ridge]
            ridge_y=[r[1] for r in ridge]
            rr,cc=skimage.draw.polygon_perimeter(ridge_y,ridge_x,im_shape)
            mask[rr,cc]=True

        return mask
            
def _rotate_point(x,y,angle)->(float,float):
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
