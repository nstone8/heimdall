import skvideo.io
import numpy as np
import scipy
import skimage.filters
import skimage.feature
import skimage.morphology
import skimage.color
import skimage.measure
from matplotlib import pyplot as plt

def get_background(vid_path,num_bg):
    #take mean of all frames
    vid=skvideo.io.vreader(vid_path,as_grey=True)
    first_frame=next(vid)[0,:,:,0]
    total=np.zeros(first_frame.shape)
    total+=first_frame
    count=1
    for frame in vid:
        total+=frame[0,:,:,0]
        count+=1
        if num_bg and (count>num_bg):
            break
    background=total/count
    return background.astype(first_frame.dtype)

def detect_ridges(vid_path,device,num_bg=None):
    background=get_background(vid_path,num_bg)
    sobel=skimage.filters.sobel(background)
    thresh=skimage.filters.threshold_otsu(sobel)
    ridges=sobel>thresh
    #get y,x coordinates of corners
    ridges_closed=skimage.morphology.binary_closing(ridges)#,disk(10))
    ridges_skeleton=skimage.morphology.skeletonize(ridges_closed)
    seg=skimage.measure.label(~ridges_closed,connectivity=1)
    #remove all objects touching edges
    edge_obj=[]
    edge_obj.extend(seg[0,:])
    edge_obj.extend(seg[:,0])
    edge_obj.extend(seg[-1,:])
    edge_obj.extend(seg[:,-1])
    
    for obj in set(edge_obj):
        seg[seg==obj]=0

    #filter out objects smaller than the ridges
    obj_sizes=[(obj,seg[seg==obj].size) for obj in set(seg.ravel()) if obj>0]
    max_size=max([obj_s[1] for obj_s in obj_sizes])
    for obj,size in obj_sizes:
        if size<0.8*max_size:
            #this object is smaller than the ridges, remove from seg
            seg[seg==obj]=0

    #for each ridge, find the two points that are furthest apart, these will be our 'corners'
    #Would probably be faster to first find the point furthest from the centroid, then the point furthest from that point

    corners=[]
    for ridge in set(seg[seg>0].ravel()):
        #I think the data is in (y,x) order
        this_ridge=np.zeros(seg.shape,seg.dtype)
        y_coords,x_coords=np.where(seg==ridge)
        this_ridge[y_coords,x_coords]=1
        moment=skimage.measure.moments(this_ridge)
        ridge_centroid=(moment[1, 0] / moment[0, 0], moment[0, 1] / moment[0, 0])
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

        corners.append([first_corner,second_corner])


    #Get 'top' corners and use them to estimate tilt, scale and number of ridges
    top_corners=[]
    for p1,p2 in corners:
        #'top' corner has lower y value
        top_c=p1 if p1[0]<p2[0] else p2
        top_corners.append(top_c)

    print(top_corners)

    #fit a line to top corners to estimate tilt
    popt,pcov=scipy.optimize.curve_fit(lambda x,m,b:[m*this_x+b for this_x in x],[p[1] for p in top_corners],[p[0] for p in top_corners],[0,top_corners[0][0]])
    detected_slope=popt[0]
    detected_intercept=popt[1]
    detected_tilt=np.arctan(detected_slope)

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

    #print('Estimated values:')
    #print('tilt:{tilt}, spacing:{spacing}, number of ridges:{ridge}, y offset:{y}, x offset:{x}'.format(tilt=detected_tilt,spacing=detected_ridge_spacing,ridge=detected_num_ridges,y=detected_y_offset,x=detected_x_offset))

    optimized_result=maximize_overlap(ridges_skeleton,device,[detected_num_ridges,detected_scale,detected_tilt,detected_y_offset,detected_x_offset,background.shape])

    #need to add in detected_num_ridges and background.shape as they were not considered in optimization
    optimized_params=[detected_num_ridges,*optimized_result.x,background.shape]

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

    return optimized_result

def maximize_overlap(ridge_mask,device,initial_guess):
    '''Parameters:
    ridge_mask: ridge skeleton mask
    device: device object
    initial_guess: list of args for device.get_mask()'''
    def total_dist(source_mask,device_mask):
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

def gen_movers(vid_path, num_bg):
    '''get moving objects in a video'''
    print('Detecting contrast')
    bg=get_background(vid_path,num_bg)
    thresholds=[]
    for frame in skvideo.io.vreader(vid_path,as_grey=True):
        frame=frame[0,:,:,0]
        diff=np.abs(frame-bg)
        thresh=skimage.filters.threshold_otsu(diff)
        thresholds.append(thresh)
    med_thresh=np.median(thresholds)
    print('Detecting moving objects')
    for frame2 in skvideo.io.vreader(vid_path,as_grey=True):
        frame2=frame2[0,:,:,0]
        diff2=np.abs(frame2-bg)
        binary=diff2>med_thresh
        yield binary

def save_vid(frames_iter,path):
    '''save iterable of frames as a video to path
    video must be able to fit in memory'''
    print('Assembling video')
    first_frame=next(frames_iter)
    frames=np.zeros((1,*first_frame.shape),first_frame.dtype)
    for f in frames_iter:
        old_shape=f.shape
        new_frame=np.zeros((1,*old_shape),f.dtype)
        new_frame[0,:,:]=f
        frames=np.append(frames,new_frame,0)
    print('writing video')
    skvideo.io.vwrite(path,frames)
    
