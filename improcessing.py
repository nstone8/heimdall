import tempfile
import os
import re
import sys
import heimdall.device
import skvideo.io
import numpy as np
import scipy
import skimage.filters
import skimage.feature
import skimage.morphology
from skimage.morphology import disk
import skimage.color
import skimage.measure
from matplotlib import pyplot as plt

def get_background(vid:'heimdall.improcessing.VidIterable',num_bg:'int'):
    '''Extract the background from a video by taking the mean of many frames.
    -----Parameters-----
    vid: heimdall.improcessing.VidIterable object representing the video to be processed
    num_bg: Number of frames to process

    -----Returns-----
    background: image of the video background as a numpy.ndarray'''
    
    #take mean of all frames
    vid=iter(vid)
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

def detect_ridges(background:np.ndarray,device:heimdall.device.RidgeSpec,debug:bool=False):
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
    ridges_closed=skimage.morphology.binary_closing(ridges)#,disk(10))
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

    #fig,ax=plt.subplots()
    #ax.imshow(skimage.color.label2rgb(seg))
    #plt.show()
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
        ridge_centroid=get_centroid(this_ridge)
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

        #calculate ridge slope and reject objects that don't match the ridges
        abs_slopes=[abs((c[0][0]-c[1][0])/(c[0][1]-c[1][1])) for c in corners]
        med_slope=np.median(abs_slopes)
        for i in range(len(corners)):
            #if the slope is off by more than 20%, can it
            if (abs_slopes[i]<(0.8*med_slope)) or (abs_slopes[i]>(1.2*med_slope)):
                del corners[i]


    #Get 'top' corners and use them to estimate tilt, scale and number of ridges
    top_corners=[]
    for p1,p2 in corners:
        #'top' corner has lower y value
        top_c=p1 if p1[0]<p2[0] else p2
        top_corners.append(top_c)

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

    #print('Estimated values:')
    #print('tilt:{tilt}, spacing:{spacing}, number of ridges:{ridge}, y offset:{y}, x offset:{x}'.format(tilt=detected_tilt,spacing=detected_ridge_spacing,ridge=detected_num_ridges,y=detected_y_offset,x=detected_x_offset))
    optimized_result=maximize_overlap(ridges_skeleton,device,[detected_num_ridges,detected_scale,detected_tilt,detected_y_offset,detected_x_offset,background.shape])
    #need to add in detected_num_ridges and background.shape as they were not considered in optimization
    optimized_params=[detected_num_ridges,*optimized_result.x,background.shape]
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

    return heimdall.device.CalibratedDevice(device,*optimized_params[0:-1])

def maximize_overlap(ridge_mask:np.ndarray,device:heimdall.device.RidgeSpec,initial_guess:list):
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

def gen_movers(vid:'heimdall.improcessing.VidIterable',bg:np.ndarray,cell_size:int,min_cell_size:int=None,debug:bool=False):
    '''Generate moving objects in a video

    -----Parameters-----
    vid: A heimdall.improcessing.VidIterable representing the video to be processed
    bg: A numpy.ndarray representing the background of vid. Usually produced by heimdall.improcessing.get_background()
    cell_size: Nominal size (in pixels) of cells in this video
    min_cell_size: The minimum object size (in pixels) to be considered a cell. If None (the default), it will be set to 50% of cell_size
    debug: Whether or not to display plots for debugging. Default is False

    -----Returns-----
    movers: A generator object which yields numpy.ndarrays in which pixels belonging to each detected object are labeled with unique integers and background pixels are labeled with zeros
    '''
    if min_cell_size==None:
        min_cell_size=.5*cell_size
    print('Detecting contrast')
    max_diff=0
    max_diff_frame=None
    for frame in vid:
        #detect frame with maximum mean diff, most likely to contain cells
        frame=frame[0,:,:,0]
        diff=np.abs(frame-bg)
        mean_diff=np.mean(diff)
        if mean_diff>max_diff:
            max_diff=mean_diff
            max_diff_frame=frame
    #find threshold where mean obj size is closest to cell_size
    #might be better if we look for max(obj_size)-mean(obj_size) closest to cell_size
    def get_mean_obj_size(frame,thresh):
        this_bin=frame>thresh
        labeled=skimage.measure.label(this_bin)
        sizes=[]
        for obj in set(labeled.ravel()):
            sizes.append(labeled[labeled==obj].size)
        return np.mean(sizes)

    min_res=scipy.optimize.minimize(lambda thresh: abs(cell_size-get_mean_obj_size(max_diff_frame,thresh)),[skimage.filters.threshold_otsu(max_diff_frame)])
    optim_thresh=min_res.x[0]
    print('Detecting moving objects')
    frame_count=0
    for frame2 in vid:
        frame2=frame2[0,:,:,0]
        diff2=np.abs(frame2-bg)
        binary=diff2>optim_thresh
        frame_count+=1
        print('thresholded frame',frame_count)
        #delete small objects
        # labeled_frame=skimage.measure.label(binary)
        # for obj in set(labeled_frame.ravel()):
        #     if labeled_frame[labeled_frame==obj].size<min_cell_size:
        #         labeled_frame[labeled_frame==obj]=0

        binary=skimage.morphology.binary_erosion(binary,disk(int(min_cell_size/2)))
        binary=skimage.morphology.binary_dilation(binary,disk(int(min_cell_size/2)))
        labeled_binary=skimage.measure.label(binary)
        if debug:
            fig,ax=plt.subplots()
            ax.imshow(skimage.color.label2rgb(labeled_binary))
            plt.show()
            response=input('Press enter to show next frame, q to stop, or anything else to finish\n')
            if response=='q':
                raise Exception('STOP')
            elif response:
                debug=False
            plt.close('all')

        yield labeled_binary

def get_vid_length(vid_path):
    '''Get the number of frames in a video
    
    -----Parameters-----
    vid_path: Path to a video file

    -----Returns-----
    num_frames: The number of frames in the video stored at vid_path
    '''
    vid=skvideo.io.vreader(vid_path)
    num_frames=0
    for frame in vid:
        num_frames+=1
    return num_frames

def save_movers_vid(frames_iter:'generator',path:str,num_frames:int,frame_by_frame:bool=False):
    '''Save iterable of frames as a video (video must be able to fit in memory)

    -----Parameters-----
    frames_iter: A generator of numpy.ndarrays representing images (such as that produced by gen_movers())
    path: Location to store video file
    num_frames: The number of frames in frames_iter
    frame_by_frame: A bool indicating whether or not to display images frame by frame while assembling video. Default is False

    -----Returns-----
    None
    '''
    print('Assembling video')
    first_frame=skimage.color.label2rgb(next(frames_iter)>0)
    frames=np.zeros((num_frames,*first_frame.shape),first_frame.dtype)
    print(first_frame.dtype)
    frames[0,:,:,:]=first_frame
    frame_index=1
    for f in frames_iter:
        f=f>0
        labeled_frame=skimage.util.img_as_ubyte(skimage.color.label2rgb(f))
        frames[frame_index,:,:,:]=labeled_frame
        frame_index+=1
        if frame_by_frame:
            fig,ax=plt.subplots()
            ax.imshow(labeled_frame)
            plt.show()
            response=input('Press enter to show next frame, q to stop, or anything else to finish saving\n')
            if response=='q':
                raise Exception('STOP')
            elif response:
                frame_by_frame=False
            plt.close('all')
    print('writing video')
    skvideo.io.vwrite(path,frames)
def get_centroid(obj_mask:np.ndarray):
    '''Get the centroid of the positive pixels in a binary image

    -----Parameters-----
    obj_mask: A numpy.ndarray representing a binary image

    -----Returns-----
    [y,x]: Centroid coordinates'''
    moment=skimage.measure.moments(obj_mask)
    centroid=(moment[1, 0] / moment[0, 0], moment[0, 1] / moment[0, 0])
    return [int(np.round(p)) for p in centroid]

class VidIterable:
    """Iterator to allow multiple scans across a video, vid_flow_direction should be the direction the cells flow, either 'left' or 'up'"""
    def __init__(self,vid_path:str,num_frames:int=None,vid_flow_direction:str='left'):
        """Create a new VidIterable object

        -----Parameters-----
        vid_path: File path where the video that this VidIterable will represent is stored
        num_frames: Number of frames from the source video to include in this VidIterable
        vid_flow_direction: The direction in which cells flow in the video. Should be either 'left' or 'up'. Default is 'left'
        
        -----Returns-----
        vid: A new VidIterable object"""
        
        self.num_frames=num_frames
        if vid_path.split('.')[-1]=='cine' or vid_flow_direction!='left':
            #This is a cine file or needs to be rotated, convert to mp4
            print('Converting .cine file to mp4 (lossless)')
            #detect platform so we can correct file paths for ffmpeg
            is_win=re.compile('.*[Ww]in.*')
            if is_win.match(sys.platform):
                corrected_vid_path='"'+vid_path+'"'
            else:
                #Put escape characters in front of spaces in file name
                corrected_vid_path=[]
                for c in vid_path:
                    if c==' ':
                        corrected_vid_path.append('\\')
                    corrected_vid_path.append(c)
                corrected_vid_path=''.join(corrected_vid_path)
            if vid_flow_direction=='up':
                rotate='-vf "transpose=2" '
            elif vid_flow_direction=='left':
                rotate=''
            else:
                raise Exception("vid_flow_direction must be 'up' or 'left'")
            if num_frames!=None:
                frames='-frames:v {0} '.format(num_frames)
            else:
                frames=''
            os_handle,new_file_path=tempfile.mkstemp(suffix='.mp4')
            #close file, we don't work with it directly
            os.close(os_handle)
            ffmpeg_command='ffmpeg -y -i {orig_file} {frames}{rotate}-f mp4 -crf 0 {new_file}'.format(orig_file=corrected_vid_path,rotate=rotate,new_file=new_file_path,frames=frames)
            print(ffmpeg_command)
            list(os.popen(ffmpeg_command))
            self.vid_path=new_file_path
            self.delete_file=True
            stats=os.stat(new_file_path)
            if stats.st_size==0:
                raise Exception('File conversion failed, check that ffmpeg is on PATH')
        else:
            #Not a cine
            self.vid_path=vid_path
            self.delete_file=False
    def __iter__(self):
        '''Return an iterator of frames in the video represented by this object
        
        -----Parameters-----
        None

        -----Returns-----
        iter: an iterator of frames in the video represented by this object'''
        return skvideo.io.vreader(self.vid_path,as_grey=True)
    def __del__(self):
        '''Clean up any temp files after this object is destroyed

        -----Parameters-----
        None

        -----Returns-----
        None'''
        #need to delete any temp files when we are destroyed
        if self.delete_file:
            os.remove(self.vid_path)
