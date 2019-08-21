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
        print('\rthresholded frame',frame_count,end='')
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
            print()
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
