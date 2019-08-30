import numpy as np
import pandas as pd
import math
import shapely.geometry
from plotly import graph_objs as go
from plotly import offline as py
from plotly.subplots import make_subplots

def get_interaction_time(cal_paths,cell_size,in_gutter_rm=True):
    paths_frame=cal_paths.to_df()
    if in_gutter_rm:
        paths_frame=paths_frame.loc[~paths_frame.loc[:,'in_gutter'],:]
    #Get the interaction time of each path at each ridge
    output_frames=[]
    ridge_sep=cal_paths.cal_device.get_ridge_sep()
    num_ridges=len(cal_paths.cal_device.ridge_coords)
    for path in set(paths_frame.loc[:,'path']):
        this_path=paths_frame.loc[paths_frame.loc[:,'path']==path,:]
        for ridge in range(num_ridges):
            #do we have data for a full ridge_sep around the ridge?
            approaching_this_ridge=this_path.loc[this_path.loc[:,'next_ridge']==ridge,:]
            if not approaching_this_ridge.shape[0]:
                #No points approaching the ridge, move on
                continue
            dist_before=np.max(approaching_this_ridge.loc[:,'dist_next_ridge'])
            if ridge==(num_ridges-1):
                #We're interested in the last ridge
                points_after=this_path.loc[this_path.loc[:,'next_ridge']=='after_last_ridge',:]
            else:
                points_after=this_path.loc[this_path.loc[:,'next_ridge']==(ridge+1),:]

            y_after=points_after.loc[:,'y']
            x_after=points_after.loc[:,'x']
            dists_after=[shapely.geometry.Point([x,y]).distance(cal_paths.cal_device.ridge_poly[ridge]) for x,y in zip(x_after,y_after)]
            if not dists_after:
                #no points after, time to move on
                continue
            dist_after=max(dists_after)
            if (dist_before>(ridge_sep/2)) and (dist_after>(ridge_sep/2)):
                #We have coverage on both sides, let's measure
                #get the first time the cell is within one radius of the ridge
                time_end=None
                interacting=approaching_this_ridge.loc[approaching_this_ridge.loc[:,'dist_next_ridge']<cell_size/2,:]
                under_ridge=this_path.loc[this_path.loc[:,'under_ridge']==ridge,:]
                if interacting.shape[0]:
                    time_start=interacting.iloc[0]['t']
                    time_end=interacting.iloc[-1]['t'] #ensure that we at least count all of the time in the interaction area even if the cell never goes under the ridge
                elif under_ridge.shape[0]>1:

                    time_start=under_ridge.iloc[0]['t']
                    #we can no longer use this point for time_end
                    under_ridge=under_ridge.iloc[1:]
                else:
                    under_ridge=pd.DataFrame() #this will force the next if statement to register that we have a time of 0
                #now find the last time the cell was under the ridge

                if under_ridge.shape[0]<1 and (not time_end):
                    output_frames.append(pd.DataFrame(dict(path=[path],ridge=[ridge],interaction_time=[0])))
                else:
                    if under_ridge.shape[0]>0:
                        time_end=under_ridge.iloc[-1]['t']
                    output_frames.append(pd.DataFrame(dict(path=[path],ridge=[ridge],interaction_time=[time_end-time_start])))

    return pd.concat(output_frames,ignore_index=True)

def get_defl_per_ridge(cal_paths):
    paths_frame=cal_paths.to_df()
    y_min=min(paths_frame.loc[:,'y'])*1.1
    y_max=max(paths_frame.loc[:,'y'])*1.1
    #get x distance between end of one ridge and beginning of the next
    ridge_sep=cal_paths.cal_device.ridge_coords[0][1][0]-cal_paths.cal_device.ridge_coords[1][0][0]
    num_ridges=len(cal_paths.cal_device.ridge_coords)
    def extend_linestring(linestring,y_min,y_max):
        line_coords=linestring.coords
        if len(line_coords)!= 2:
            raise Exception('Ridge offset lines should only have two points')
        this_slope=(line_coords[0][1]-line_coords[1][1])/(line_coords[0][0]-line_coords[1][0])
        this_intercept=line_coords[0][1]-(this_slope*line_coords[0][0])
        get_x=lambda y: (y-this_intercept)/this_slope
        x_min,x_max=[get_x(y) for y in [y_min,y_max]]
        return shapely.geometry.LineString([[x_min,y_min],[x_max,y_max]])

    #this must be the ridge separation in the x direction / 2. Otherwise we aren't sampling at the same point before and after the ridge
    ridge_offsets=cal_paths.cal_device.get_ridge_offsets(ridge_sep/2)

    ridge_offsets=[[extend_linestring(up,y_min,y_max),extend_linestring(down,y_min,y_max)] for up,down in ridge_offsets]
    path=[]
    ridge=[]
    defl=[]
    for p in set(paths_frame.loc[:,'path']):
        this_path=paths_frame.loc[paths_frame.loc[:,'path']==p,:]
        path_coords=list(zip(this_path.loc[:,'x'],this_path.loc[:,'y']))
        if len(path_coords)<2:
            #We only have one point, move on
            continue
        if all([p==path_coords[0] for p in path_coords]):
            #We have multiple copies of the same point
            continue
        path_linestring=shapely.geometry.LineString(path_coords)
        for r in range(num_ridges):
            if path_linestring.intersects(ridge_offsets[r][0]) and path_linestring.intersects(ridge_offsets[r][1]):
                #we have data between one offset before and one after this ridge use the last intersecting point upstream and the first downstream to calc deflection
                try:
                    upstream=path_linestring.intersection(ridge_offsets[r][0])
                    downstream=path_linestring.intersection(ridge_offsets[r][1])
                except shapely.errors.TopologicalError:
                    #This (rare) error sometimes occurs when a path is coincident with a ridge offset
                    print('Ignoring TopologicalError')
                    continue
                if type(upstream)!=shapely.geometry.point.Point:
                    upstream=upstream[-1]
                if type(downstream)!=shapely.geometry.point.Point:
                    downstream=downstream[0]
                this_defl=downstream.coords[0][1]-upstream.coords[0][1]
                path.append(p)
                ridge.append(r)
                defl.append(this_defl)
    return pd.DataFrame(dict(path=path,ridge=ridge,deflection=defl))

def get_cumulative_defl(cal_paths):
    per_ridge=get_defl_per_ridge(cal_paths)
    path=[]
    ridge=[]
    cum_defl=[]
    for p in set(per_ridge.loc[:,'path']):
        this_path=per_ridge.loc[per_ridge.loc[:,'path']==p,:]
        this_ridge=0
        cumulative=0
        this_ridge_frame=this_path.loc[this_path.loc[:,'ridge']==this_ridge]
        while(this_ridge_frame.shape[0]>0):
            cumulative+=this_ridge_frame.iloc[0]['deflection']
            path.append(p)
            ridge.append(this_ridge)
            cum_defl.append(cumulative)
            this_ridge+=1
            this_ridge_frame=this_path.loc[this_path.loc[:,'ridge']==this_ridge]
    return pd.DataFrame(dict(path=path,ridge=ridge,cumulative_deflection=cum_defl))

def make_boxplot(frames,names,ylabel,column,filename='temp_plot.html'):
    traces=[]
    for f,n in zip(frames,names):
        this_trace=go.Box(x=f.loc[:,'ridge'],y=f.loc[:,column],name=n)
        traces.append(this_trace)
    layout=go.Layout(yaxis=dict(title=ylabel),xaxis=dict(title='Ridge'),boxmode='group')
    fig=go.Figure(data=traces,layout=layout)
    py.plot(fig,filename=filename)

def plot_interaction_time(*calibrated_paths_and_names,cell_size,in_gutter_rm=True,filename='temp_plot.html'):
    frames=[]
    cal_paths=[cn[0] for cn in calibrated_paths_and_names]
    names=[cn[1] for cn in calibrated_paths_and_names]
    for c in cal_paths:
        inter=get_interaction_time(c,cell_size,in_gutter_rm)
        frames.append(inter)
    make_boxplot(frames,names,'Interaction Time','interaction_time',filename)

def plot_deflection_per_ridge(*calibrated_paths_and_names,filename='temp_plot.html'):
    frames=[]
    cal_paths=[cn[0] for cn in calibrated_paths_and_names]
    names=[cn[1] for cn in calibrated_paths_and_names]
    for c in cal_paths:
        defl=get_defl_per_ridge(c)
        frames.append(defl)
    make_boxplot(frames,names,'Deflection (µm)','deflection',filename)

def plot_defl_vs_inter(*calibrated_paths_and_names,cell_size,in_gutter_rm=True,num_col=3,filename='temp_plot.html'):
    our_colors=[
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
    ]
    if len(our_colors)<len(calibrated_paths_and_names):
        our_colors*=math.ceil(len(cal_paths)/len(our_colors))
    cal_paths=[cn[0] for cn in calibrated_paths_and_names]
    names=[cn[1] for cn in calibrated_paths_and_names]
    defl=[]
    inter=[]
    num_ridges=0
    for c in cal_paths:
        this_inter=get_interaction_time(c,cell_size,in_gutter_rm)
        this_defl=get_defl_per_ridge(c)
        defl.append(this_defl)
        inter.append(this_inter)
        num_ridges_defl=max(this_defl.loc[:,'ridge'])
        num_ridges_inter=max(this_inter.loc[:,'ridge'])
        if max(num_ridges_defl,num_ridges_inter)>num_ridges:
            num_ridges=max(num_ridges_defl,num_ridges_inter)
    num_plots=num_ridges
    if num_col>num_plots:
        num_col=num_plots
    num_rows=math.ceil(num_plots/num_col)
    titles=['Ridge {}'.format(r) for r in range(num_plots)]
    fig=make_subplots(rows=num_rows,cols=num_col,subplot_titles=titles)
    traces=[]
    first=True
    for r in range(num_plots):
        this_plot_num=r+1
        this_col=this_plot_num%num_col
        if num_rows<2:
            this_col=this_plot_num
        this_col=this_col if this_col else num_col
        this_row=((this_plot_num)//num_col)+1
        if (this_col-1) and (not (this_plot_num%num_col)):
            this_row-=1
        for d,i,n,color in zip(defl,inter,names,our_colors):
            this_ridge_defl=d.loc[d.loc[:,'ridge']==r,:]
            this_ridge_inter=i.loc[i.loc[:,'ridge']==r,:]
            this_ridge_defl.set_index(['ridge','path'])
            this_ridge_inter.set_index(['ridge','path'])
            all_everything=pd.concat([this_ridge_defl,this_ridge_inter],axis=1)
            scatter_args=dict(x=all_everything.loc[:,'deflection'],y=all_everything.loc[:,'interaction_time'],name=n,mode='markers',legendgroup=n,marker=dict(color=color))
            if first:
                scatter_args['showlegend']=True
            else:
                scatter_args['showlegend']=False
            fig.add_trace(go.Scatter(**scatter_args),row=this_row,col=this_col)
        first=False
    fig.update_yaxes(title='Interaction Time')
    fig.update_xaxes(title='Deflection (µm)')
    py.plot(fig,filename=filename)
