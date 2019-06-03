import numpy as np
import pandas as pd
import shapely.geometry

def get_interaction_time(cal_paths,cell_size,in_gutter_rm=True):
    paths_frame=cal_paths.to_df()
    if in_gutter_rm:
        paths_frame=paths_frame.loc[~paths_frame.loc[:,'in_gutter'],:]
    #Get the interaction time of each path at each ridge
    output_frames=[]
    ridge_sep=cal_paths.cal_device.ridge_coords[0][0][0]-cal_paths.cal_device.ridge_coords[1][0][0]
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
                #get the first time the cell is within one diameter of the ridge
                interacting=approaching_this_ridge.loc[approaching_this_ridge.loc[:,'dist_next_ridge']<cell_size,:]
                under_ridge=this_path.loc[this_path.loc[:,'under_ridge']==ridge,:]
                if interacting.shape[0]:
                    time_start=interacting.iloc[0]['t']
                elif under_ridge.shape[0]>1:
                    
                    time_start=under_ridge.iloc[0]['t']
                    #we can no longer use this point for time_end
                    under_ridge=under_ridge.iloc[1:]
                else:
                    under_ridge=pd.DataFrame() #this will force the next if statement to register that we have a time of 0
                #now find the last time the cell was under the ridge

                if under_ridge.shape[0]<1:
                    output_frames.append(pd.DataFrame(dict(path=[path],ridge=[ridge],interaction_time=[0])))
                else:
                    time_end=under_ridge.iloc[under_ridge.shape[0]-1]['t']
                    output_frames.append(pd.DataFrame(dict(path=[path],ridge=[ridge],interaction_time=[time_end-time_start])))

    return pd.concat(output_frames,ignore_index=True)

def get_defl_per_ridge(cal_paths):
    paths_frame=cal_paths.to_df()
    y_min=min(paths_frame.loc[:'y'])*1.1
    y_max=max(paths_frame.loc[:,'y'])*1.1
    ridge_sep=cal_paths.cal_device.ridge_coords[0][0][0]-cal_paths.cal_device.ridge_coords[1][0][0]
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

    ridge_offsets=cal_paths.cal_device.get_ridge_offsets(ridge_sep/2)

    ridge_offsets=[[extend_linestring(up,y_min,y_max),extend_linestring(down,y_min,y_max)] for up,down in ridge_offsets]
    path=[]
    ridge=[]
    defl=[]
    for p in set(paths_frame.loc[:,'path']):
        this_path=paths_frame.loc[paths_frame.loc[:,'path']==p,:]
        path_coords=zip(this_path.loc[:,'x'],this_path.loc[:,'y'])
        path_linestring=shapely.geometry.LineString(list(path_coords))
        for r in range(num_ridges):
            if path_linestring.intersects(ridge_offsets[r][0]) and path_linestring.intersects(ridge_offsets[r][1]):
                #we have data between one offset before and one after this ridge use the last intersecting point upstream and the first downstream to calc deflection
                upstream=path_linestring.intersection(ridge_offsets[r][0])
                downstream=path_linestring.intersection(ridge_offsets[r][1])
                if (type(upstream)!=shapely.geometry.point.Point) or (type(downstream)!=shapely.geometry.point.Point):
                    raise Exception('more than one intersection with ridge offsets')
                this_defl=downstream.coords[1]-upstream.coords[1]
                path.append(p)
                ridge.append(r)
                defl.append(this_defl)
    return pd.DataFrame(dict(path=path,ridge=ridge,deflection=deflection))
