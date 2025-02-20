import pathlib
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import SessionTools.two_photon as st2p
from . import session





def offset_stats(sess_df, load_row):
    stats_df = {'fly_id': [],
                'cl': [],
                'offset_ch1': [],
                'offset_ch2': [],
                'offset_var_ch1': [],
                'offset_var_ch2': [],
                'offset_diff': [],
                'abs_offset_diff': [],
                'fwhm_ch1': [],
                'fwhm_ch2': []}
    
    for _, row in sess_df.iterrows():
        ts = session.GetTS(load_row(row), channels=[0,1])

        stats_df['fly_id'].append(row['fly_id'])
        stats_df['cl'].append(row['closed_loop'])

        mean_offsets = np.angle(ts.offset_c.mean(axis=-1))
        # print(mean_offsets)
        stats_df['offset_ch1'].append(mean_offsets[0])
        stats_df['offset_ch2'].append(mean_offsets[1])

        offset_var = sp.stats.circvar(ts.offset,axis=-1)
        stats_df['offset_var_ch1'].append(offset_var[0])
        stats_df['offset_var_ch2'].append(offset_var[1])

        offset_diff = np.angle(np.exp(1j*(ts.offset[0]-ts.offset[1])))
        stats_df['offset_diff'].append(sp.stats.circmean(offset_diff))
        stats_df['abs_offset_diff'].append(sp.stats.circmean(np.abs(offset_diff)))

        stats_df['fwhm_ch1'].append(ts.fwhm[0])
        stats_df['fwhm_ch2'].append(ts.fwhm[1])

    return pd.DataFrame(stats_df)

def offset_stats_unique(stats_df):
    stats_df_unique = {'fly_id': [],
                'cl': [],
                'offset_ch1': [],
                'offset_ch2': [],
                'offset_var_ch1': [],
                'offset_var_ch2': [],
                'offset_diff': [],
                'abs_offset_diff': [],
                'fwhm_ch1': [],
                'fwhm_ch2': []}
    
    for fly in stats_df['fly_id'].unique():

        cl_mask = (stats_df['fly_id']==fly)*(stats_df['cl']>=1) # closed_loop ==1 is the very first experience in closed loop
                                                            # closed_loop >1 takes data where fly has at least 10 min of 
                                                            # experience prior to imaging
        dark_mask = (stats_df['fly_id']==fly)*(stats_df['cl']==0) 
        if (cl_mask.sum()>0) and (dark_mask.sum()>0): # take only flies with both closed loop and dark data
            
            stats_df_unique['fly_id'].append(fly)
            stats_df_unique['cl'].append(1)
            stats_df_unique['offset_ch1'].append(sp.stats.circmean(stats_df.loc[cl_mask, 'offset_ch1']))
            stats_df_unique['offset_ch2'].append(sp.stats.circmean(stats_df.loc[cl_mask, 'offset_ch2']))

            stats_df_unique['offset_var_ch1'].append(stats_df.loc[cl_mask, 'offset_var_ch1'].mean())
            stats_df_unique['offset_var_ch2'].append(stats_df.loc[cl_mask, 'offset_var_ch2'].mean())

            stats_df_unique['offset_diff'].append(sp.stats.circmean(stats_df.loc[cl_mask, 'offset_diff']))
            stats_df_unique['abs_offset_diff'].append(stats_df.loc[cl_mask, 'abs_offset_diff'].mean())

            stats_df_unique['fwhm_ch1'].append(stats_df.loc[cl_mask, 'fwhm_ch1'].mean())
            stats_df_unique['fwhm_ch2'].append(stats_df.loc[cl_mask, 'fwhm_ch2'].mean())

            stats_df_unique['fly_id'].append(fly)
            stats_df_unique['cl'].append(0)
            stats_df_unique['offset_ch1'].append(sp.stats.circmean(stats_df.loc[dark_mask, 'offset_ch1']))
            stats_df_unique['offset_ch2'].append(sp.stats.circmean(stats_df.loc[dark_mask, 'offset_ch2']))

            stats_df_unique['offset_var_ch1'].append(stats_df.loc[dark_mask, 'offset_var_ch1'].mean())
            stats_df_unique['offset_var_ch2'].append(stats_df.loc[dark_mask, 'offset_var_ch2'].mean())

            stats_df_unique['offset_diff'].append(sp.stats.circmean(stats_df.loc[dark_mask, 'offset_diff']))
            stats_df_unique['abs_offset_diff'].append(stats_df.loc[dark_mask, 'abs_offset_diff'].mean())

            stats_df_unique['fwhm_ch1'].append(stats_df.loc[dark_mask, 'fwhm_ch1'].mean())
            stats_df_unique['fwhm_ch2'].append(stats_df.loc[dark_mask, 'fwhm_ch2'].mean())
    return pd.DataFrame(stats_df_unique)

