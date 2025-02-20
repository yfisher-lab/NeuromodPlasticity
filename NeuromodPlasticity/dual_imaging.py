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
        ts = session.GetTS(load_row(row))

        stats_df['fly_id'].append(row['fly_id'])
        stats_df['cl'].append(row['cl'])

        mean_offsets = np.angle(ts.offset_c.mean(axis=-1))
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