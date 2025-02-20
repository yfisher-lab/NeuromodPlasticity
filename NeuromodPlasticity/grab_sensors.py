import pathlib
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt

import SessionTools.two_photon as st2p
from . import session

class SessMaker():

    def __init__(self, basedir=None, figfolder=None, mkdir=True, id_suffix=None):

        self.basedir = pathlib.Path(basedir)
        self.sess_df = pd.read_csv(self.basedir / 'sessions.csv')
        self._id_suffix=id_suffix
        self.fig_folder = pathlib.Path(figfolder)
        if mkdir:
            self.fig_folder.mkdir(parents=True, exist_ok=True)

        self.add_fly_id()
        self.filter_usable()
        

    def add_fly_id(self):
        fly_id = []
        if self._id_suffix is None:
            for _, row in self.sess_df.iterrows():
                fly_id.append(row['date'] + '_' + row['fly'])
        else:
            for _, row in self.sess_df.iterrows():
                fly_id.append(row['date'] + '_' + row['fly'] + '_' + self._id_suffix)
            self.sess_df['grab'] = self._id_suffix
        
        self.sess_df['fly_id'] = fly_id

    def filter_usable(self):
        self.sess_df = self.sess_df.loc[self.sess_df['usable']==1]

    def load_row(self, row):
        outdir = pathlib.PurePath(self.basedir / row['date'] / row['fly'] / row['session'] / 'preprocess.pkl')
        return st2p.preprocessing.EBImagingSession.from_file(outdir)
    
    @property
    def fly_ids(self):
        return self.sess_df['fly_id'].unique()



def offset_stats(sess_cls):
    stats_df = {'fly_id': [],
                'cl': [],
                'offset_var':[],
                'offset_mean': [],
                'offset_mag': []}


    for _,row in sess_cls.sess_df.iterrows():
        
        
        ts = session.GetTS(sess_cls.load_row(row))

        stats_df['fly_id'].append(row['fly_id'])
        stats_df['cl'].append(row['closed_loop'])
         
        stats_df['offset_var'].append(sp.stats.circvar(ts.offset))

        offset_c = ts.offset_c.mean()
        stats_df['offset_mean'].append(np.angle(offset_c))
        stats_df['offset_mag'].append( np.abs(offset_c))

    return pd.DataFrame.from_dict(stats_df)
    
    
def offset_stats_unique(stats_df):
        #reduce to one entry per condition per fly by averaging
    stats_df_unique = {'fly_id': [],
                        'cl': [],
                        'dark': [],
                        'offset_var':[],
                        'offset_var_logit':[],
                        'offset_mean':[],
                        'offset_mag':[],
                        }

    fly_ids = stats_df['fly_id'].unique()
    for r, fly in enumerate(fly_ids):
        
        cl_mask = (stats_df['fly_id']==fly)*(stats_df['cl']>=1) # closed_loop ==1 is the very first experience in closed loop
                                                            # closed_loop >1 takes data where fly has at least 10 min of 
                                                            # experience prior to imaging
        dark_mask = (stats_df['fly_id']==fly)*(stats_df['cl']==0) 
        if (cl_mask.sum()>0) and (dark_mask.sum()>0): # take only flies with both closed loop and dark data
            
            
            cl = stats_df['offset_var'].loc[cl_mask].mean() #average across sessions
            cl_mu = sp.stats.circmean(stats_df['offset_mean'].loc[cl_mask])

            stats_df_unique['fly_id'].append(fly)
            stats_df_unique['cl'].append(1)
            stats_df_unique['dark'].append(0)
            stats_df_unique['offset_var'].append(cl)
            stats_df_unique['offset_var_logit'].append(sp.special.logit(cl)) # logit transform for mixed effects model below
            stats_df_unique['offset_mean'].append(cl_mu)
            stats_df_unique['offset_mag'].append(stats_df['offset_mag'].loc[cl_mask].mean())
        
            dark = stats_df['offset_var'].loc[dark_mask].mean() # average across flies
            dark_mu = sp.stats.circmean(stats_df['offset_mean'].loc[dark_mask])
            
            stats_df_unique['fly_id'].append(fly)
            stats_df_unique['cl'].append(0)
            stats_df_unique['dark'].append(1)
            stats_df_unique['offset_var'].append(dark)
            stats_df_unique['offset_var_logit'].append(sp.special.logit(dark)) # logit tranform for mixed effect model below
            stats_df_unique['offset_mean'].append(dark_mu)
            stats_df_unique['offset_mag'].append(stats_df['offset_mag'].loc[dark_mask].mean())
            
    return pd.DataFrame.from_dict(stats_df_unique)

def offset_stats_plot(stats_df_unique):
    # reformat for plotting only
    stats_df_plot = {'fly_id': [],
                    'offset_var_dark':[],
                    'offset_var_closed_loop':[],
                    }
    fly_ids = stats_df_unique['fly_id'].unique()
    for fly in fly_ids:
        stats_df_plot['fly_id'].append(fly)
        
        fly_mask = stats_df_unique['fly_id']==fly
        
        _df = stats_df_unique.loc[fly_mask]
        
        _df_d = _df.loc[_df['dark']==1]
        stats_df_plot['offset_var_dark'].append(_df_d['offset_var'])
        
        _df_cl = _df.loc[_df['dark']==0]
        stats_df_plot['offset_var_closed_loop'].append(_df_cl['offset_var'])
    return pd.DataFrame.from_dict(stats_df_plot)


def rho_stats(sess_cls, dh_bins):
    stats_df = {'fly_id': [],
            'cl': [],
            'rho_dig': [],
            'F_dig': [],
            }

    for _,row in sess_cls.sess_df.iterrows():
        
            
        ts = session.GetTS(sess_cls.load_row(row))
        
        
        dh = np.diff(np.unwrap(ts.heading_sm))/ts.dt
        dh = np.concatenate([[0],dh])
        
        dh_dig = np.digitize(np.abs(dh), dh_bins) - 1

        rho_dig = np.array([ts.rho[dh_dig == i].mean() for i in range(len(dh_bins))])
        F_dig = np.array([ts.outer_ring[dh_dig == i].mean() for i in range(len(dh_bins))])
        
        stats_df['fly_id'].append(row['fly_id'])
        stats_df['cl'].append(row['closed_loop'])
        stats_df['rho_dig'].append(rho_dig)
        stats_df['F_dig'].append(F_dig)
        
    return pd.DataFrame.from_dict(stats_df)
        
        
def reformat_rho_stats_for_reg(grouped_stats, dh_bins):
    reg_df = {'fly_id': [],
           'dark': [],
           'rho': [],
           'F': [],
           'dh': [],
           }
    for _, row in grouped_stats.iterrows():
        # print(row)
        for i, dh in enumerate(dh_bins):
            rho = row['rho_dig'][i]
            F = row['F_dig'][i]
            if ~np.isnan(rho):
                reg_df['fly_id'].append(row['fly_id'])
                reg_df['dark'].append(row['dark'])
                reg_df['rho'].append(rho)
                reg_df['dh'].append(dh)
                reg_df['F'].append()

    reg_df = pd.DataFrame.from_dict(reg_df)

    reg_df.reset_index()
    return reg_df
