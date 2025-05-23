import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt

from . import session

def offset_stats(sess_df, load_row):
    stats_df = {'fly_id': [],
           'cl': [],
           'rnai': [],
            'dark': [],
           'offset_var':[],
           'offset_mean': [],
           'offset_mag': []}


    for _,row in sess_df.iterrows():
        
        
        ts = session.GetTS(load_row(row))

        stats_df['fly_id'].append(row['fly_id'])
        stats_df['cl'].append(row['closed_loop'])
        stats_df['dark'].append(row['dark'])
        stats_df['rnai'].append(row['rnai'])


        stats_df['offset_var'].append(sp.stats.circvar(ts.offset))

        offset_c = ts.offset_c.mean()
        stats_df['offset_mean'].append(np.angle(offset_c))
        stats_df['offset_mag'].append( np.abs(offset_c))

    return pd.DataFrame.from_dict(stats_df)
    
    
def offset_stats_unique(stats_df):
        #reduce to one entry per condition per fly by averaging
    stats_df_unique = {'fly_id': [],
                    'cl': [],
                    'rnai': [],
                    'dark': [],
                    'offset_var':[],
                    'offset_var_logit':[],
                    'offset_mean':[],
                    'offset_mag':[],
                    }

    fly_ids = stats_df['fly_id'].unique()
    for r, fly in enumerate(fly_ids):
        
        cl_mask = (stats_df['fly_id']==fly)*(stats_df['cl']>1) # closed_loop ==1 is the very first experience in closed loop
                                                            # closed_loop >1 takes data where fly has at least 10 min of 
                                                            # experience prior to imaging
        dark_mask = (stats_df['fly_id']==fly)*(stats_df['dark']>=1) 
        if (cl_mask.sum()>0) and (dark_mask.sum()>0): # take only flies with both closed loop and dark data
            
            rnai = stats_df['rnai'].loc[cl_mask]
            
            cl = stats_df['offset_var'].loc[cl_mask].mean() #average across sessions
            cl_mu = sp.stats.circmean(stats_df['offset_mean'].loc[cl_mask])

            stats_df_unique['fly_id'].append(fly)
            stats_df_unique['cl'].append(1)
            stats_df_unique['rnai'].append(rnai.iloc[0])
            stats_df_unique['dark'].append(0)
            stats_df_unique['offset_var'].append(cl)
            stats_df_unique['offset_var_logit'].append(sp.special.logit(cl)) # logit transform for mixed effects model below
            stats_df_unique['offset_mean'].append(cl_mu)
            stats_df_unique['offset_mag'].append(stats_df['offset_mag'].loc[cl_mask].mean())
        
            dark = stats_df['offset_var'].loc[dark_mask].mean() # average across flies
            dark_mu = sp.stats.circmean(stats_df['offset_mean'].loc[dark_mask])
            
            stats_df_unique['fly_id'].append(fly)
            stats_df_unique['cl'].append(0)
            stats_df_unique['rnai'].append(rnai.iloc[0])
            stats_df_unique['dark'].append(1)
            stats_df_unique['offset_var'].append(dark)
            stats_df_unique['offset_var_logit'].append(sp.special.logit(dark)) # logit tranform for mixed effect model below
            stats_df_unique['offset_mean'].append(dark_mu)
            stats_df_unique['offset_mag'].append(stats_df['offset_mag'].loc[dark_mask].mean())
            
    return pd.DataFrame.from_dict(stats_df_unique)

def offset_stats_plot(stats_df_unique):
    # reformat for plotting only
    stats_df_plot = {'fly_id': [],
                    'rnai': [],
                    'offset_var_dark':[],
                    'offset_var_closed_loop':[],
                    }
    fly_ids = stats_df_unique['fly_id'].unique()
    for fly in fly_ids:
        stats_df_plot['fly_id'].append(fly)
        
        fly_mask = stats_df_unique['fly_id']==fly
        
        _df = stats_df_unique.loc[fly_mask]
        stats_df_plot['rnai'].append(_df['rnai'].iloc[0])
        
        _df_d = _df.loc[_df['dark']==1]
        stats_df_plot['offset_var_dark'].append(_df_d['offset_var'])
        
        _df_cl = _df.loc[_df['dark']==0]
        stats_df_plot['offset_var_closed_loop'].append(_df_cl['offset_var'])
    return pd.DataFrame.from_dict(stats_df_plot)


def rho_stats(sess_df, load_row, dh_bins):
    stats_df = {'fly_id': [],
            'cl': [],
            'rnai': [],
            'dark': [],
            'rho_dig': [],
            }

    for _,row in sess_df.iterrows():
        
            
        ts = session.GetTS(load_row(row))
        
        
        dh = np.diff(np.unwrap(ts.heading_sm))/ts.dt
        dh = np.concatenate([[0],dh])
        
        dh_dig = np.digitize(np.abs(dh), dh_bins) - 1

        rho_dig = np.array([ts.rho[dh_dig == i].mean() for i in range(len(dh_bins))])
        
        stats_df['fly_id'].append(row['fly_id'])
        stats_df['cl'].append(row['closed_loop'])
        stats_df['dark'].append(row['dark'])
        stats_df['rnai'].append(row['rnai'])
        stats_df['rho_dig'].append(rho_dig)
        
    return pd.DataFrame.from_dict(stats_df)
        
        
def reformat_rho_stats_for_reg(grouped_stats, dh_bins):
    reg_df = {'fly_id': [],
           'rnai': [],
           'dark': [],
           'rho': [],
           'dh': [],
           }
    for _, row in grouped_stats.iterrows():
        # print(row)
        for i, dh in enumerate(dh_bins):
            rho = row['rho_dig'][i]
            if ~np.isnan(rho):
                reg_df['fly_id'].append(row['fly_id'])
                reg_df['rnai'].append(row['rnai'])
                reg_df['dark'].append(row['dark'])
                reg_df['rho'].append(rho)
                reg_df['dh'].append(dh)
        

    reg_df = pd.DataFrame.from_dict(reg_df)

    reg_df.reset_index()
    return reg_df


def dphi_stats(sess_df, load_row, dh_bins):
    stats_df = {'fly_id': [],
           'cl': [],
           'rnai': [],
           'dark': [],
           'dh_dig': [],
           'dphi_dig':[],
           }

    for _,row in sess_df.iterrows():
        
        
        ts = session.GetTS(load_row(row),
                            t_sigma=.1, h_sigma=.1)
        
        # bar_vis_mask = np.abs(ts.heading)<(3/4*np.pi)
        dh =np.diff(np.unwrap(ts.heading_sm))/ts.dt
        
        
        d_phi =  np.diff(np.unwrap(ts.phi))/ts.dt
        
        # dh, d_phi = dh[bar_vis_mask[1:]], d_phi[bar_vis_mask[1:]]
        
        dh_dig = np.digitize(dh, dh_bins) -1

        dphi_dig = np.array([d_phi[dh_dig == i].mean() for i in range(len(dh_bins))])
        # nan_mask = np.isnan(dphi_dig)
        # dphi_dig = sp.interpolate.interp1d(dh_bins[:-1][~nan_mask], dphi_dig[~nan_mask], kind='linear', bounds_error=False)(dh_bins)

        stats_df['fly_id'].append(row['fly_id'])
        stats_df['cl'].append(row['closed_loop'])
        stats_df['dark'].append(row['dark'])
        stats_df['rnai'].append(row['rnai'])
        stats_df['dh_dig'].append(dh_dig)
        stats_df['dphi_dig'].append(dphi_dig)
    return pd.DataFrame.from_dict(stats_df)
        
        
def reformat_dphi_stats_for_reg(grouped_stats, dh_bins):
    reg_df = {'fly_id': [],
           'rnai': [],
           'dark': [],
           'dphi': [],
           'dh': [],
           }
    for _, row in grouped_stats.iterrows():
        # print(row)
        for i, dh in enumerate(dh_bins):
            dphi = row['dphi_dig'][i]
            if ~np.isnan(dphi):
                reg_df['fly_id'].append(row['fly_id'])
                reg_df['rnai'].append(row['rnai'])
                reg_df['dark'].append(row['dark'])
                reg_df['dphi'].append(dphi)
                reg_df['dh'].append(dh)
        

    reg_df = pd.DataFrame.from_dict(reg_df)

    reg_df.reset_index()
    return reg_df

def plot_sess_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times = np.arange(0, 360, 60)):
    fig, ax = plt.subplots(2, 2, figsize=[15,4], sharey=True, sharex=True)
    
    
    def get_time_ticks_inds(time, plot_times):
        inds = []
        for t in plot_times:
            inds.append(np.argmin(np.abs(time-t)))
        return inds
        
    
    def plot_row(key, row, cmap):
        dff = ts_dict[key].dff
        time = ts_dict[key].time
        x = np.arange(dff.shape[1])
        heading_ = (ts_dict[key].heading+np.pi)/(2*np.pi)*15
        h = ax[row,0].imshow(dff, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(h, ax=ax[row,0])
        
        ax[row,0].scatter(x, heading_, color='orange', s=5)
        
        h = ax[row,1].imshow(ts_dict[key].dff_h_aligned, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(h, ax=ax[row,1])
        ax[row,1].scatter(x, 7.5*np.ones_like(heading_), color='orange', s=5)
        
        ax[row,0].set_ylabel('ROIs')
        ax[row,0].set_yticks([-0.5,7.5,15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        
        ax[row,0].set_xticks(get_time_ticks_inds(time, plot_times), labels=plot_times)
        ax[row,0].set_xlabel('Time (s)')
        
        ax[row,0].set_title(key)
        
        
        
    
    plot_row('closed_loop', 0, 'Greys')
    plot_row('dark', 1, 'Purples')
    fig.suptitle(ts_dict['fly'])
    
    fig.tight_layout()
    
    return fig, ax

def plot_sess_histograms(ts_dict, bins = np.linspace(-np.pi, np.pi, num=17)):
    
    fig_hist, ax_hist = plt.subplots()
    fig_polar, ax_polar = plt.subplots(subplot_kw={'projection':'polar'})
    centers = (bins[1:]+bins[:-1])/2
    def plot_hist(key, color):
        offset = ts_dict[key].offset
        hist, _ = np.histogram(offset, bins=bins)
        hist = hist/hist.sum()
        
        ax_hist.fill_between(centers, hist, color=color, alpha=.4)
        
        offset_c_mu = ts_dict[key].offset_c.mean()
        ax_polar.plot(np.angle(offset_c_mu)*np.ones([2,]), [0, np.abs(offset_c_mu)], color=color, linewidth=2, label=key)
        
        
    plot_hist('closed_loop', 'black')
    plot_hist('dark', plt.cm.Purples(.8))
    
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.set_xticks([-np.pi,0,np.pi], labels=[r'-$\pi$', '0', r'$\pi$'])
    ax_hist.set_xlabel('Offset')
    ax_hist.set_yticks([0,.05, .1,.15, .2, .25])
    ax_hist.set_ylim([0,.25])
    ax_hist.set_ylabel('Proportion')
    ax_hist.set_title(ts_dict['fly'])        
    fig_hist.tight_layout()    
    
    ax_polar.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['0', r'$\pi$/2', r'$\pi$', r'3$\pi$/2'])
    ax_polar.set_yticks([0,.2, .4, .6, .8])
    ax_polar.set_ylim([0,.8])
    ax_polar.set_title(ts_dict['fly'])
    ax_polar.legend()
    fig_polar.tight_layout()
    
    return (fig_hist, ax_hist), (fig_polar, ax_polar)