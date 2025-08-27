import pathlib
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt

from matplotlib.gridspec import GridSpec as GS
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


import SessionTools.two_photon as st2p
from . import session


def offset_stats_DTE(data, channel_lookup):
    stats_df = {'fly_id': [],
                'cl': [],
                'offset_ch1': [],
                'offset_ch2': [],
                'offset_var_ch1': [],
                'offset_var_ch2': [],
                'offset_diff': [],
                'abs_offset_diff': [],
                'pva_diff': [],
                'fwhm_ch1': [],
                'fwhm_ch2': []}
    

    for fly, multitrial_dict in data.items():
        flyname = fly.split("\\")[-1]
        for trial_name, trial_dict in multitrial_dict.items():
            ts = session.GetTS_DTE(trial_dict, channel_lookup[fly], dh_sigma=.3)

            stats_df['fly_id'].append(flyname)
            stats_df['cl'].append(1)

            mean_offsets = np.angle(ts.offset_c.mean(axis=-1))
            # print(mean_offsets)
            stats_df['offset_ch1'].append(mean_offsets[0])
            stats_df['offset_ch2'].append(mean_offsets[1])

            offset_var = sp.stats.circvar(ts.offset,axis=-1)
            stats_df['offset_var_ch1'].append(offset_var[0])
            stats_df['offset_var_ch2'].append(offset_var[1])

            offset_diff = np.angle(np.exp(1j*(ts.offset[0,:]-ts.offset[1,:])))
            stats_df['offset_diff'].append(sp.stats.circmean(offset_diff))
            stats_df['abs_offset_diff'].append(sp.stats.circmean(np.abs(offset_diff)))

            pva_diff = np.abs(np.angle(np.exp(1j*np.diff(ts.phi,axis=0)))).mean()
            stats_df['pva_diff'].append(pva_diff)

            stats_df['fwhm_ch1'].append(ts.fwhm[0])
            stats_df['fwhm_ch2'].append(ts.fwhm[1])

    return pd.DataFrame(stats_df)

def offset_stats(sess_df, load_row):
    stats_df = {'fly_id': [],
                'cl': [],
                'offset_ch1': [],
                'offset_ch2': [],
                'offset_var_ch1': [],
                'offset_var_ch2': [],
                'offset_diff': [],
                'abs_offset_diff': [],
                'pva_diff': [],
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

        offset_diff = np.angle(np.exp(1j*(ts.offset[0,:]-ts.offset[1,:])))
        stats_df['offset_diff'].append(sp.stats.circmean(offset_diff))
        stats_df['abs_offset_diff'].append(sp.stats.circmean(np.abs(offset_diff)))

        pva_diff = np.abs(np.angle(np.exp(1j*np.diff(ts.phi,axis=0)))).mean()
        stats_df['pva_diff'].append(pva_diff)

        stats_df['fwhm_ch1'].append(ts.fwhm[0])
        stats_df['fwhm_ch2'].append(ts.fwhm[1])

    return pd.DataFrame(stats_df)

def offset_stats_unique(stats_df, cl_only=False):
    stats_df_unique = {'fly_id': [],
                'cl': [],
                'offset_ch1': [],
                'offset_ch2': [],
                'offset_var_ch1': [],
                'offset_var_ch2': [],
                'offset_diff': [],
                'abs_offset_diff': [],
                'pva_diff': [],
                'fwhm_ch1': [],
                'fwhm_ch2': []}
    
    for fly in stats_df['fly_id'].unique():

        cl_mask = (stats_df['fly_id']==fly)*(stats_df['cl']>=1) # closed_loop ==1 is the very first experience in closed loop
                                                            # closed_loop >1 takes data where fly has at least 10 min of 
                                                            # experience prior to imaging
        dark_mask = (stats_df['fly_id']==fly)*(stats_df['cl']==0) 

        if cl_only:
            mask_sum = cl_mask.sum()>0
        else:
            mask_sum = (cl_mask.sum()>0) and (dark_mask.sum()>0)
        if mask_sum: # take only flies with both closed loop and dark data
            
            stats_df_unique['fly_id'].append(fly)
            stats_df_unique['cl'].append(1)
            stats_df_unique['offset_ch1'].append(sp.stats.circmean(stats_df.loc[cl_mask, 'offset_ch1']))
            stats_df_unique['offset_ch2'].append(sp.stats.circmean(stats_df.loc[cl_mask, 'offset_ch2']))

            stats_df_unique['offset_var_ch1'].append(stats_df.loc[cl_mask, 'offset_var_ch1'].mean())
            stats_df_unique['offset_var_ch2'].append(stats_df.loc[cl_mask, 'offset_var_ch2'].mean())

            stats_df_unique['offset_diff'].append(np.angle(np.exp(1j*stats_df.loc[cl_mask, 'offset_diff']).mean()))
            stats_df_unique['abs_offset_diff'].append(stats_df.loc[cl_mask, 'abs_offset_diff'].mean())

            stats_df_unique['pva_diff'].append(stats_df.loc[cl_mask, 'pva_diff'].mean())

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

            stats_df_unique['pva_diff'].append(stats_df.loc[dark_mask, 'pva_diff'].mean())

            stats_df_unique['fwhm_ch1'].append(stats_df.loc[dark_mask, 'fwhm_ch1'].mean())
            stats_df_unique['fwhm_ch2'].append(stats_df.loc[dark_mask, 'fwhm_ch2'].mean())
    return pd.DataFrame.from_dict(stats_df_unique)

def rho_stats(sess_df, load_row, dh_bins, abs=True):
    stats_df = {'fly_id': [],
                'cl': [],
                'rho1_dig': [],
                'rho2_dig': [],
                'pva_diff': [],
                }
    for _, row in sess_df.iterrows():
        ts = session.GetTS(load_row(row), channels=[0,1], dh_sigma=.3)

        stats_df['fly_id'].append(row['fly_id'])
        stats_df['cl'].append(row['closed_loop'])

        # dh = np.diff(np.unwrap(ts.heading_sm))/ts.dt
        # dh = np.concatenate([[0],dh])
        dh = ts.dh
        if abs:
            dh_dig = np.digitize(np.abs(dh), dh_bins) - 1
        else:
            dh_dig = np.digitize(dh, dh_bins) - 1

        rho1_dig = np.array([ts.rho[0, dh_dig == i].mean() for i in range(len(dh_bins))])
        stats_df['rho1_dig'].append(rho1_dig)

        rho2_dig = np.array([ts.rho[1, dh_dig == i].mean() for i in range(len(dh_bins))])
        stats_df['rho2_dig'].append(rho2_dig)

        if abs:
            pvd = np.abs(np.angle(np.exp(1j*np.diff(ts.phi,axis=0)))).ravel()
        else:
            pvd = np.angle(np.exp(1j*np.diff(ts.phi,axis=0))).ravel()
        pva_diff = np.array([pvd[(dh_dig==i)].mean() for i in range(len(dh_bins))])
        stats_df['pva_diff'].append(pva_diff)

    return pd.DataFrame.from_dict(stats_df)

def rho_stats_DTE(data, channel_lookup, dh_bins, abs=True):
    stats_df = {'fly_id': [],
                'cl': [],
                'rho1_dig': [],
                'rho2_dig': [],
                'pva_diff': [],
                }
    for fly, multitrial_dict in data.items():
        for trial_name, trial_dict in multitrial_dict.items():
            ts = session.GetTS_DTE(trial_dict, channel_lookup[fly], dh_sigma=.3)
        

            stats_df['fly_id'].append(fly)
            stats_df['cl'].append(1)

        
            dh = ts.dh
            if abs:
                dh_dig = np.digitize(np.abs(dh), dh_bins) - 1
            else:
                dh_dig = np.digitize(dh, dh_bins) - 1

            rho1_dig = np.array([ts.rho[0, dh_dig == i].mean() for i in range(len(dh_bins))])
            stats_df['rho1_dig'].append(rho1_dig)

            rho2_dig = np.array([ts.rho[1, dh_dig == i].mean() for i in range(len(dh_bins))])
            stats_df['rho2_dig'].append(rho2_dig)

            if abs:
                pvd = np.abs(np.angle(np.exp(1j*np.diff(ts.phi,axis=0)))).ravel()
            else:
                pvd = np.angle(np.exp(1j*np.diff(ts.phi,axis=0))).ravel()
            pva_diff = np.array([pvd[dh_dig==i].mean() for i in range(len(dh_bins))])
            stats_df['pva_diff'].append(pva_diff)

    return pd.DataFrame.from_dict(stats_df)

def reformat_rho_stats():
    pass

def pvdiff_rho_stats(sess_df, load_row, pv_bins):
    stats_df = {'fly_id': [],
                'cl': [],
                'pva_diff': [],
                }
    
    for _, row in sess_df.iterrows():
        ts = session.GetTS(load_row(row), channels=[0,1])

        stats_df['fly_id'].append(row['fly_id'])
        stats_df['cl'].append(row['closed_loop'])

        pva_diff = np.abs(np.angle(np.exp(1j*np.diff(ts.phi,axis=0)))).ravel()
        rho_dig = np.digitize(ts.rho.mean(axis=0), bins=pv_bins)
        pva_diff = np.array([pva_diff[rho_dig==i].mean() for i in range(pv_bins.shape[0])])
        stats_df['pva_diff'].append(pva_diff)
    return pd.DataFrame.from_dict(stats_df)

def reformat_pvdiff_rho_stats():
    pass


def cross_corr_stats(sess_df, load_row, delays, times):
    r_df = {'fly': [],
            'cl': [],
            'r': [],
            'argmax': []}
    delays = np.arange(-50,51)
    times = delays*.01
    for _, row in sess_df.iterrows():
        # cross correlation of pva values
        ts = session.GetTS(load_row(row), channels=(0,1))
        
        r = np.zeros_like(delays, dtype=float)
        for i, d in enumerate(delays):
            
            r[i] = np.abs(np.correlate(np.exp(1j*ts.phi[0,:]), np.exp(1j*np.roll(ts.phi[1,:],d)))/ts.phi.shape[-1])[0]
            # r[i] = np.abs(np.correlate(ts.phi[0,:], np.roll(ts.phi[1,:],d)/ts.phi.shape[-1])[0])

        r_df['fly'].append(row['fly_id'])
        r_df['cl'].append(row['closed_loop'])
        
    f = sp.interpolate.interp1d(delays*ts.dt, r, kind='linear', bounds_error=False, fill_value='extrapolate')
    r_df['r'].append(f(times))
    r_df['argmax'].append(np.argmax(f(times)))
    
    return pd.DataFrame.from_dict(r_df)


def plot_sess_heatmaps(ts, fly_id, sess_name, vmin=-.5, vmax=.5, plot_times = np.arange(0,360,60),
                       ch1_heatmap = 'Greys', ch2_heatmap = 'Greens'):

    fig, ax = plt.subplots(3,2, figsize=[15,6], sharey=True, sharex=True)

    def get_time_ticks_inds(time, plot_times):
        inds = []
        for t in plot_times:
            inds.append(np.argmin(np.abs(time-t)))
        return inds
    
    x = np.arange(ts.dff.shape[-1])

    heading_ = (ts.heading+np.pi)/(2*np.pi)*15

    h = ax[0, 0].imshow(ts.dff[0, :, :], aspect='auto', interpolation='none',
                        cmap=ch1_heatmap, vmin=vmin, vmax=vmax)
    ax[0, 0].scatter(x, heading_, s=5, color='orange')
    fig.colorbar(h, ax=ax[0,0])

    h = ax[0, 1].imshow(ts.dff_h_aligned[0, :, :], aspect='auto', interpolation='none',
                        cmap=ch1_heatmap, vmin=vmin, vmax=vmax)
    ax[0, 1].scatter(x, 7.5*np.ones_like(heading_), s=5, color='orange')
    fig.colorbar(h, ax=ax[0,1])

    h = ax[1,0].imshow(ts.dff[1, :, :], aspect='auto', interpolation='none',
                       cmap=ch2_heatmap, vmin=vmin, vmax=vmax)
    ax[1,0].scatter(x, heading_, s=5, c='orange')
    fig.colorbar(h, ax=ax[1,0])

    h = ax[1, 1].imshow(ts.dff_h_aligned[1, :, :], aspect='auto', interpolation='none',
                        cmap=ch2_heatmap, vmin=vmin, vmax=vmax)
    ax[1, 1].scatter(x, 7.5*np.ones_like(heading_), s=5, color='orange')
    fig.colorbar(h, ax=ax[1,1])

    ax[0,0].set_title('Ch 1')
    ax[1,0].set_title('Ch 2')

    phi_ = (ts.phi+np.pi)/2/np.pi*15
    cmap = plt.get_cmap(ch1_heatmap)
    ax[2,0].scatter(x, phi_[0,:], color=cmap(.8), s=5, alpha=.4)
    cmap = plt.get_cmap(ch2_heatmap)
    ax[2,0].scatter(x, phi_[1,:], color=cmap(.8), s=5, alpha=.4)
    fig.colorbar(h, ax=ax[2,0])

    phi_diff = np.angle(np.exp(1j*np.diff(ts.phi, axis=0)))
    phi_diff = (phi_diff+np.pi)/2/np.pi*15
    ax[2,1].scatter(x, phi_diff, c='blue', s=5)
    fig.colorbar(h, ax=ax[2,1])

    plot_times = plot_times[plot_times<ts.time.iloc[-1]]
    for a in ax.flatten():
        a.set_ylabel('ROIs')
        a.set_yticks([-0.5,7.5,15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        
        a.set_xticks(get_time_ticks_inds(ts.time, plot_times), labels=plot_times)
        a.set_xlabel('Time (s)')

    
    fig.suptitle(f'{fly_id} - {sess_name}')
    fig.tight_layout()

    return fig, ax
    
def plot_pva_diff_histograms(ts, fly_id, sess_name,
                             bins = np.linspace(-np.pi, np.pi, num=17), 
                             color='black'):
    
    fig, ax = plt.subplots()
    centers = (bins[1:] + bins[:-1])/2

    pva_diff = np.angle(np.exp(1j*(np.diff(ts.phi,axis=0))))

    hist, _ = np.histogram(pva_diff, bins=bins)
    hist = hist/hist.sum()

    ax.fill_between(bins[:-1], hist, color=color, alpha=.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([-np.pi,0,np.pi], labels=[r'-$\pi$', '0', r'$\pi$'])
    ax.set_xlabel(r'$\Delta$ PVA')
    ax.set_ylabel('Proportion')
    # ax.set_yticks([0,.05, .1,.15, .2, .25])
    # ax.set_ylim([0,.25])
    
    fig.suptitle(f'{fly_id} - {sess_name}')
    fig.tight_layout()
    return fig, ax


def plot_sess_histograms(ts, fly_id, sess_name, bins = np.linspace(-np.pi, np.pi, num=17), 
                         cmaps=('Greys', 'Greens')):
    
    fig_hist, ax_hist = plt.subplots()
    fig_polar, ax_polar = plt.subplots(subplot_kw={'projection':'polar'})
    centers = (bins[1:]+bins[:-1])/2


    def plot_hist(ch, cmap, hatch=None):
        offset = ts.offset[ch,:]
        hist, _ = np.histogram(offset, bins=bins)
        hist = hist/hist.sum()
        
        offset_c_mu = ts.offset_c[ch,:].mean()
        
        _cmap = plt.get_cmap(cmap)
        color = _cmap(.8)
        if hatch is not None:
            ax_hist.fill_between(centers, hist, color='none', alpha=.4, hatch=hatch, edgecolor=color)
            ax_polar.plot(np.angle(offset_c_mu)*np.ones([2,]), [0, np.abs(offset_c_mu)], color=color, linewidth=2,
                          linestyle='--', alpha=.4)
            
        else:
            ax_hist.fill_between(centers, hist, color=color, alpha=.4)
            ax_polar.plot(np.angle(offset_c_mu)*np.ones([2,]), [0, np.abs(offset_c_mu)], color=color, linewidth=2)
       
        
        
        
    
    fig_hist.suptitle(f'{fly_id} - {sess_name}')
    fig_polar.suptitle(f'{fly_id} - {sess_name}')
        
    plot_hist(0, cmaps[0])
    plot_hist(1, cmaps[1])  
        
    
    
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.set_xticks([-np.pi,0,np.pi], labels=[r'-$\pi$', '0', r'$\pi$'])
    ax_hist.set_xlabel('Offset')
    ax_hist.set_yticks([0,.05, .1,.15, .2, .25])
    ax_hist.set_ylim([0,.25])
    ax_hist.set_ylabel('Proportion')
    
    fig_hist.tight_layout()    
    
    ax_polar.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['0', r'$\pi$/2', r'$\pi$', r'3$\pi$/2'])
    ax_polar.set_yticks([0,.2, .4, .6, .8])
    
    # ax_polar.legend()
    fig_polar.tight_layout()
    
    return (fig_hist, ax_hist), (fig_polar, ax_polar)


def plot_transpose_heatmaps(ts, fly_id, sess_name, vmin=-.5, vmax=.5, plot_times = np.arange(0,360,60),
                       ch1_heatmap = 'Greys', ch2_heatmap = 'Greens'):

    fig = plt.figure(figsize=[9, 15])

    gs = GS(1,4, width_ratios=[2,2,1,1])
    
    ax = [fig.add_subplot(gs[0]), 
          fig.add_subplot(gs[1]), 
          fig.add_subplot(gs[2]), 
          fig.add_subplot(gs[3])]
    

    def get_time_ticks_inds(time, plot_times):
        inds = []
        for t in plot_times:
            inds.append(np.argmin(np.abs(time-t)))
        return inds
    
    x = np.arange(ts.dff.shape[-1])

    heading_ = (ts.heading+np.pi)/(2*np.pi)*15

    h = ax[0].imshow(ts.dff[0, :, :].T, aspect='auto', interpolation='none',
                     cmap=ch1_heatmap, vmin=vmin, vmax=vmax)
    ax[0].scatter(heading_, x, s=5, color='orange')
    fig.colorbar(h, ax=ax[0])

    
    h = ax[1].imshow(ts.dff[1, :, :].T, aspect='auto', interpolation='none',
                     cmap=ch2_heatmap, vmin=vmin, vmax=vmax)
    ax[1].scatter(heading_, x, s=5, c='orange')
    fig.colorbar(h, ax=ax[1])

    
    ax[0].set_title('Ch 1')
    ax[1].set_title('Ch 2')

    phi_ = (ts.phi+np.pi)/2/np.pi*15
    cmap = plt.get_cmap(ch1_heatmap)
    ax[2].scatter(phi_[0,:], x, color=cmap(.8), s=5, alpha=.4)
    cmap = plt.get_cmap(ch2_heatmap)
    ax[2].scatter(phi_[1,:], x, color=cmap(.8), s=5, alpha=.4)
    fig.colorbar(h, ax=ax[2])

    phi_diff = np.angle(np.exp(1j*np.diff(ts.phi, axis=0)))
    phi_diff = (phi_diff+np.pi)/2/np.pi*15
    ax[3].scatter(phi_diff, x, c='blue', s=5)
    fig.colorbar(h, ax=ax[3])

    plot_times = plot_times[plot_times<ts.time.iloc[-1]]
    for a in ax:
        a.set_xlabel('ROIs')
        a.set_xticks([-0.5,7.5,15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        
        a.set_yticks(get_time_ticks_inds(ts.time, plot_times), labels=plot_times)
        a.set_ylabel('Time (s)')

        a.set_ylim([ts.dff.shape[-1]-0.5, -.5])

    
    fig.suptitle(f'{fly_id} - {sess_name}')
    fig.tight_layout()

    return fig, ax
    


def plot_sess_heatmaps_w_hist(ts, fly_id, sess_name, vmin=-.5, vmax=3, plot_times = np.arange(0,360,60),
                       ch1_heatmap = 'Greys', ch2_heatmap = 'Greens', twindow=None,
                       bins = np.linspace(-np.pi, np.pi, num=17)):
    
    fig = plt.figure(figsize=[8,4])
    gs = GS(2,4, figure=fig, width_ratios=[6,1,.8, .2],wspace=.05,hspace=.8)
    heatmap_axs = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[1,0])]
    hist_axs = [fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,1])]
    cbar_ax = [fig.add_subplot(gs[i,3]) for i in range(2)]


    def get_time_ticks_inds(time, plot_times):
        inds = []
        for t in plot_times:
            inds.append(np.argmin(np.abs(time-t)))
        return inds
    
    def plot_row(ch, row, cmap, title, hatch=None):
        dff = ts.dff[ch,:,:]
        heading = ts.heading
        offset = ts.offset[ch,:]
        
        time = ts.time

        if twindow is not None:
            mask = (time>=twindow[0]) * (time<=twindow[1])
        else:
            mask = np.ones_like(time)>0

        dff= dff[:,mask]
        time = time[mask]
        heading = heading[mask]

        x = np.arange(dff.shape[1])
        heading_ = (heading + np.pi) / (2 * np.pi) * 15
        h = heatmap_axs[row].imshow(dff, aspect='auto', cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
        fig.colorbar(h, cax=cbar_ax[row])
        heatmap_axs[row].scatter(x, heading_, color='orange', s=5)

        heatmap_axs[row].set_ylabel('ROIs')
        heatmap_axs[row].set_yticks([-0.5,7.5,15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        heatmap_axs[row].yaxis.set_minor_locator(AutoMinorLocator())
        
        _plot_times = plot_times[plot_times<time.iloc[-1]]
        heatmap_axs[row].set_xticks(get_time_ticks_inds(time, _plot_times), labels=_plot_times)
        heatmap_axs[row].set_xlabel('Time (s)')
        
        heatmap_axs[row].set_title(title)

        
      
        centers = (bins[:-1] + bins[1:]) / 2
        hist, _ = np.histogram(offset, bins=bins)
        hist = hist / hist.sum()  # normalize
        if hatch is None:
            hist_axs[row].fill_betweenx(centers, 0, hist, color=cmap(.8), alpha=.5)
        else:
            hist_axs[row].fill_betweenx(centers, 0, hist, alpha=1, hatch=hatch,color='none', edgecolor=cmap(.8))

        # hist_axs[row].set_yticks([-np.pi,-3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], 
        #                          labels=[r'-$\pi$', '', '', '', r'0','', '', '', r'$\pi$'])
        hist_axs[row].set_yticks([-np.pi, 0,  np.pi], 
                                 labels=[r'-$\pi$',  r'0', r'$\pi$'])
        hist_axs[row].yaxis.set_minor_locator(AutoMinorLocator())
        hist_axs[row].set_ylim([np.pi, -np.pi])
        # hist_axs[row].set_xlim(left=0)
        hist_axs[row].grid(True, axis='y', linestyle='-', alpha=0.8,linewidth=2.5, which='major')
        ygridlines = hist_axs[row].get_ygridlines()
        ygridlines[1].set_color('orange')
        hist_axs[row].grid(True, axis='y', linestyle=':', alpha=0.5, linewidth=1.5, which='minor')
        hist_axs[row].set_ylabel('Offset')
        hist_axs[row].yaxis.tick_right()
        hist_axs[row].yaxis.set_label_position('right')
        hist_axs[row].set_xlabel('Prop.')
        offset_var = sp.stats.circvar(offset, low=-np.pi, high=np.pi)
        hist_axs[row].set_title(f"variance={offset_var:.2f}" )
    

    plot_row(0, 0, plt.cm.get_cmap('Greys'), 'EPGs')
    plot_row(1, 1, plt.cm.get_cmap('Greens'), 'ELs')
    
    fig.suptitle(f'{fly_id} - {sess_name}')

    return fig, (heatmap_axs, hist_axs, cbar_ax)
