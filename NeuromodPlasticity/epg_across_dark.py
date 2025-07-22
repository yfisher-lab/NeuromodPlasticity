import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec as GS
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


def plot_transpose_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times = np.arange(0, 360, 60), twindow = None):
    fig, ax = plt.subplots(2, 3, figsize=[6,15], sharey=True, sharex=True)
    
    
    def get_time_ticks_inds(time, plot_times):
        inds = []
        for t in plot_times:
            inds.append(np.argmin(np.abs(time-t)))
        return inds
        
    
    def plot_col(key, col, cmap):
        dff = ts_dict[key].dff
        time = ts_dict[key].time

        if twindow is not None:
            mask = (time>=twindow[0]) * (time<=twindow[1])
        else:
            mask = np.ones_like(time)>0
        dff= dff[:,mask]
        time = time[mask]


        x = np.arange(dff.shape[1])
        heading_ = (ts_dict[key].heading[mask]+np.pi)/(2*np.pi)*15
        h = ax[0,col].imshow(dff.T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(h, ax=ax[0, col])
        
        ax[0,col].scatter(heading_, x, color='orange', s=5)
        
        h = ax[1, col].imshow(ts_dict[key].dff_h_aligned[:,mask].T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(h, ax=ax[1, col])
        ax[1, col].scatter(7.5*np.ones_like(heading_), x, color='orange', s=5)
        
        ax[0, col].set_xlabel('ROIs')
        ax[0, col].set_xticks([-0.5,7.5,15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        
        _plot_times = plot_times[plot_times<time.iloc[-1]]
        ax[0, col].set_yticks(get_time_ticks_inds(time, _plot_times), labels=_plot_times)
        ax[0, col].set_ylabel('Time (s)')
        
        ax[0, col].set_title(key)
        
        
        
    
    plot_col('closed_loop 1', 0, 'Greys')
    plot_col('dark', 1, 'Purples')
    plot_col('closed_loop 2', 2, 'Greys')
    fig.suptitle(ts_dict['fly'])
    
    fig.tight_layout()
    
    return fig, ax




def plot_sess_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times = np.arange(0, 360, 60), twindow = None):
    fig, ax = plt.subplots(3, 2, figsize=[15,6], sharey=True, sharex=True)
    
    
    def get_time_ticks_inds(time, plot_times):
        inds = []
        for t in plot_times:
            inds.append(np.argmin(np.abs(time-t)))
        return inds
        
    
    def plot_row(key, row, cmap):
        dff = ts_dict[key].dff
        time = ts_dict[key].time

        if twindow is not None:
            mask = (time>=twindow[0]) * (time<=twindow[1])
        else:
            mask = np.ones_like(time)>0
        dff= dff[:,mask]
        time = time[mask]


        x = np.arange(dff.shape[1])
        heading_ = (ts_dict[key].heading[mask]+np.pi)/(2*np.pi)*15
        h = ax[row,0].imshow(dff, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(h, ax=ax[row,0])
        
        ax[row,0].scatter(x, heading_, color='orange', s=5)
        
        h = ax[row,1].imshow(ts_dict[key].dff_h_aligned[:,mask], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(h, ax=ax[row,1])
        ax[row,1].scatter(x, 7.5*np.ones_like(heading_), color='orange', s=5)
        
        ax[row,0].set_ylabel('ROIs')
        ax[row,0].set_yticks([-0.5,7.5,15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        
        _plot_times = plot_times[plot_times<time.iloc[-1]]
        ax[row,0].set_xticks(get_time_ticks_inds(time, _plot_times), labels=_plot_times)
        ax[row,0].set_xlabel('Time (s)')
        
        ax[row,0].set_title(key)
        
        
        
    
    plot_row('closed_loop 1', 0, 'Greys')
    plot_row('dark', 1, 'Purples')
    plot_row('closed_loop 2', 2, 'Greys')
    fig.suptitle(ts_dict['fly'])
    
    fig.tight_layout()
    
    return fig, ax

def plot_sess_histograms(ts_dict, bins = np.linspace(-np.pi, np.pi, num=17)):
    
    fig_hist, ax_hist = plt.subplots()
    fig_polar, ax_polar = plt.subplots(subplot_kw={'projection':'polar'})
    centers = (bins[1:]+bins[:-1])/2
    def plot_hist(key, color, hatch=None):
        offset = ts_dict[key].offset
        hist, _ = np.histogram(offset, bins=bins)
        hist = hist/hist.sum()
        
        offset_c_mu = ts_dict[key].offset_c.mean()
        
        if hatch is not None:
            ax_hist.fill_between(centers, hist, color='none', alpha=.4, hatch=hatch, edgecolor=color)
            ax_polar.plot(np.angle(offset_c_mu)*np.ones([2,]), [0, np.abs(offset_c_mu)], color=color, linewidth=2, label=key,
                          linestyle='--', alpha=.4)
            
        else:
            ax_hist.fill_between(centers, hist, color=color, alpha=.4)
            ax_polar.plot(np.angle(offset_c_mu)*np.ones([2,]), [0, np.abs(offset_c_mu)], color=color, linewidth=2, label=key)        
        
    plot_hist('closed_loop 1', 'black')
    plot_hist('dark', plt.cm.Purples(.8), hatch=None)
    plot_hist('closed_loop 2', 'black', hatch='/')
    
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
    ax_polar.set_title(ts_dict['fly'])
    ax_polar.legend()
    fig_polar.tight_layout()
    
    return (fig_hist, ax_hist), (fig_polar, ax_polar)


def plot_sess_heatmaps_w_hist(ts_dict, bins = np.linspace(-np.pi, np.pi, num=17),vmin=-.5, vmax=.5, 
                              plot_times = np.arange(0, 360, 60), twindow = None):
    
    fig = plt.figure(figsize=[8,6])
    gs = GS(3,4, figure=fig, width_ratios=[6,1,.8, .2],wspace=.05,hspace=.8)
    heatmap_axs = [fig.add_subplot(gs[0,0])]
    heatmap_axs.extend([fig.add_subplot(gs[i,0],sharey=heatmap_axs[0]) for i in range(1,3)])
    hist_axs = [fig.add_subplot(gs[0,1])]
    hist_axs.extend([fig.add_subplot(gs[i,1],sharex=hist_axs[0]) for i in range(1,3)])
    cbar_ax = [fig.add_subplot(gs[i,3]) for i in range(3)]
    
    def get_time_ticks_inds(time, plot_times):
        inds = []
        for t in plot_times:
            inds.append(np.argmin(np.abs(time-t)))
        return inds
    
    def plot_row(key, row, cmap, hatch=None):
        dff = ts_dict[key].dff
        heading = ts_dict[key].heading
        offset = ts_dict[key].offset
        
        time = ts_dict[key].time

        if twindow is not None:
            mask = (time>=twindow[0]) * (time<=twindow[1])
        else:
            mask = np.ones_like(time)>0
        dff= dff[:,mask]
        time = time[mask]
        heading = heading[mask]

        x = np.arange(dff.shape[1])
        heading_ = (heading + np.pi) / (2 * np.pi) * 15
        h = heatmap_axs[row].imshow(dff, aspect='auto', cmap=cmap, interpolation='none', vmin=-.5, vmax=3)
        fig.colorbar(h, cax=cbar_ax[row])
        heatmap_axs[row].scatter(x, heading_, color='orange', s=5)

        heatmap_axs[row].set_ylabel('ROIs')
        heatmap_axs[row].set_yticks([-0.5,7.5,15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        heatmap_axs[row].yaxis.set_minor_locator(AutoMinorLocator())
        
        _plot_times = plot_times[plot_times<time.iloc[-1]]
        heatmap_axs[row].set_xticks(get_time_ticks_inds(time, _plot_times), labels=_plot_times)
        heatmap_axs[row].set_xlabel('Time (s)')
        
        heatmap_axs[row].set_title(key)

        
      
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
      

        



    plot_row('closed_loop 1', 0, plt.cm.Greys)
    plot_row('dark', 1, plt.cm.Purples, hatch=None)
    plot_row('closed_loop 2', 2, plt.cm.Greys, hatch='/')
    hist_axs[0].set_xlim(left=0)
    fig.suptitle(ts_dict['fly'])
    return fig, (heatmap_axs, hist_axs, cbar_ax)  # Return the figure and axes for further customization if needed