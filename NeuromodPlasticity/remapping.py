import numpy as np
import scipy as sp
from matplotlib import pyplot as plt





def get_opto_resp(pp):
    time = pp.voltage_recording_aligned['Time(ms)']
    opto_trig = pp.voltage_recording_aligned[' Opto Trigger']



    opto_trig_bool = np.concatenate([[0],np.diff((1*(opto_trig>1000)))>0])>0
    opto_inds = np.argwhere(opto_trig_bool).flatten()


    volume_dt = np.diff(np.array(pp.metadata['frame_times']),axis=0)[0,0]    
    
    n_frames = int(pp.metadata['mark_points']['duration']/1000/volume_dt)
    opto_off_inds = opto_inds + n_frames - 1

    for ind in opto_inds:
        pp.timeseries['rois'][:,:,ind:ind+n_frames] = np.nan
    pp.timeseries['rois'][:,:,:int(2/volume_dt)]=np.nan
    dff = pp.calculate_zscored_F('rois', exp_detrend=False, zscore=True, background_ts='background')[-1,:,:]
    
    
    
    n_rois, n_points, n_reps, n_frames = dff.shape[0], 8, 5, 2
    opto_resp = np.zeros((n_rois, n_points, n_reps))

    assert n_points*n_reps == len(opto_off_inds), "wrong number of opto inds"
    mp = 0
    for rep in range(n_reps):
        for point in range(n_points):
            ind = opto_off_inds[mp]
           
            opto_resp[:,point,rep] = np.nanmean(dff[:,ind:ind+3],axis=-1)
            mp+=1
            
    return opto_resp


def plot_min_dist(df, h_ax, x=np.arange(3), color='black'):
    for _, row in df.iterrows():
        b = row['baseline mean offset']
        p0 = row['post_0deg mean offset']
       
        
        _p180 = row['post_180deg mean offset']
        p180 = np.array([_p180, _p180+2*np.pi, _p180-2*np.pi])
        p180 = p180[np.argmin(np.abs(b-p180))]
        
        h_ax.scatter(x, [b,p0,p180], color=color)
        # h_ax.scatter(x, [b,p0,p180], color=plt.cm.hsv((b+np.pi)/(2*np.pi)), alpha=1)
        h_ax.plot(x, [b,p0,p180], color=color, alpha=.2)

def plot_sess_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times = np.arange(0, 180, 60)):
    fig, ax = plt.subplots(3, 2, figsize=[15,6], sharey=True, sharex=True)
    
    
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
        
        
        
    
    plot_row('baseline', 0, 'Greys')
    plot_row('post_0deg', 1, 'PuRd')
    plot_row('post_180deg', 2, 'GnBu')
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
        
        
    plot_hist('baseline', 'black')
    plot_hist('post_0deg', plt.cm.PuRd(.8))
    plot_hist('post_180deg', plt.cm.GnBu(.8))
    
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
    ax_polar.set_yticks([0,.2, .4],['','', ''])
    ax_polar.set_title(ts_dict['fly'])
    ax_polar.legend()
    fig_polar.tight_layout()
    
    return (fig_hist, ax_hist), (fig_polar, ax_polar)
