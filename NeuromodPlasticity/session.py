
import numpy as np
import scipy as sp


import SessionTools.two_photon as st2p


class GetTS():
    
    def __init__(self, pp, **kwargs):
        
        self.pp = pp
        self.dff = None
        self.heading = None
        self.heading_sm = None
        self.offset = None
        self.phi = None
        self.rho = None
        self.heading_bins = None
        self.dff_h_aligned = None
        self.time = None
        self.n_rois = None
        self.dt = None

        
        self.get_ts(**kwargs)
        self.heading_aligned()
        
        
    def get_ts(self, channels=-1, exp_detrend=True, zscore=True, background_ts='background',
               circ_sigma=.5, t_sigma=.1, h_sigma=.1, neural_shift=-.2):
        
        
        
        
        self.time = self.pp.voltage_recording_aligned['Time(ms)']/1000
        self.time = self.time-self.time[0]
        
        
        self.dt = np.diff(self.time).mean()
        neural_shift_inds = int(neural_shift/self.dt)
        
        self.heading = np.angle(np.exp(1j*(-1*self.pp.voltage_recording_aligned[' Heading'].to_numpy()-np.pi)))
        x_h, y_h = st2p.utilities.pol2cart(np.ones_like(self.heading), self.heading)
        x_h, y_h = sp.ndimage.gaussian_filter1d(x_h, h_sigma/self.dt), sp.ndimage.gaussian_filter1d(y_h, h_sigma/self.dt)
        _, self.heading_sm = st2p.utilities.cart2pol(x_h, y_h)
        
        self.dff = self.pp.calculate_zscored_F('rois', exp_detrend=exp_detrend, zscore=zscore, 
                                               background_ts=background_ts)[channels,:,:]
        
        self.dff = sp.ndimage.gaussian_filter1d(sp.ndimage.gaussian_filter1d(self.dff, t_sigma/self.dt, axis=-1),
                                                circ_sigma,axis=1, mode='wrap')
        
        self.dff = np.roll(self.dff, neural_shift_inds, axis=-1)
        
        
        
        self.n_rois = self.dff.shape[0]  
        x_f,y_f = st2p.utilities.pol2cart(self.dff ,np.linspace(-np.pi,np.pi,num=self.n_rois)[:,np.newaxis])
        self.rho, self.phi = st2p.utilities.cart2pol(x_f.mean(axis=0), y_f.mean(axis=0))
        
        _,self.offset = st2p.utilities.cart2pol(*st2p.utilities.pol2cart(np.ones(self.heading.shape),self.phi-self.heading))
        
        
        
    def heading_aligned(self):
        self.heading_bins = np.linspace(-np.pi, np.pi, num=self.n_rois+1)
        heading_dig = np.digitize(self.heading, self.heading_bins)-1

        self.dff_h_aligned = np.zeros_like(self.dff)
        for ind in range(self.heading.shape[0]):
            self.dff_h_aligned[:,ind] = np.roll(self.dff[:,ind], -heading_dig[ind]+8)
    
    @property
    def offset_c(self):
        return np.exp(1j*self.offset)
    
    