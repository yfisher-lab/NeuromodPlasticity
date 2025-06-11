import numpy as np
import scipy as sp
from sklearn.decomposition import PCA

from . import params

import neuprint as npt

import navis
import navis.interfaces.neuprint as neu

def npt_client(webpage='neuprint.janelia.org', dataset='hemibrain:v1.2.1', token=params.NPT_TOKEN):
    """
    Create a neuprint client object with the specified parameters.
    
    Args:
        webpage (str): The URL of the neuprint server.
        dataset (str): The name of the dataset to use.
        token (str): The authentication token for the neuprint server.
        
    Returns:
        npt.Client: A neuprint client object.
    """
    return npt.Client(webpage, dataset=dataset, token=token)



class EBCoordinateSystem():

    def __init__(self):

        self.orig_eb_mesh = neu.fetch_roi("EB")
        self.orig_origin = self.orig_eb_mesh.vertices.mean(axis=0)
        self.eb_basis = PCA(n_components=3).fit(self.orig_eb_mesh.vertices).components_


        self.eb_vertices = None
        self.eb_vertices_circ = {'phase':None, 'radius':None, 'radius_scaled':None}
        self._get_eb_mesh_coordinates()

        self._radius_min_lookup = None
        self._radius_max_lookup = None
        self._build_eb_rad_scaler()
        self._scaled_eb_radius()

        self.eb_vertices_xsec = {'radius':None, 'phase':None, 'scaled_radius': None}
        self._xsec_radius_lookup = None
        self._xsec_z_lookup = None
        self._build_xsec_lookup()
        self._get_eb_xsec_coords()

        self._xsec_rad_spline = None
        self._fit_xsec_rad_spline()
        self._scale_eb_xsec_rad()

        
    
    def change_of_basis(self, x):
        """
        x: N,3 array of coordinates in Neuprint coordinate system (x,y,z)
        """
        return (x-self.orig_origin) @ self.eb_basis.T
    
    @staticmethod
    def cart3d_to_polar2d(x):
        "assume data is centered at origin"
        phase = np.arctan2(x[:,1], x[:,0])
        radius = np.sqrt(x[:,0]**2 + x[:,1]**2)
        return phase, radius
    
    def _get_eb_mesh_coordinates(self):
        self.eb_vertices = self.change_of_basis(self.orig_eb_mesh.vertices)
        phase, radius = self.cart3d_to_polar2d(self.eb_vertices)
        self.eb_vertices_circ['phase'], self.eb_vertices_circ['radius'] = phase, radius

    def _build_eb_rad_scaler(self):
        phase, radius = self.eb_vertices_circ['phase'], self.eb_vertices_circ['radius']
        
        bins = np.linspace(-np.pi, np.pi, num=int(360/2) + 1)
        phase_bin_inds = np.digitize(phase, bins) 
        radius_min = [np.amin(radius[phase_bin_inds == i]) for i in range(1, len(bins))]
        radius_max = [np.amax(radius[phase_bin_inds == i]) for i in range(1, len(bins))]

        x = np.concatenate([bins[:-1]-2*np.pi, bins[:-1], bins[:-1]+2*np.pi])
        self._radius_min_lookup = sp.interpolate.interp1d(x, np.concatenate([radius_min for i in range(3)]), kind='cubic')
        self._radius_max_lookup = sp.interpolate.interp1d(x, np.concatenate([radius_max for i in range(3)]), kind='cubic')

        

    def scale_radius(self, phase, radius):
        return ((radius - self._radius_min_lookup(phase))/(self._radius_max_lookup(phase)-self._radius_min_lookup(phase)) + .2)/1.2
    

    def _scaled_eb_radius(self):
        """
        """
        phase, radius = self.eb_vertices_circ['phase'], self.eb_vertices_circ['radius']
        self.eb_vertices_circ['radius_scaled'] = self.scale_radius(phase, radius)
        

    def get_circ_coordinates(self, eb_x):
        """
        eb_x: (N,3) coordinates after applying change of basis
        """
        
        return self.cart3d_to_polar2d(eb_x)
    
    def _build_xsec_lookup(self):
        radius, phase, height = self.eb_vertices_circ['radius'], self.eb_vertices_circ['phase'], self.eb_vertices[:,2]

        bins = np.linspace(-np.pi, np.pi, num=int(360/2) + 1)
        phase_bin_inds = np.digitize(phase, bins) 

        r_com, h_com = [], []
        for i in range(1, len(bins)):
            rads = radius[phase_bin_inds==i]
            h = height[phase_bin_inds==i]
            
            r_com.append(rads.mean())
            h_com.append(h.mean())

        x = np.concatenate([bins[:-1]-2*np.pi, bins[:-1], bins[:-1]+2*np.pi])
        self._xsec_radius_lookup = sp.interpolate.interp1d(x, np.concatenate([r_com for i in range(3)]), kind='cubic')
        self._xsec_z_lookup = sp.interpolate.interp1d(x, np.concatenate([h_com for i in range(3)]), kind='cubic')

    def get_xsec_coords(self, eb_phase, radius, z):
        """
        eb_phase: phase along main circular axis of ring
        radius: radius along main circular axis
        z: height in eb cartesian basis
        """

        x = radius - self._xsec_radius_lookup(eb_phase)
        y = z - self._xsec_z_lookup(eb_phase)
        xy = np.zeros([x.shape[0],2])
        xy[:,0], xy[:,1] = x, y

        return self.cart3d_to_polar2d(xy)
    
    def _get_eb_xsec_coords(self):
        self.eb_vertices_xsec['phase'], self.eb_vertices_xsec['radius'] = self.get_xsec_coords(
                                                                                self.eb_vertices_circ['phase'], 
                                                                                self.eb_vertices_circ['radius'], 
                                                                                self.eb_vertices[:,2]
                                                                                )
        
    def _fit_xsec_rad_spline(self):
        knots = np.linspace(-np.pi,np.pi, num = 20)
        eb_phase, xsec_phase, xsec_rad = self.eb_vertices_circ['phase'], self.eb_vertices_xsec['phase'], self.eb_vertices_xsec['radius']
        self._xsec_rad_spline = sp.interpolate.LSQBivariateSpline(eb_phase, xsec_phase, xsec_rad, knots, knots)

    def scale_xsec_rad(self, eb_phase, xsec_phase, xsec_rad):

        return xsec_rad/self._xsec_rad_spline.ev(eb_phase, xsec_phase)

    def _scale_eb_xsec_rad(self):
        self.eb_vertices_xsec['radius_scaled'] = self.scale_xsec_rad(
                                                        self.eb_vertices_circ['phase'], 
                                                        self.eb_vertices_xsec['phase'],
                                                        self.eb_vertices_xsec['radius']
                                                        )


