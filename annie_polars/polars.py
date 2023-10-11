import pathlib
import numpy as np
import scipy as sp
import scipy.io as io
import scipy.signal as sig
import scipy.interpolate as interp
import matplotlib.pyplot as plt

from annie_gravity_model import GravityModel
from annie_gravity_model import get_list_of_data_files
from annie_gravity_model import sort_data_files_by_alpha
from annie_gravity_model import plot_filtered_forces
from annie_gravity_model import plot_grab_sections

class Polars:

    def __init__(self, param):
        self.param = param
        self.gravity_model = GravityModel()
        self.setup_gravity_model()

    def plot(self):
        self.load_datasets()

    def setup_gravity_model(self):
        """
        Setup model for gravity subtraction.
        """
        if 'gravity_file' in self.param:
            self.gravity_model.load(self.param['gravity_file'])
        elif 'gravity_dir' in self.param:
            fit_param = self.param.get_default('fit_param', None)
            self.gravity_model.fit(self.param['gravity_data'], fit_param=fit_param)
        else:
            raise ValueError('gravity model missing')

    def load_datasets(self):
        data_files = self.get_list_of_data_files()
        alphas, data_files = sort_data_files_by_alpha(data_files)
        datasets = {}
        print('loading datasets')
        for alpha, file in zip(alphas, data_files):
            print(f'  {file}')
            data = io.loadmat(str(file))

            # Extract the data we need for polars
            t = data['t_FT_s'][:,0]
            eta = data['wingkin_s'][:,2]
            phi = data['wingkin_s'][:,3]
            dphi = data['wingkin_s'][:,9]
            fx = data['FT_conv_s'][0,:]
            fy = data['FT_conv_s'][2,:]
            fz = data['FT_conv_s'][1,:]

            # Cut out sections between t_lim[0] and t_lim[1]
            if self.param['t_lim'] is not None:
                mask_t_lim = np.logical_and(t >= self.param['t_lim'][0], t <= self.param['t_lim'][1])
                t = t[mask_t_lim]
                eta = eta[mask_t_lim]
                phi = phi[mask_t_lim]
                dphi = dphi[mask_t_lim]
                fx = fx[mask_t_lim]
                fy = fy[mask_t_lim]
                fz = fz[mask_t_lim]

            # Lowpass filter force data
            dt = t[1] - t[0]
            force_filt = sig.butter(4, self.param['fcut'], btype='low', output='ba', fs=1/dt)
            fx_filt = sig.filtfilt(*force_filt, fx)
            fy_filt = sig.filtfilt(*force_filt, fy)
            fz_filt = sig.filtfilt(*force_filt, fz)

            # Optional plot showing filtered and unfilterd force data
            display = self.param.setdefault('display', {})
            if display.setdefault('filtered_forces', False):
                plot_filtered_forces(t, fx, fy, fz, fx_filt, fy_filt, fz_filt, alpha)

            # Get sections where dphi is equal to maximum
            mask_pos = dphi >= dphi.max() 
            t_pos = t[mask_pos]
            eta_pos = eta[mask_pos]
            phi_pos = phi[mask_pos]
            dphi_pos = dphi[mask_pos]
            fx_filt_pos = fx_filt[mask_pos]
            fy_filt_pos = fy_filt[mask_pos]
            fz_filt_pos = fz_filt[mask_pos]

            # Get sections where dphi is equal to -maximum
            mask_neg = dphi <= dphi.min() 
            t_neg = t[mask_neg]
            eta_neg = eta[mask_neg]
            phi_neg = phi[mask_neg]
            dphi_neg = dphi[mask_neg]
            fx_filt_neg = fx_filt[mask_neg]
            fy_filt_neg = fy_filt[mask_neg]
            fz_filt_neg = fz_filt[mask_neg]

            # Save datasets as function of eta
            eta_pos_val = eta_pos.max()
            eta_neg_val = eta_neg.min()
            
            datasets[eta_pos_val] = { 
                    't'    : t_pos, 
                    'eta'  : eta_pos,
                    'phi'  : phi_pos, 
                    'dphi' : dphi_pos,
                    'fx'   : fx_filt_pos,
                    'fy'   : fy_filt_pos,
                    'fz'   : fz_filt_pos, 
                    }

            datasets[eta_neg_val] = { 
                    't'    : t_neg, 
                    'eta'  : eta_neg,
                    'phi'  : phi_neg, 
                    'dphi' : dphi_neg,
                    'fx'   : fx_filt_neg,
                    'fy'   : fy_filt_neg,
                    'fz'   : fz_filt_neg, 
                    }

            # Optional plot showing grab sections for gravitational model
            if display.setdefault('grab_sections', False):
                datafull = {
                        't'   : t, 
                        'eta' : eta, 
                        'phi' : phi, 
                        'dphi': dphi, 
                        'fx'  : fx_filt, 
                        'fy'  : fy_filt, 
                        'fz'  : fz_filt
                        }
                plot_grab_sections(datafull, datasets[eta_pos_val], datasets[eta_neg_val], alpha)




    def get_list_of_data_files(self):
        """ Get list of relvant data files - filter by v and xi. """
        data_dir = pathlib.Path(self.param['data_dir'])
        v = self.param['data_v']
        xi = self.param['data_xi']
        data_files = get_list_of_data_files(data_dir,v=v, xi=xi) 
        return data_files





# Utility functions
# ---------------------------------------------------------------------------------------
