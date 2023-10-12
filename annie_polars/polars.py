import pathlib
import numpy as np
import scipy as sp
import scipy.io as io
import scipy.signal as sig
import scipy.interpolate as interp
import matplotlib.pyplot as plt

from annie_gravity_model import GravityModel
from annie_gravity_model import load_datasets

class Polars:

    def __init__(self, prm):
        self.prm = prm
        self.gravity_model = GravityModel()
        self.setup_gravity_model()

    def plot(self):
        data_dir = self.prm['data_dir']
        print(data_dir)
        data_prm = self.prm['data_prm']
        plot_prm = self.prm.setdefault('plot_prm', {})

        datasets = load_datasets(data_dir, data_prm, plot_prm)
        datasets = cutout_transient_sections(datasets, data_prm, plot_prm)

        #eta_pts = np.array(sorted(datasets.keys()))
        #fx_pts = np.zeros_like(eta_pts) 
        #fy_pts = np.zeros_like(eta_pts)
        #fz_pts = np.zeros_like(eta_pts)

        #for i, eta in enumerate(eta_pts):

        #    phi = datasets[eta]['phi']

        #    grav_phi_min = self.gravity_model.phi.min() 
        #    grav_phi_max = self.gravity_model.phi.max()
        #    phi_mask = np.logical_and(phi>=grav_phi_min, phi<=grav_phi_max)
        #    phi = phi[phi_mask]

        #    fx_orig = datasets[eta]['fx'][phi_mask]
        #    fy_orig = datasets[eta]['fy'][phi_mask]
        #    fz_orig = datasets[eta]['fz'][phi_mask]

        #    fx_grav = self.gravity_model.fx(eta, phi)
        #    fy_grav = self.gravity_model.fy(eta, phi)
        #    fz_grav = self.gravity_model.fz(eta, phi)

        #    fx_aero = fx_orig - fx_grav
        #    fy_aero = fy_orig - fy_grav
        #    fz_aero = fz_orig - fz_grav

        #    fx_pts[i] = -fx_aero.mean()
        #    fy_pts[i] = fy_aero.mean()
        #    #fx_pts[i] = -np.median(fx_aero)
        #    #fy_pts[i] =  np.median(fy_aero)
        #    #fz_pts[i] = fz_aero.mean()
        #    if eta >= 0:
        #        fz_pts[i] = -fz_aero.mean()
        #        #fz_pts[i] = -np.median(fz_aero)
        #    else:
        #        fz_pts[i] = fz_aero.mean()
        #        #fz_pts[i] = np.median(fz_aero)


        #    #fg, ax = plt.subplots(2,1)
        #    #ax[0].plot(phi, fz_orig, '.b')
        #    #ax[0].plot(phi, fz_grav, '.g')
        #    #ax[0].set_ylabel('fz')
        #    #ax[1].grid(True)
        #    #ax[1].plot(phi, fz_aero, '.b')
        #    #ax[1].grid(True)
        #    #ax[1].set_ylabel('fz sub')
        #    #ax[1].set_xlabel('phi (deg)')
        #    #plt.show()

        #fg, ax = plt.subplots(1,1)
        #ax.plot(eta_pts, fz_pts, 'ob')
        #ax.plot(eta_pts, fx_pts, 'or')
        #ax.set_xlabel('eta')
        #ax.set_ylabel('fz')
        #ax.grid(True)
        #plt.show()


    def setup_gravity_model(self):
        """
        Setup model for gravity subtraction.
        """
        if 'gravity_file' in self.prm:
            self.gravity_model.load(self.prm['gravity_file'])
        elif 'gravity_dir' in self.prm:
            fit_prm = self.prm.get_default('fit_prm', None)
            self.gravity_model.fit(self.prm['gravity_data'], fit_prm=fit_prm)
        else:
            raise ValueError('gravity model missing')


# Utility
# ---------------------------------------------------------------------------------------

def cutout_transient_sections(datasets, data_prm, plot_prm):
    datasets_mod = {}
    eta_vals = np.array(sorted(datasets.keys()))

    for eta in eta_vals:

        data = datasets[eta]
        ind = data['ind']

        mask = np.diff(ind) > 1
        bdry = np.concatenate(([0], ind[1:][mask], [ind[-1]]))
        bdry_pairs = zip(bdry[:-1], bdry[1:])



        phi = datasets[eta]['phi']

        fg, ax = plt.subplots(1,1)
        ax.plot(ind, phi, '.')
        ax.plot(ind[1:][mask], phi[1:][mask], '.r')
        ax.set_xlabel('ind')
        ax.set_ylabel('phi')
        ax.grid(True)
        plt.show()






