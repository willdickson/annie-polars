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
        datasets_mod = cutout_transient_sections(datasets, data_prm, plot_prm)

        eta_pts = np.array(sorted(datasets.keys()))
        fx_pts = np.zeros_like(eta_pts) 
        fy_pts = np.zeros_like(eta_pts)
        fz_pts = np.zeros_like(eta_pts)

        for i, eta in enumerate(eta_pts):

            forces = get_forces(datasets, eta,  self.gravity_model)
            forces_mod = get_forces(datasets_mod, eta, self.gravity_model)


            fx_pts[i] = -forces['fx']['aero'].mean()
            if eta >= 0:
                fz_pts[i] = -forces['fz']['aero'].mean()
            else:
                fz_pts[i] = forces['fz']['aero'].mean()


            #fg, ax = plt.subplots(2,1)
            #ax[0].plot(forces['phi'], forces['fz']['orig'], '.b')
            #ax[0].plot(forces['phi'], forces['fz']['grav'], '.g')
            #ax[0].plot(forces_mod['phi'], forces_mod['fz']['orig'], '.r')
            #ax[0].plot(forces_mod['phi'], forces_mod['fz']['grav'], '.m')
            #ax[0].set_ylabel('fz')
            #ax[1].grid(True)
            #ax[1].plot(forces['phi'], forces['fz']['aero'], '.b')
            #ax[1].plot(forces_mod['phi'], forces_mod['fz']['aero'], '.r')
            #ax[1].grid(True)
            #ax[1].set_ylim(0, 1.1*forces['fz']['aero'].max())
            #ax[1].set_ylabel('fz sub')
            #ax[1].set_xlabel('phi (deg)')
            #plt.show()

        fg, ax = plt.subplots(1,1)
        ax.plot(np.absolute(eta_pts), fz_pts, 'ob')
        ax.plot(np.absolute(eta_pts), fx_pts, 'or')
        ax.set_xlabel('eta')
        ax.set_ylabel('fz')
        ax.grid(True)
        plt.show()


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
        # Find locations where original indices jump, e.g. diff > 1. These indicate 
        # separate grab sections. Find boundary points on either side of each grab
        # section. 
        data = datasets[eta]
        ind = data['ind']
        mask = np.diff(ind) > 1
        pos = np.arange(ind.shape[0])
        jump_pos = [0, *list(pos[1:][mask]), pos[-1] + 1] 
        jump_bdry_pairs = [(x,y-1) for x, y in zip(jump_pos[:-1], jump_pos[1:])]

        # Add modified data to dataset_mod
        data_mod = {}
        for name, item in data.items():
            data_mod[name] = np.array([]) 
            for n0, n1 in jump_bdry_pairs:
                delta_n = n1 - n0
                c0 = n0 + int(data_prm['trans_cut'][0]*delta_n)
                c1 = n0 + int(data_prm['trans_cut'][1]*delta_n)
                data_mod[name] = np.concatenate((data_mod[name], item[c0:c1]))
        datasets_mod[eta] = data_mod
    return datasets_mod


def get_forces(datasets, eta, gravity_model): 
    phi = datasets[eta]['phi']
    grav_phi_min = gravity_model.phi.min() 
    grav_phi_max = gravity_model.phi.max()
    phi_mask = np.logical_and(phi>=grav_phi_min, phi<=grav_phi_max)
    phi = phi[phi_mask]
    forces = {}
    forces['phi'] = phi
    forces['fx'] = {}
    forces['fy'] = {}
    forces['fz'] = {}
    forces['fx']['orig'] = datasets[eta]['fx'][phi_mask]
    forces['fy']['orig'] = datasets[eta]['fy'][phi_mask]
    forces['fz']['orig'] = datasets[eta]['fz'][phi_mask]
    forces['fx']['grav'] = gravity_model.fx(eta, phi)
    forces['fy']['grav'] = gravity_model.fy(eta, phi)
    forces['fz']['grav'] = gravity_model.fz(eta, phi)
    forces['fx']['aero'] = forces['fx']['orig'] - forces['fx']['grav']
    forces['fy']['aero'] = forces['fy']['orig'] - forces['fy']['grav']
    forces['fz']['aero'] = forces['fz']['orig'] - forces['fz']['grav']
    return forces











