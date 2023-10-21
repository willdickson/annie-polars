import copy
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

        # Get data extraction and plotting parameters
        data_prm = self.prm['data_prm']
        plot_prm = self.prm.setdefault('plot_prm', {})

        # Load datasets and remove transient sections
        datasets_full = load_datasets(data_dir, data_prm, plot_prm)
        datasets_sect = cutout_transient_sections(datasets_full, data_prm, plot_prm)

        # Get measured, gravitational and aerodynamic forces
        forces_full = {eta: self.get_forces(data) for eta, data in datasets_full.items()}
        forces_sect = {eta: self.get_forces(data) for eta, data in datasets_sect.items()}
        etas_sorted = np.array(sorted(datasets_sect.keys()))

        # Optional plots for checking analysis steps
        if plot_prm.get('trans_cutouts', False):
            for eta in etas_sorted: 
                plot_trans_cut(forces_full[eta], forces_sect[eta])
        if plot_prm.get('mean_aero_forces', False):
            plot_mean_aero(etas_sorted, forces_full, forces_sect)

        # Extract lift and drag forces
        fx_mean_aero, fy_mean_aero, fz_mean_aero = get_mean_aero(etas_sorted, forces_sect)

        # Split into positive and negative etas (not done).
        forces_by_eta_sign = get_forces_by_eta_sign(etas_sorted, fx_mean_aero, fz_mean_aero)
        if plot_prm.get('pos_neg_forces', False):
            plot_pos_neg_forces(forces_by_eta_sign) 

        # Average forces for positive and negative eta
        forces = average_forces_for_pos_neg_eta(forces_by_eta_sign)
        if plot_prm.get('lift_and_drag', False):
            plot_lift_and_drag(forces)

        # Calculate force coefficients
        dphi_deg = data_prm['v']
        wing_prm = self.prm['wing_prm']
        fluid_prm = self.prm['fluid_prm']
        coeffs = get_force_coeffs(forces, dphi_deg, wing_prm, fluid_prm)
        plot_polars(coeffs)



    def get_forces(self, data): 
        """
        Extract measured, gravitational and aerodynamic forces.
        """
        phi = data['phi']
        eta = data['eta'][0]
        grav_phi_min = self.gravity_model.phi.min() 
        grav_phi_max = self.gravity_model.phi.max()
        phi_mask = np.logical_and(phi>=grav_phi_min, phi<=grav_phi_max)
        phi = phi[phi_mask]
        t = data['t'][phi_mask]
        ind = data['ind'][phi_mask]
        forces = {}
        forces['t'] = t
        forces['phi'] = phi
        forces['ind'] = ind 
        forces['fx'] = {}
        forces['fy'] = {}
        forces['fz'] = {}
        forces['fx']['meas'] = data['fx'][phi_mask]
        forces['fy']['meas'] = data['fy'][phi_mask]
        forces['fz']['meas'] = data['fz'][phi_mask]
        forces['fx']['grav'] = self.gravity_model.fx(eta, phi)
        forces['fy']['grav'] = self.gravity_model.fy(eta, phi)
        forces['fz']['grav'] = self.gravity_model.fz(eta, phi)
        forces['fx']['aero'] = forces['fx']['meas'] - forces['fx']['grav']
        forces['fy']['aero'] = forces['fy']['meas'] - forces['fy']['grav']
        forces['fz']['aero'] = forces['fz']['meas'] - forces['fz']['grav']
        forces['fx']['mean_aero'] = forces['fx']['aero'].mean()
        forces['fy']['mean_aero'] = forces['fy']['aero'].mean()
        forces['fz']['mean_aero'] = forces['fz']['aero'].mean()
        return forces


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
def fx_sign(eta):
    return -1

def fy_sign(eta):
    return 1

def fz_sign(eta):
    if eta >= 0:
        rval = -1
    else:
        rval = 1
    return rval

def force_sign(fname, eta):
    func_table = {
            'fx': fx_sign, 
            'fy': fy_sign,
            'fz': fz_sign,
            }
    return func_table[fname](eta)

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


def get_mean_aero(etas, forces): 
    """
    Extracts mean aerodynamic forces for each eta.  Adjusts forces for sign. 

    Arguments:
      etas   = eta wing angles 
      forces = dictionary containing forces as a function of eta. 
    """
    mean_aero = []
    for fname in ('fx', 'fy', 'fz'):
        mean_vals = np.array([force_sign(fname, eta)*forces[eta][fname]['mean_aero'] for eta in etas])
        mean_aero.append(mean_vals)
    return tuple(mean_aero)


def get_forces_by_eta_sign(eta, fx, fz): 
    """
    Split mean aero dynamics forces by eta sign (eta >= 0 and eta <= 0). 

    Arguments:
      eta = wing kinematics angle eta in degrees.
      fx  = x-component of mean aerodynamic force.
      fy  = y-component of mean aerodynamic force.

    Returns:
      forces_by_eta_sign = dict keyed by 'pos', 'neg' giving aerodynamic
      forces for case where eta >=0 and eta <= 0.

    """
    forces_by_eta_sign = { }
    eta_sign_to_mask_func = {'pos': np.greater_equal, 'neg': np.less_equal }
    for sign_str, mask_func in eta_sign_to_mask_func.items():
        mask = mask_func(eta, 0)
        eta_mask_deg = eta[mask]
        aoa_mask_deg = 90.0 - np.absolute(eta_mask_deg)
        ind_sort = np.argsort(aoa_mask_deg)
        eta_sort_deg = eta_mask_deg[ind_sort]
        aoa_sort_deg = aoa_mask_deg[ind_sort]
        eta_sort_rad = np.deg2rad(eta_sort_deg)
        aoa_sort_rad = np.deg2rad(aoa_sort_deg)
        fx_sort = fx[mask][ind_sort]
        fz_sort = fz[mask][ind_sort]
        lift_sort, drag_sort = get_lift_and_drag(aoa_sort_rad, fx_sort, fz_sort)
        forces_by_eta_sign[sign_str] = {
                'eta': {
                    'deg' : eta_sort_deg,
                    'rad' : eta_sort_rad,
                    },
                'aoa': {
                    'deg': aoa_sort_deg, 
                    'rad': aoa_sort_rad,
                    },
                'fx'     : fx_sort,
                'fz'     : fz_sort, 
                'lift'   : lift_sort, 
                'drag'   : drag_sort,
                }
    return forces_by_eta_sign


def average_forces_for_pos_neg_eta(forces_by_eta_sign):
    """
    Average forced for eta>=0 and eta<=0 kinematics.  

    Arguments:
      forces_by_eta_sing = dict keyed by 'pos' and 'neg' giving aerodynamic forces
      for cases where eta>=0 and eta<=0.

    Returns:
      forces = dictionary of aerodynamics forces.

    """
    forces = copy.deepcopy(forces_by_eta_sign['pos'])
    for k in forces:
        if k in ('eta', 'aoa'):
            continue
        forces[k] = 0.5*(forces[k] + forces_by_eta_sign['neg'][k])
    return forces



def get_lift_and_drag(aoa, fx, fz): 
    """
    Calculates lift and drag given the angle-of-attack (in radians) and the  x
    and z components of the forces.  

    Arguments:
      aoa = angle of attach in radians
      fx  = x-component of aerodynamic force 
      fz  = z-component of aerodynamic force

    Returns:
      lift = lift acting on wing
      drag = drag acting on wing
    """

    lift = fz*np.cos(aoa) - fx*np.sin(aoa)
    drag = fz*np.sin(aoa) + fx*np.cos(aoa)
    return lift, drag


def get_force_coeffs(forces, dphi_deg, wing_prm, fluid_prm):
    dphi_rad = np.deg2rad(dphi_deg)
    wing_length = wing_prm['length']
    mean_chord = wing_prm['mean_chord']
    nd_2nd_mom = wing_prm['nd_2nd_mom']
    density = fluid_prm['density']
    tip_velocity = (dphi_rad*wing_length)**2
    wing_area = mean_chord*wing_length
    const = 2.0/(density*(tip_velocity**2)*wing_area*nd_2nd_mom)
    coeffs = {}
    coeffs['eta'] = copy.deepcopy(forces['eta'])
    coeffs['aoa'] = copy.deepcopy(forces['aoa'])
    coeffs['lift'] = const*forces['lift']
    coeffs['drag'] = const*forces['drag']
    return coeffs


def plot_trans_cut(data_full, data_sect, abscissa='ind'): 
    """
    Plot transient cutouts as function of 't','phi' or 'ind'.
    """
    if abscissa not in ('t', 'phi', 'ind'):
        assert RuntimeError(f'unknown abscissa type {abscissa}')
    abscissa_labels = {
            't'   : 't (sec)', 
            'phi' : 'phi (deg)',
            'ind' : 'ind',
            }
    fg, ax = plt.subplots(3,3, sharex=True)
    force_names = ('fx', 'fy', 'fz')
    force_types = ('meas', 'grav', 'aero')
    fmax = -np.inf
    fmin =  np.inf
    for i, fname in enumerate(force_names):
        for j, ftype in enumerate(force_types):
            abscissa_full = data_full[abscissa]
            abscissa_sect = data_sect[abscissa]
            force_full = data_full[fname][ftype]
            force_sect = data_sect[fname][ftype]
            ax[i,j].plot(abscissa_full, force_full, '.b')
            ax[i,j].plot(abscissa_sect, force_sect, '.r')
            ax[i,j].grid(True)
            fmax = max(fmax, force_full.max(), force_sect.max())
            fmin = min(fmin, force_full.min(), force_sect.min())
            if i == 0:
                ax[i,j].set_title(ftype)
            if i+1 == len(force_names):
                ax[i,j].set_xlabel(abscissa_labels[abscissa])
            if j == 0:
                ax[i,j].set_ylabel(fname)
    for i, _ in enumerate(force_names):
        for j, _ in enumerate(force_types):
            frng = fmax - fmin
            ax[i,j].set_ylim(fmin - 0.1*frng, fmax + 0.1*frng)
    plt.show()


def plot_mean_aero(etas_sorted, forces_full, forces_sect): 
    """
    Plots the mean aerodynamic forcs for fx and fz as a function of the sorted
    eta values for both the full and transient cutout grab sections. 
    """
    fg, ax = plt.subplots(2,1)
    legend_info = {'line'  : {0: [], 1: []}, 'label' : {0: [], 1: []}}
    for forces, name, style in [(forces_full, 'full', 'ob'), (forces_sect, 'sect', 'or')]: 
        fx, fy, fz = get_mean_aero(etas_sorted, forces)
        fz_line, = ax[0].plot(etas_sorted, fz, style)
        legend_info['line'][0].append(fz_line)
        legend_info['label'][0].append(f'fz {name}')
        ax[0].set_ylabel('fz')
        ax[0].grid(True)
        fx_line, = ax[1].plot(etas_sorted, fx, style)
        legend_info['line'][1].append(fz_line)
        legend_info['label'][1].append(f'fz {name}')
        ax[1].set_ylabel('fx')
        ax[1].set_xlabel('eta')
        ax[1].grid(True)
    ax[0].legend(legend_info['line'][0],legend_info['label'][0], loc='upper right')
    ax[1].legend(legend_info['line'][1],legend_info['label'][1], loc='upper right')
    plt.show()


def plot_pos_neg_forces(forces_by_eta_sign): 
    """
    Plot lift and drag forces for positive and negative eta separately. 
    """
    fg, ax = plt.subplots(2,1)
    sign_to_style = {'pos': 'or', 'neg': 'ob'}
    for eta_sign, force_data in forces_by_eta_sign.items():
        aoa = force_data['aoa']['deg']
        lift = force_data['lift']
        drag = force_data['drag']
        style = sign_to_style[eta_sign]

        lift_line, = ax[0].plot(aoa, lift, style)
        ax[0].grid(True)
        ax[1].set_ylabel('lift')

        drag_line, = ax[1].plot(aoa, drag, style)
        ax[1].grid(True)
        ax[1].set_ylabel('drag')
    ax[1].set_xlabel('aoa (deg)')
    plt.show()

def plot_lift_and_drag(forces):
    fg, ax = plt.subplots(1,1)
    lift_line, = ax.plot(forces['aoa']['deg'], forces['lift'], 'ob')
    drag_line, = ax.plot(forces['aoa']['deg'], forces['drag'], 'or')
    ax.grid(True)
    ax.set_xlabel('aoa (deg)')
    ax.set_ylabel('forces')
    ax.legend((lift_line, drag_line), ('lift', 'drag'), loc='upper right')
    plt.show()


def plot_polars(coeffs):
    fg, ax = plt.subplots(1,1)
    lift_line, = ax.plot(coeffs['aoa']['deg'], coeffs['lift'], 'ob')
    drag_line, = ax.plot(coeffs['aoa']['deg'], coeffs['drag'], 'or')
    ax.grid(True)
    ax.set_xlabel('aoa (deg)')
    ax.set_ylabel('coeff')
    ax.legend((lift_line, drag_line), ('lift', 'drag'), loc='upper right')
    plt.show()











