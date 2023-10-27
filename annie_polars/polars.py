import numpy as np
from . import utility
from annie_gravity_model import GravityModel
from annie_gravity_model import load_datasets

class Polars:

    def __init__(self, prm):
        self.prm = prm
        self.gravity_model = GravityModel()
        self.setup_gravity_model()

    def run(self):

        print()
        data_dir = self.prm['data_dir']
        print(f'data directory: {data_dir}')
        print()

        # Get data extraction and plotting parameters
        data_prm = self.prm['data_prm']
        plot_prm = self.prm.setdefault('plot_prm', {})
        print(f'  gain corr:    {round(data_prm["gain_corr"], 4)}')

        # Load datasets and remove transient sections
        datasets_full = load_datasets(data_dir, data_prm, plot_prm)
        datasets_sect = utility.cutout_transient_sections(datasets_full, data_prm, plot_prm)

        # Get measured, gravitational and aerodynamic forces
        forces_full = {eta: self.get_forces(data) for eta, data in datasets_full.items()}
        forces_sect = {eta: self.get_forces(data) for eta, data in datasets_sect.items()}
        etas_sorted = np.array(sorted(datasets_sect.keys()))

        # Optional plots for checking analysis steps
        if plot_prm.get('trans_cutouts', False):
            for eta in etas_sorted: 
                utility.plot_trans_cut(forces_full[eta], forces_sect[eta])
        if plot_prm.get('mean_aero_forces', False):
            utility.plot_mean_aero(etas_sorted, forces_full, forces_sect)

        # Extract lift and drag forces
        fx_mean_aero, fy_mean_aero, fz_mean_aero = utility.get_mean_aero(etas_sorted, forces_sect)

        # Split into positive and negative etas (not done).
        forces_by_eta_sign = utility.get_forces_by_eta_sign(etas_sorted, fx_mean_aero, fz_mean_aero)
        if plot_prm.get('pos_neg_forces', False):
            utility.plot_pos_neg_forces(forces_by_eta_sign) 

        # Average forces for positive and negative eta
        forces = utility.average_forces_for_pos_neg_eta(forces_by_eta_sign)
        if plot_prm.get('lift_and_drag', False):
            utility.plot_lift_and_drag(forces)

        # Calculate force coefficients
        dphi_deg = data_prm['v']
        wing_prm = self.prm['wing_prm']
        fluid_prm = self.prm['fluid_prm']
        coeffs = utility.get_force_coeffs(forces, dphi_deg, wing_prm, fluid_prm)
        utility.plot_polars(coeffs)



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
        forces['fx']['aero'] = forces['fx']['meas'] - 1*forces['fx']['grav']
        forces['fy']['aero'] = forces['fy']['meas'] - 1*forces['fy']['grav']
        forces['fz']['aero'] = forces['fz']['meas'] - 1*forces['fz']['grav']
        #forces['fx']['aero'] = forces['fx']['grav']
        #forces['fy']['aero'] = forces['fy']['grav']
        #forces['fz']['aero'] = forces['fz']['grav']
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


