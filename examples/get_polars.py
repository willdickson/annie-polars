import pathlib
from annie_polars import Polars

prm = {
        'gravity_file'  : pathlib.Path('gravity_model.pkl'),
        'data_dir'      : pathlib.Path('../../v_60_90_results_with_conv/'),
        'data_prm'      : {
            'v'         : 90,          # trial velocity (deg/sec)
            'xi'        : 0,           # wing kinematic xi value (deg)
            'fcut'      : 10.0,        # lowpass filter cutoff frequency (Hz)
            't_lim'     : (1.5, 7.5),  # lower & upper bounds for time range (sec)
            'eta_lim'   : (-80, 80),   # lower & upper bounds for eta (deg)
            'trans_cut' : (0.6, 0.9),  # transient cutout region, fraction (dimensionless)
            'gain_corr' : 2.0,         # Gain correction factor (due to analog ref issues)
            }, 
        'wing_prm'  : {
            'length'     : 0.248,  # (m)
            'mean_chord' : 0.0883, # (m)
            'nd_2nd_mom' : 0.35,   # dimensionless
            },
        'fluid_prm' : {
            'density' : 880,  # (kg/m**3)
            },
        'plot_prm'  : {
            'filtered_forces'  : False, 
            'grab_sections'    : False, 
            'trans_cutouts'    : False,
            'mean_aero_forces' : False,
            'pos_neg_forces'   : False, 
            'lift_and_drag'    : False,
            },
        }

polars = Polars(prm)
polars.run()
