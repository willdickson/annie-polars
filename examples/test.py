import pathlib
from annie_polars import Polars

prm = {
        'gravity_file'  : pathlib.Path('gravity_model.pkl'),
        'data_dir'      : pathlib.Path('../../v_60_90_results_with_conv/'),
        'data_prm'      : {
            'v'         : 90,
            'xi'        : 0, 
            'fcut'      : 10.0, 
            't_lim'     : (1.5, 7.5),
            'eta_lim'   : (-80, 80),
            'trans_cut' : (0.6, 0.9),
            }, 
        'plot_prm'  : {
            'filtered_forces': False, 
            'grab_sections'  : False, 
            'trans_cutouts'  : False,
            'mean_aero'      : False,
            }
        }

polars = Polars(prm)
polars.plot()
