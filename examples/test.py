import pathlib
from annie_polars import Polars

param = {
        'gravity_file': pathlib.Path('gravity_model.pkl'),
        'data_dir'    : pathlib.Path('../../v_60_90_results_with_conv/'),
        'data_v'      : 90,
        'data_xi'     : 0, 
        't_lim'       : (0.8, 7.3),
        'eta_lim'     : None,
        'fcut'        : 10.0, 
        'display'     : {
            'filtered_forces': False, 
            'grab_sections'  : True, 
            }
        }

polars = Polars(param)
polars.plot()
