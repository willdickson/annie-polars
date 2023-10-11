import pathlib
from annie_polars import Polars

param = {
        'gravity_file': pathlib.Path('gravity_model.pkl'),
        'data_dir'    : pathlib.Path('../../v_60_90_results_with_conv/'),
        'data_v'      : 90,
        'data_xi'     : 0, 
        't_lim'       : None,
        'eta_lim'     : None,
        'fcut'        : 10.0, 
        'display'     : {
            'filtered_forces': True, 
            }
        }

polars = Polars(param)
polars.plot()
