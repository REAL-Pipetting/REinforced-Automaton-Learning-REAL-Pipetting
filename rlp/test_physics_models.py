import physics_models
import pandas as pd
import math


def test_beers_law():
    test_coeffs = pd.read_csv('test_coeff.csv')
    test_solution_1 = {'A': .1, 'B': .4, 'C': 1.0}  # 'Name'->'Molarity'
    test_path_length = 1  # cm
    test_max_wl = 800
    test_min_wl = 400
    wl, spec = physics_models.beers_law(test_solution_1,
                                   test_path_length, test_coeffs,
                                   test_min_wl, test_max_wl)
    
    
    expected_spec = [1.4, 1.65, 1.7]
    expected_wl = [400, 600, 800]
    
    for i in range(expected):
        assert math.isclose(wl[i], expected_wl[i]), f'Expected wavelength {expected_wl[i]}, got {wl[i]} instead'
        assert math.isclose(spec[i], expected_spec[i]), f'Expected wavelength {expected_spec[i]}, got {spec[i]} instead'
        
    assert len(expected_spec) is len(spec), f'Expected results length of {len(expected_spec)}, got {len(spec)} instead'
