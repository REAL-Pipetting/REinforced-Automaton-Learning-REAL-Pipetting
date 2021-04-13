import physics_models
import pandas as pd
import math


def test_beers_law():
    test_coeffs = pd.read_csv('test_coeff.csv')
    test_solution_1 = {'A': .1, 'B': .4, 'C': 1.0}  # 'Name'->'Molarity'
    test_path_length = 1  # cm
    res = physics_models.beers_law(test_solution_1,
                                   test_path_length, test_coeffs)
    expected = 1.4
    assert math.isclose(res, expected), f'Expected {expected}, got {res} instead'

    test_coeffs = pd.read_csv('test_coeff.csv')
    test_solution_2 = {'A': 10, 'B': 0, 'C': .01}
    test_path_length = .25
    res = physics_models.beers_law(test_solution_2,
                                   test_path_length, test_coeffs)
    expected = 2.50125
    assert math.isclose(res, expected), f'Expected {expected}, got {res} instead'