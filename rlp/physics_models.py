def beers_law(solution, path_length, coeffs):
    """
    Applys Beers-Lampert Law to get the total absorbtion
    of a solution.
    Inputs:
    solution: Dictionary of the form Name->Concentration
    path_length: float
    coeffs: pd dataframe with two columns 'Name', 'Coefficients'

    Returns the overall absorbtion as a float
    """
    absorbtion = 0
    coeffs_with_index = coeffs.set_index('Name')
    for key in solution:
        absorbtion += solution[key] * coeffs_with_index.loc[key, 'Coefficient']
    return absorbtion*path_length