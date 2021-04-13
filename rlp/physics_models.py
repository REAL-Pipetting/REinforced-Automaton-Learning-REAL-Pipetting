def beers_law(solution, path_length, coeffs, min_wavelength, max_wavelength):
    """
    Applys Beers-Lampert Law to get the total absorbtion
    of a solution.
    Inputs:
    solution: Dictionary of the form Name->Concentration
    path_length: float
    coeffs: pd dataframe with column 'Wavelength' and coefficients in other column.
    Each other column corresponds to the coffiecient at that wavelength for the 
    title of the column which should be the same as the solution
    
    'Wavelength', 'Dye1', 'Dye2'
    600, .1, 0
    601, .2, .1

    Returns the overall absorbtion as a float
    """
    wavelengths = []
    spectra = []
    absorbtion = 0
    coeffs_with_index = coeffs.set_index('Wavelength')
    coeffs_with_index.sort_index(inplace=True)
    coeffs_with_index.dropna(inplace=True)
    
    for index, row in coeffs_with_index.iterrows():
        if index >= min_wavelength and index <= max_wavelength:
            for key in solution:
                absorbtion += solution[key] * coeffs_with_index.loc[index, key]
            wavelengths.append(index)
            spectra.append(absorbtion)
            absorbtion = 0
        
    
    return wavelengths, spectra*path_length