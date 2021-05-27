import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt 


def spectra_to_color(wavelength, A, red, green, blue):
    
    '''This function calculates the x-y color coordinates from spectra. Use this function to generate data 
       to train the model. Inputs are 2x1 arrays.
       A = spectra to convert into x-y color coodrinates
       red = spectra of red stock 
       green = spectra of green stock
       blue = spectra of blue stock 
       wavelength = wavelength of the specta'''
    
    P = A
    def func3(x,y,z):
        MSE = np.sum((P - (x*red + y*green + z*blue))**2)
        return 1/(MSE+0.001) 

    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1
    zmin = 0
    zmax = 1

    pbounds = {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
    optimizer = BayesianOptimization(f=func3, pbounds=pbounds, verbose=3)
    optimizer.maximize(init_points = 20, n_iter = 30)
    best_params = optimizer.max["params"]
    found_x = best_params['x']
    found_y = best_params['y']
    found_z = best_params['z']
    max_value = func3(found_x, found_y, found_z)
    
    RED = found_x*red
    GREEN = found_y*green
    BLUE = found_z*blue
    
    X = []
    Y = []
    Z = []
    dx = wavelength[1] - wavelength[0]
    for i in range(len(P)):
        X.append((RED[i]*P[i]*dx))
        Y.append((GREEN[i]*P[i]*dx))
        Z.append((BLUE[i]*P[i]*dx))

    X = np.sum(X)
    Y = np.sum(Y)
    Z = np.sum(Z)

    x = X/(X+Y+Z)
    y = Y/(X+Y+Z)
    z = 1 - x - y
    return x,y,z
    

def concentration_to_color(c_red, c_green, c_blue, red, green, blue, wavelength):
    
    '''Takes in the concentrations and absorbances of the red, green, and blue dye and outputs the x-y coordinates
       of the color corresponding to this mixture
       c_red = concentration of red stock 
       c_green = concentration of green stock 
       c_blue = concentration of blue stock 
       red = spectra of red stock 
       green = spectra of green stock
       blue = spectra of blue stock 
       wavelength = wavelength of the specta'''
   
    RED = c_red*red
    GREEN = c_green*green
    BLUE = c_blue*blue 

    P = RED + GREEN + BLUE

    X = []
    Y = []
    Z = []
    dx = wavelength[1] - wavelength[0]
    for i in range(len(P)):
        X.append((RED[i]*P[i]*dx))
        Y.append((GREEN[i]*P[i]*dx))
        Z.append((BLUE[i]*P[i]*dx))

    X = np.sum(X)
    Y = np.sum(Y)
    Z = np.sum(Z)

    x = X/(X+Y+Z)
    y = Y/(X+Y+Z)
    z = 1 - x - y
    return x,y,z

def color_to_concentration(x2, y2, red, green, blue, wavelength):
    
    '''Takes in an x,y coordinate of the color and the absorbances of the stock solutions and outputs a 
        red-green-blue concentration of stock to match this color
       x2 = x-color coordinate 
       y2 = y-color coordinate 
       red = spectra of red stock 
       green = spectra of green stock
       blue = spectra of blue stock 
       wavelength = wavelength of the specta'''
    
    def func(x,y):
        RED = x*red 
        GREEN = y*green 
        BLUE = (1- x - y)*blue
        P = RED + GREEN + BLUE

        X = []
        Y = []
        Z = []
        dx = wavelength[1] - wavelength[0]

        for i in range(len(P)):
            X.append((RED[i]*P[i]*dx))
            Y.append((GREEN[i]*P[i]*dx))
            Z.append((BLUE[i]*P[i]*dx))
        X = np.sum(X)
        Y = np.sum(Y)
        Z = np.sum(Z)
        x1 = X/(X+Y+Z)
        y1 = Y/(X+Y+Z)
        z1 = 1 - x1 - y1
        error = (x1 - x2)**2 + (y1 - y2)**2

        return 1/error


    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1


    pbounds = {'x': (xmin, xmax), 'y': (ymin, ymax)}

    optimizer = BayesianOptimization(f=func, pbounds=pbounds, verbose=3)

    optimizer.maximize(init_points = 20, n_iter = 30)

    best_params = optimizer.max["params"]

    found_x = best_params['x']
    found_y = best_params['y']
    max_value = func(found_x, found_y)
    c_red = found_x
    c_green = found_y
    c_blue = 1 - found_x - found_y
    
    return c_red, c_green, c_blue 

def new_spectra_from_stock(c_red, c_green, c_blue, red, green, blue, wavelength):
    
    ''' Creates a new wavelength - Absorbance spectra with different concentrations of 
        the original Red, Green, Blue stock solutions'''
    
    b = 1
    A = red*c_red*b + green*c_green*b + blue*c_blue*b
    return A

def plot_spectra_from_color(x,y,new_red_stock_Abs, new_green_stock_Abs, new_blue_stock_Abs, wavelength):
    
    ''' Plots a specta from x-y color coordinates and the absorbance spectra of the red, green, and blue stock soltuions'''
    
    assert new_red_stock_Abs.shape == new_green_stock_Abs.shape, 'Shapes of the absorbance array''s must be equal' 
    assert new_red_stock_Abs.shape == new_blue_stock_Abs.shape, 'Shapes of the absorbance array''s must be equal' 
    assert new_red_stock_Abs.shape == wavelength.shape, 'Shapes of the absorbance and wavelength array''s must be equal' 
    
    c_red, c_green, c_blue = color_to_concentration(x,y, new_red_stock_Abs, new_green_stock_Abs, new_blue_stock_Abs, wavelength)
    A = new_red_stock_Abs*c_red + new_green_stock_Abs*c_green + new_blue_stock_Abs*c_blue
    fig, ax = plt.subplots(figsize=(15,7))
    ax.plot(wavelength, A)
    return wavelength, A 
