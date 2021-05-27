import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from bayes_opt import BayesianOptimization
from color_functions import color_to_concentration, concentration_to_color, new_spectra_from_stock, plot_spectra_from_color, spectra_to_color
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def extract_data():
    Sample_concentrations = pd.read_excel('Sample_concentrations.xlsx')
    red_conc = Sample_concentrations[['Red conc']].values
    green_conc = Sample_concentrations[['Green conc']].values
    blue_conc = Sample_concentrations[['Blue conc']].values
    sample_conc = np.hstack((red_conc, green_conc, blue_conc))

    sample_spectra = pd.read_excel('Sample_spectra.xlsx')
    sample_spectra = np.asarray(sample_spectra)
    wavelength = sample_spectra[:,0]
    red = sample_spectra[:,1]
    green = sample_spectra[:,2]
    blue = sample_spectra[:,3]

    x_color_list = []
    y_color_list = []

    for i in range(1, sample_spectra.shape[1]):
        x_color, y_color, _ = spectra_to_color(wavelength, sample_spectra[:,i], red, green, blue)
        x_color_list.append(x_color)
        y_color_list.append(y_color)
    actual_color = np.asarray([x_color_list, y_color_list]).T

    x = sample_conc
    y = actual_color
   
    return x,y 
    
    
    
def predict_spectra(x_color_objective, y_color_objective, x,y):
    sample_spectra = pd.read_excel('Sample_spectra.xlsx')
    sample_spectra = np.asarray(sample_spectra)
    wavelength = sample_spectra[:,0]
    red = sample_spectra[:,1]
    green = sample_spectra[:,2]
    blue = sample_spectra[:,3]
    c_red, c_green, c_blue = linear_regression(x,y, x_color_objective, y_color_objective)
    x_color, y_color, _ = concentration_to_color(c_red, c_green, c_blue, red, green, blue, wavelength)
    plot_spectra_from_color(x_color,y_color, red, green, blue, wavelength)

    c_red, c_green, c_blue = decision_tree_regressor(x,y, x_color_objective, y_color_objective)
    x_color, y_color, _ = concentration_to_color(c_red, c_green, c_blue, red, green, blue, wavelength)
    plot_spectra_from_color(x_color,y_color, red, green, blue, wavelength)
    
def obtain_data(n_samples):
    
    ''' Obtains x-y coodrinates of color samples by testing random concentration values for stock solutions.
        Outputs a dataset with the concentrations that were tested, and the correspoinding x-y color coordinates.'''
    
    df = pd.read_excel('Food dye Spectra.xlsx')
    red = df[['e_red']].values
    blue = df[['e_blue']].values
    green = df[['e_green']].values
    wavelength = df[['Wavelength ']].values
    
    y = []
    x_red = []
    x_green = []
    x_blue = []

    for i in range(n_samples):    
        c_red = np.random.rand(1)[0]
        c_green = np.random.rand(1)[0]
        c_blue = np.random.rand(1)[0]
        x_red.append(c_red)
        x_green.append(c_green)
        x_blue.append(c_blue)
        x_color, y_color, z_color = concentration_to_color(c_red, c_green, c_blue, red, green, blue, wavelength)
        y.append(np.array([x_color, y_color]))
    x = np.vstack((x_red, x_green, x_blue))
    x = x.transpose()
    return x, y

def decision_tree_regressor(x_train, y_train, x_color_goal, y_color_goal):
    '''Optimizing the hyperparameters of the Decision tree regressor'''
    
    df = pd.read_excel('Food dye Spectra.xlsx')
    red = df[['e_red']].values
    blue = df[['e_blue']].values
    green = df[['e_green']].values
    wavelength = df[['Wavelength ']].values
    
    def func1(x,y):
        estimator = DecisionTreeRegressor(max_depth= np.int(x))
        clf = BaggingRegressor(base_estimator=estimator, n_estimators=np.int(y))
        clf = clf.fit(x_train, y_train)
        scores = cross_val_score(clf, x_train, y_train, cv=5)
        MSE = np.mean(scores)
        return 1/MSE

    xmin = 1
    xmax = 50
    ymin = 1
    ymax = 50

    pbounds = {'x': (xmin, xmax), 'y': (ymin, ymax)}

    optimizer = BayesianOptimization(f=func1, pbounds=pbounds, verbose=3)

    optimizer.maximize(init_points = 10, n_iter = 20)

    best_params = optimizer.max["params"]

    found_x = best_params['x']
    found_y = best_params['y']

    max_value = func1(found_x, found_y)
    
    MSE = 1/func1(found_x, found_y)
    
    ###########################################################################################
    '''Training the Decision Tree Regressor with the best hyperparameters'''
    
    estimator = DecisionTreeRegressor(max_depth= np.int(found_x))
    clf = BaggingRegressor(base_estimator=estimator, n_estimators=np.int(found_y))
    clf = clf.fit(x_train, y_train)
    
    ###########################################################################################   
    '''Using the Decision Tree Regressor to obtain the best concentrations for a desired color'''
    
    def func2(x,y,z):
        prediction_array = clf.predict(np.array([x, y, z]).reshape(-1,1).transpose())[0]
        x_color = prediction_array[0]
        y_color = prediction_array[1]
        error = (x_color - x_color_goal)**2 + (y_color - y_color_goal)**2
        return 1/error

    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1
    zmin = 0
    zmax = 1

    pbounds = {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
    optimizer = BayesianOptimization(f=func2, pbounds=pbounds, verbose=3)
    optimizer.maximize(init_points = 20, n_iter = 30)
    best_params = optimizer.max["params"]
    found_x = best_params['x']
    found_y = best_params['y']
    found_z = best_params['z']
    max_value = func2(found_x, found_y, found_z)
    ###########################################################################################
    
    a = np.array([found_x,found_y,found_z]).reshape(1,-1)
    x_predicted, y_predicted = clf.predict(a)[0]
    print('The best concentrations are: ')
    print('Red:', found_x)
    print('Green:', found_y)
    print('Blue:', found_z)
    print()
    print('Which give predicted color coordinates of:')
    print('x predicted', x_predicted)
    print('y predicted', y_predicted)
    
    x_actual,y_actual,z_actual = concentration_to_color(found_x,found_y,found_z, red, green, blue, wavelength)
    print('These concentrations give acutal color coordinates of:')
    print('x actual', x_actual)
    print('y actual', y_actual)
    
    return found_x, found_y, found_z
    
def linear_regression(x_train, y_train, x_color_goal, y_color_goal):
    
    '''Uses linear regression to fit the data'''
    
    df = pd.read_excel('Food dye Spectra.xlsx')
    red = df[['e_red']].values
    blue = df[['e_blue']].values
    green = df[['e_green']].values
    wavelength = df[['Wavelength ']].values
    reg = LinearRegression().fit(x_train, y_train)    
 
    def func2(x,y,z):
        prediction_array = reg.predict(np.array([x, y, z]).reshape(-1,1).transpose())[0]
        x_color = prediction_array[0]
        y_color = prediction_array[1]
        error = (x_color - x_color_goal)**2 + (y_color - y_color_goal)**2
        return 1/error

    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1
    zmin = 0
    zmax = 1

    pbounds = {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
    optimizer = BayesianOptimization(f=func2, pbounds=pbounds, verbose=3)
    optimizer.maximize(init_points = 20, n_iter = 30)
    best_params = optimizer.max["params"]
    found_x = best_params['x']
    found_y = best_params['y']
    found_z = best_params['z']
    max_value = func2(found_x, found_y, found_z)
    
    a = np.array([found_x,found_y,found_z]).reshape(1,-1)
    x_predicted, y_predicted = reg.predict(a)[0]
    print('The best concentrations are: ')
    print('Red:', found_x)
    print('Green:', found_y)
    print('Blue:', found_z)
    print()
    print('Which give predicted color coordinates of:')
    print('x predicted', x_predicted)
    print('y predicted', y_predicted)
    
    x_actual,y_actual,z_actual = concentration_to_color(found_x,found_y,found_z, red, green, blue, wavelength)
    print('These concentrations give acutal color coordinates of:')
    print('x actual', x_actual)
    print('y actual', y_actual)

    return found_x, found_y, found_z