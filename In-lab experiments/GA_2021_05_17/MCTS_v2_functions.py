import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from GA_functions import fitness, select_parents, crossover, mutation, GA_algorithm, GA_algorithm_unnormalized, conc_to_spectra, perform_iteration, set_seed #normalize_and_pca
#import pylab

def zeroth_iteration(conc_array, spectra_array, x_test):
    seed = np.random.randint(0,100)
    set_seed(seed)

    spectra = spectra_array
    desired = x_test
    spectra = spectra.T
    spectra = MinMaxScaler().fit(spectra).transform(spectra).T
    desired = MinMaxScaler().fit(x_test).transform(x_test).T
    desired = desired.reshape(1,-1)[0].reshape(-1,1).T

    median_fitness_list = []
    max_fitness_list = []
    iteration  = []
    mutation_rate_list = []
    fitness_multiplier_list = []
    #Calculate Fitness
    a, median_fitness, max_fitness = fitness(spectra, conc_array, desired) #desired is normalized 

    #Appending to list
    i = 0
    median_fitness_list.append(median_fitness)
    max_fitness_list.append(max_fitness)
    iteration.append(i)
    next_gen_conc = conc_array 
    current_gen_spectra = spectra_array

    #Plotting
    plt.scatter(iteration, median_fitness_list, label = 'median fitness')
    plt.scatter(iteration, max_fitness_list, label = 'max fitness')
    plt.ylim([0,3])
    plt.legend()
    plt.show()
    print('The max fitness is:', max_fitness)
    print('The median fitness is:', median_fitness)
    return next_gen_conc, current_gen_spectra, median_fitness_list, max_fitness_list, iteration, mutation_rate_list, fitness_multiplier_list


def perform_Surrogate_Prediction(next_gen_conc, conc_array_actual, spectra_array_actual):
    #lr = LinearRegression().fit(conc_array_actual, spectra_array_actual)
    #score = lr.score(conc_array_actual, spectra_array_actual)
    #spectra_prediction = lr.predict(next_gen_conc)
    gpr = GaussianProcessRegressor().fit(conc_array_actual, spectra_array_actual)
    score = gpr.score(conc_array_actual, spectra_array_actual)
    spectra_prediction = gpr.predict(next_gen_conc)
    return spectra_prediction, score 

def MCTS(Iterations_per_move, moves, GA_iterations, current_gen_spectra, next_gen_conc, x_test, conc_array_actual, spectra_array_actual, seed, n_samples):
    spectra_array_actual = spectra_array_actual**1
    next_gen_conc_original = next_gen_conc
    
    dictionary_of_moves = {}
    move_fitness_list = []
    moves_list = []
    for move_number in range(moves):
        for GA_iteration in range(GA_iterations):
            if GA_iteration == 0:
                for cols in range(move_number+1):
                    mutation_rate_array = (np.round(np.random.uniform(0,10,Iterations_per_move))/10).reshape(-1,1)
                    fitness_multiplier_array = (np.round(np.random.uniform(1,4,Iterations_per_move))).reshape(-1,1)
                    move = np.hstack((mutation_rate_array, fitness_multiplier_array))
                    if cols == 0:
                        move_array = move
                    else:
                        move_array = np.hstack((move_array, move))
            else:
                optimize_array = np.array([10]*Iterations_per_move).reshape(-1,1)
                move_array, _, _ = GA_algorithm_unnormalized(fitness_array[:, move_number].reshape(-1,1), move_array, optimize_array, n_samples, Iterations_per_move, 0.3,2)

            if move_number == 0:
                all_moves_array = move_array
            else:
                all_moves_array = np.hstack((all_moves_array, move_array))
            
            for I in range(Iterations_per_move):
                Fitness_move_1 = []
                for J in range(0,2*(move_number+1),2):
                    if J == 0:
                        next_gen_conc = next_gen_conc_original
                    next_gen_conc, median_fitness, max_fitness = perform_iteration(current_gen_spectra, next_gen_conc, x_test, 20, n_samples, move_array[I,J],move_array[I,J+1])
                    simulated_spectra, surrogate_score = perform_Surrogate_Prediction(next_gen_conc, conc_array_actual, spectra_array_actual)
                    simulated_spectra = simulated_spectra.T
                    simulated_spectra = MinMaxScaler().fit(simulated_spectra).transform(simulated_spectra).T
                    desired = MinMaxScaler().fit(x_test).transform(x_test).T
                    desired = desired.reshape(1,-1)[0].reshape(-1,1).T
                    a, median_fitness, max_fitness = fitness(simulated_spectra, next_gen_conc, desired) #desired is normalized 
                    Fitness_move_1.append(max_fitness)
                fitness_list = np.asarray(Fitness_move_1).reshape(1,-1)

                if I == 0:
                    fitness_array = fitness_list
                else:
                    fitness_array = np.vstack((fitness_array, fitness_list))

        move_fitness = np.hstack((move_array, fitness_array))
        #Save Moves and Fitness in a dictionary 
        dictionary_of_moves.update({move_number + 1:move_fitness})

    best_move_list = []
    for m in range(moves):
        dictionary_of_moves[moves]
        row_number_max_fitness = np.unravel_index(np.argmax(dictionary_of_moves[m+1][:,-1]), dictionary_of_moves[m+1][:,-1].shape)
        best_move = dictionary_of_moves[m+1][row_number_max_fitness,:]
        best_move_list.append(best_move[-1][-1])
    best_move_array = np.asarray(best_move_list)
    best_move_number = np.unravel_index(np.argmax(best_move_array), best_move_array.shape)
    best_move_number = best_move_number[0] + 1
    dictionary_of_moves[best_move_number]
    row_number_max_fitness = np.unravel_index(np.argmax(dictionary_of_moves[best_move_number][:,-1]), dictionary_of_moves[best_move_number][:,-1].shape)
    best_play = dictionary_of_moves[best_move_number][row_number_max_fitness, :]
    max_fitness = best_play[0][-1]
    mutation_rate = best_play[0][0]
    fitness_multiplier = best_play[0][1]
    return mutation_rate, fitness_multiplier, best_play, best_move_number, max_fitness, surrogate_score, desired, current_gen_spectra


def nth_iteration(Iterations, Moves_ahead, GA_iterations, n_samples, current_gen_spectra, next_gen_conc, x_test, conc_array_actual, spectra_array_actual, seed, median_fitness_list, max_fitness_list, iteration, mutation_rate_list, fitness_multiplier_list):
    set_seed(seed)
    mutation_rate, fitness_multiplier, best_move, best_move_turn, max_fitness, surrogate_score, desired_1, current_gen_spectra_1 = MCTS(Iterations, Moves_ahead, GA_iterations, current_gen_spectra, next_gen_conc, x_test, conc_array_actual, spectra_array_actual, seed, n_samples)
    print('The best move has a fitness value of', max_fitness)
    print('The best move occurs in', best_move_turn, 'turns.')
    print()
    print('The surrogate model has a score of:', surrogate_score)
    print()
    mutation_rate_list.append(mutation_rate)
    fitness_multiplier_list.append(fitness_multiplier)
    
    current_gen_spectra = current_gen_spectra.T
    current_gen_spectra = MinMaxScaler().fit(current_gen_spectra).transform(current_gen_spectra).T
    next_gen_conc, median_fitness, max_fitness = perform_iteration(current_gen_spectra, next_gen_conc, x_test, 20, n_samples, mutation_rate,fitness_multiplier)
    print(next_gen_conc)
    #conc_array = np.vstack((conc_array, next_gen_conc))
    
    
    return mutation_rate, fitness_multiplier, mutation_rate_list, fitness_multiplier_list, best_move, best_move_turn, max_fitness, surrogate_score, next_gen_conc
    
def plot_fitness(next_gen_conc, current_gen_spectra, x_test, median_fitness_list, max_fitness_list, iteration):
    #Normalize Data 
    spectra = current_gen_spectra
    spectra = spectra.T
    spectra = MinMaxScaler().fit(spectra).transform(spectra).T
    desired = MinMaxScaler().fit(x_test).transform(x_test).T
    desired = desired.reshape(1,-1)[0].reshape(-1,1).T

    #Calculate Fitness

    a, median_fitness, max_fitness = fitness(spectra, next_gen_conc, desired) #desired is normalized 

    #Appending to list
    i = iteration[-1]
    i = i + 1
    median_fitness_list.append(median_fitness)
    max_fitness_list.append(max_fitness)
    iteration.append(i)

    #Plotting
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(iteration, median_fitness_list, label = 'median fitness')
    ax.plot(iteration, max_fitness_list, label = 'max fitness')
    #ax[0].set_ylim([0,5])
    #ax[1].set_ylim([0,5])
  
    ax.set_xticks(iteration)
    
    ax.set_ylabel('fitness')
    ax.set_xlabel('Iteration')
    ax.legend()
    plt.show()
    print('The max fitness is:', max_fitness)
    print('The median fitness is:', median_fitness)
    return median_fitness_list, max_fitness_list, iteration  
    
def plot_spectra(current_gen_spectra, x_test, wavelength, iteration):
    spectra = current_gen_spectra
    spectra = spectra.T
    spectra = MinMaxScaler().fit(spectra).transform(spectra).T
    desired = MinMaxScaler().fit(x_test).transform(x_test).T
    desired = desired.reshape(1,-1)[0].reshape(-1,1).T
    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (20, 7))
    ax[0].plot(wavelength,desired.T, label = 'Target', linewidth = 5, c = 'k')
    ax[0].set_title('Spectra of All Samples')
    ax[0].set_ylabel('Absorbance')
    ax[0].set_xlabel('Wavelength (nm)')
    for ii in range(spectra.shape[0]):
        ax[0].plot(wavelength,spectra[ii,:])
    ax[0].legend()
    
    fitness_list = []    
    for ii in range(spectra.shape[0]):
        fitness = 1/np.sum(np.abs(spectra[ii,:] - desired))
        fitness_list.append(fitness)
    fitness_array = np.asarray(fitness_list).reshape(-1,1)
    array = np.hstack((spectra, fitness_array))
    sorted_array = array[np.argsort(array[:, -1])]
    ax[1].plot(wavelength,desired.T, label = 'Target', linewidth = 5, c = 'k')
    ax[1].plot(wavelength, sorted_array[-1,:-1], label = 'Best Sample', linewidth = 3)
    ax[1].set_title('Spectra of Best Sample')
    ax[1].set_ylabel('Absorbance')
    ax[1].set_xlabel('Wavelength (nm)')
    figure_name = 'Iteration_' + str(iteration[-1]) + '.png'
    plt.savefig(figure_name)
        
        
        
        
        
