import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def conc_to_spectra(conc, red, green, blue, wavelength):
    for i in range(conc.shape[0]):
        c_red = conc[i,0]
        c_green = conc[i,1]
        c_blue = conc[i,2]
        spectra1 = c_red*red + c_green*green + c_blue*blue
        if i == 0:
            spectra = spectra1
        else:
            spectra = np.vstack((spectra, spectra1))
    return spectra

def normalize_and_pca(x_train):
    x_train = x_train.T
    x_train = MinMaxScaler().fit(x_train).transform(x_train)
    x_train = x_train.T

    pca = PCA(n_components=3)
    x_train_new = pca.fit(x_train)
    x_train_spectra = pca.transform(x_train)
    return x_train_spectra

def fitness(spectra, conc,  desired):
    '''Sorts an array by its fitness with the most fit row at the bottom of the array. 
       
       Inputs:
     - spectra: A 2D array of spectra values dimensionally reduced to 3 columns by pca. 
     - conc: A 2D array of the normalized concentrations used to create the spectra. It should have 3 columns and the same number of rows as the spectra array.
     - desired: A 2D array of 1 row and 3 columns, with each column representing the desired dimensionally reduced spectra value. The fitness value is determined by how close the other spectras are to this one. 
    
       Outputs:
     - sorted_array: A 2D array with 7 columns and the same number of rows as the inputs "spectra" and "conc". Columns 0-2 are the spectra, columns 3-6 are the concentrations, and column 7 is the fitness score. The rows are sorted so that the most fit row is at the bottm of the array. 
       '''
    
    fit_score_array = [] 
    for i in range(spectra.shape[0]):
        fit_score = 1/np.sum(np.abs(spectra[i,:] - desired)+0.0001)
        fit_score_array.append(fit_score)
    fitness_score = np.asarray(fit_score_array)
    new_array = np.hstack((spectra, conc, fitness_score.reshape(1,-1).T))
    sorted_array = new_array[np.argsort(new_array[:, -1])]
    lower_fitness, upper_fitness = np.array_split(sorted_array, 2)
    return upper_fitness

def select_parents(sorted_array, n_parents):
    
    '''Randomly selects parents, the ones with a higher fitness will have a higher chance of being selected. Uses a 
       roulette wheel approach where the probability of being selected is proportional to the fitness.
       
       Inputs:
     - sorted_array: 2D array of 7 columns. Columns 0-2 are the spectra after being dimensionally reduced to 3 columns     (pca). Columns 3-6 are the normalized concentrations. Column 7 is the fitness of the row.
     - n_parents determines how many parents are created.
       
       Outputs:
     - parents: 2D array of the same number of columns as sorted_array, but with n_parents rows. Rows with higher fitness should appear in a higher frequency in this row than ones with lower fitness.'''
    
    fitness_list = sorted_array[:,-1]**5 #the number 2 controls how selective the selection process is. A higher number will be more selective
    fitness_sum = np.sum(fitness_list)
    probability = fitness_list/fitness_sum
    cumsum = np.cumsum(probability)
    for itr in range(n_parents):    
        rand_num = np.random.rand()
        for i in range(cumsum.shape[0]):
            if cumsum[i] > rand_num:
                UB = cumsum[i]
                if i == 0:
                    LB = cumsum[i]
                    break
                else:
                    LB = cumsum[i-1]
                    break
        if itr == 0:
            parents = sorted_array[i]
        else:
            parents = np.vstack((parents, sorted_array[i]))
    return parents 


def crossover(parents, n_offspring):
    '''
       Performs a crossover between the parents to create offspring that have characteristcs of both parents.
       
       Inputs:
     - parents: A 2D array of 3 columns, representing the concentrations, and n_parents rows. 
     - n_offspring: An integer representing the number of offspring to be created from these parents. 
    
       Outputs 
     - offspring_array: A 2D array of 3 columns and n_offspring rows. 
    
    '''
    offspring_red_conc_list = []
    offspring_green_conc_list = []
    offspring_blue_conc_list = []
    
    for i in range(n_offspring):
        random_row1 = np.int(np.round(np.random.rand()*parents.shape[0]-1)) 
        random_row2 = np.int(np.round(np.random.rand()*parents.shape[0]-1))  
        parents_conc = parents[:,3:6]
        p1 = parents_conc[random_row1,:] #selects first parent
        p2 = parents_conc[random_row2,:] #selects second parent
        
        #divides parents into red, green, and blue concs 
        p1_red_conc = str(p1[0])
        p1_green_conc = str(p1[1])
        p1_blue_conc = str(p1[2])
        p2_red_conc = str(p2[0])
        p2_green_conc = str(p2[1])
        p2_blue_conc = str(p2[2])
        
        #function to round to sig figs, so 0.5 rounds to 0.500
        def normalize_sig_figs(p1_red_conc):
            if len(p1_red_conc) < 5:
                p1_red_conc = p1_red_conc + '0' + '0'
            return p1_red_conc

        p1_red_conc = normalize_sig_figs(p1_red_conc)
        p1_green_conc = normalize_sig_figs(p1_green_conc)
        p1_blue_conc = normalize_sig_figs(p1_blue_conc)
        p2_red_conc = normalize_sig_figs(p2_red_conc)
        p2_green_conc = normalize_sig_figs(p2_green_conc)
        p2_blue_conc = normalize_sig_figs(p2_blue_conc)
        
        #Function that performs the crossover 
        def cross_parents(p1_red_conc, p2_red_conc): 
            '''p1_red_conc should be a string with 5 letters, ex: 0.500
            p2_red_conc should be same thing, but from the different parent, ex: 0.682. 

            returns a float of the crossovered parents, ex: 0.580'''


            zero = p1_red_conc[0]
            decimal = p1_red_conc[1]
            p1_digit1 = p1_red_conc[2]
            p1_digit2 = p1_red_conc[3]
            p1_digit3 = p1_red_conc[4]

            p2_digit1 = p2_red_conc[2]
            p2_digit2 = p2_red_conc[3]
            p2_digit3 = p2_red_conc[4]

            random_number1 = np.random.rand()
            if random_number1 < 0.5:
                digit1 = p1_digit1
            else:
                digit1 = p2_digit1

            random_number2 = np.random.rand()
            if random_number2 < 0.5:
                digit2 = p1_digit2
            else:
                digit2 = p2_digit2

            random_number3 = np.random.rand()
            if random_number3 < 0.5:
                digit3 = p1_digit3
            else:
                digit3 = p2_digit3

            offspring_red_conc = float(zero + decimal + digit1 + digit2 + digit3)
            return offspring_red_conc

        offspring_red_conc = cross_parents(p1_red_conc, p2_red_conc)
        offspring_green_conc = cross_parents(p1_green_conc, p2_green_conc)
        offspring_blue_conc = cross_parents(p1_blue_conc, p2_blue_conc)

        offspring_red_conc_list.append(offspring_red_conc)
        offspring_green_conc_list.append(offspring_green_conc)
        offspring_blue_conc_list.append(offspring_blue_conc)

    offspring = np.asarray([offspring_red_conc_list, offspring_green_conc_list, offspring_blue_conc_list]).T
    return offspring

def mutation(array, rate):
    '''
        Performs a mutation on some of the values in the offspring array. It converts the value to a string and then changes one of the digits to a random number. 
        
        Inputs:
      - array: A 2D array of the concentrations of the offspring. It should have any number or rows, but 3 columns. The first column is the red concentration, the second is the green and the third is the blue. 
      - rate: The mutation rate. It is a float from 0 to 1, with 1 being a high mutation rate. 
      
        Outputs:
      - array: returns an array with mutated values. It should have the same dimensions of the input array. 
    
    
    '''
    def normalize_sig_figs(p1_red_conc):
        if len(p1_red_conc) < 5:
            p1_red_conc = p1_red_conc + '0' + '0'
        return p1_red_conc

    for j in range(array.shape[0]):
        for i in range(array.shape[1]): 
            if np.random.rand() < rate:
                conc = str(array[j,i])
                conc = normalize_sig_figs(conc)
                column = int(np.round(np.random.uniform(2,4))) #columns 2,3,4
                random_int = str(int(np.round(np.random.uniform(0,9)))) #numbers from 0-9
                if column == 2:
                    digit1 = random_int
                    digit2 = conc[3]
                    digit3 = conc[4]
                elif column == 3:
                    digit1 = conc[2]
                    digit2 = random_int
                    digit3 = conc[4]
                else:
                    digit1 = conc[2]
                    digit2 = conc[3]
                    digit3 = random_int 
                mutated_conc = conc[0] + conc[1] + digit1 + digit2 + digit3
                mutated_conc = float(mutated_conc)
                array[j,i] = mutated_conc
    return array


def GA_algorithm(x_train_spectra, y_train_conc, x_test, n_parents, n_offspring, mutation_rate):
    array = fitness(x_train_spectra, y_train_conc, x_test)
    
    parents = select_parents(array, n_parents)
    offspring = crossover(parents, n_offspring)
    #conc_offspring = offspring[:,x_train_spectra.shape[1]:-1]
    conc_offspring = offspring
    
    conc_offspring_unique = conc_offspring
    #conc_offspring_unique = np.unique(conc_offspring, axis=0)
    
    conc_offspring_mutated = mutation(conc_offspring_unique, mutation_rate)
    
    conc_offspring_mutated = np.abs(conc_offspring_mutated)
    for j in range(conc_offspring_mutated.shape[0]):
        row_sum = np.sum(conc_offspring_mutated[j,:])
        for i in range(conc_offspring_mutated.shape[1]):
            conc_offspring_mutated[j,i] = conc_offspring_mutated[j,i]/row_sum
    return conc_offspring_mutated
