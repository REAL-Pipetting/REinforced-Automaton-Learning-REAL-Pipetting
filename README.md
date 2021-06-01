# REinforced-Automaton-Learning-REAL-Pipetting
[![Build Status](https://travis-ci.com/REAL-Pipetting/REinforced-Automaton-Learning-REAL-Pipetting.svg?branch=main)](https://travis-ci.com/REAL-Pipetting/REinforced-Automaton-Learning-REAL-Pipetting)
[![Coverage Status](https://coveralls.io/repos/github/REAL-Pipetting/REinforced-Automaton-Learning-REAL-Pipetting/badge.svg?branch=main)](https://coveralls.io/github/REAL-Pipetting/REinforced-Automaton-Learning-REAL-Pipetting?branch=main)


This project integrates reinforcement learning with the open source pipetting robots from Opentrons (OT2) to guide future batched trials in high throughput experiments. We utilize two reinforcement learning algorithms:
1. Genetic Algorithm
2. Gaussian Process Batch Upper Confidence Bound (GP-BUCB)<sup>1</sup>


[1] Desautels, T; Krause, A.; Burdick, J. Parallelizing Exploration-Exploitation Tradeoffs in Gaussian Process Bandit Optimization. Journal of Machine Learning Research 15 (2014) 4053-4103.

### Genetic Algorithm
The Genetic algorithm (GA) is a technique based on natural selection. A number of parents is selected from the highest fitness individuals. Genes from these individuals are then randomly selected to form a new generation of samples. This new generation’s genes are randomly mutated and then the cycle repeats and parents are selected from the new generation. See the diagram below for an example of how a batch of the GA algorithm would work. In this case, there are two genes and the fitness of each sample is the sum of the genes. After the fitness calculation, new parents would be selected from the children. 

![GAdiagram](docs/GA_diagram.png)


### GP-BUCB
The Gaussian Process - Upper Confidence Bound algorithm discretizes features to develop a parameter space with number of dimensions equal to the number of features. At each point in the space holds an average reward and standard deviation. It models the reward function as a Gaussian Process. The GP-BUCB selects the sample with the highest upper bound (mean + std * beta^½)  where beta is a hyper parameter that affects how much to weight the standard deviation in the sample process. Higher beta means a higher priority is given to samples with higher uncertainty, encouraging the GP-BUCB to explore the space more. As it samples from the parameter space, the GP-BUCB updates the mean and uncertainty for every point in the parameter space using Gaussian Process Regression. Our implementation of the GP-BUCB uses scikit-learn's implementation of GP Regression. 

![GPBUCBdiagram](docs/GPBUCB_diagram.png)


The diagram below shows the GP-BUCB solving a problem using two features and sampling three times per batch. The location in the parameter space marked with the black box is the location of the solution. The combination of features the GPBUCB chooses to sample each batch are marked green. In this case, the GP-BUCB is able to reach an exact solution, but that does not have to be the case. If given a problem with a solution not in the parameter space, it will converge to the best solution within the parameter space. 

![GPBUCBviz](docs/GPBUCB_viz.png)


There are additional operations depending on which implementation of the GP-BUCB you use.
All are found in ucb.py. 

GPBUCBv2 includes “hallucinated samples”. It is the same as the default GP-BUCB except during batch’s it will do GP Regression after every sample is selected. Since it can not get the true reward for a sample mid-batch, it “hallucinates” the rewards as the current expected reward (the mean). By doing this, it gives the GP-BUCB more confidence in the region around the sample encouraging the algorithm to explore more within a batch. After the batch is done, the hallucinated rewards are replaced with the true rewards.

GPBUCBv3 also uses hallucinated samples. Additionally, GPBUCBv3 will prune regions of the parameter space deemed unlikely to contain the best solution. The criteria it uses is if the best mean - standard deviation (i.e. a conservative estimate of the best possible reward) is higher than a points mean+standard deviation * beta^1/2 (i.e. a optimistic estimate of that points rewards) than that point is removed from the parameter space. By doing this, GPBUCBv3 is faster than GPBUCBv2, especially as more and more of the space is pruned.


Both the GA and GP-BUCB functions only return value features, and the only input they expect while learning is the reward associate with those combination of features. Therefore, any problem may be "hooked-up" to these algorithms as long as one as the user is able to provide the reward given a combination of features. 



### Installation
To install realpy:
1. Clone this repo
2. ```pip install /path/to/directory/with/this/readme/```

Or navigate to this directory in the terminal and ```pip install .```

Afterwards, you can ```import realpy``` for use in any python file.
