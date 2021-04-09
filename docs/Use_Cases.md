## Use cases
1. Collect environmental information
- *User* - Researcher initializing OT2 robot
- *Function* - Collect environmental variables
- *Results* - Return current environmental state

2. Set constraints given environmental state
- *User* - Researcher initializing OT2 robot
- *Function* - Impose physical constraints given environment state, e.g., max solution available, percents can’t exceed 100%
- *Results* - Sets physical constraints

3. Check that actions fall within constraints
- *User* - OT2 robot
- *Function* - Checks that available or proposed actions are physically possible given the set constraints (i.e. a go/ no go checkpoint)
- *Results* - Returns whether the actions are okay or must be revised

4. Use Beer-Lambert to simulate the UV/ Vis spectra from inputs
- *User* - Researcher testing data points
- *Function* -  Use physics to obtain output spectra from the inputs
- *Results* - A set of output spectra for each set of input parameters 
	Inputs: dye concentration, basis spectra, set of actions

5. Assign reward for each set of input parameters
- *User* - Researcher testing data points 
- *Function* -  Calculates the reward for a set of input parameters based on the output spectra and desired spectra
- *Results* - A reward for the sample is determined and it is sent back to the RL model 
		
6. Calculate/update physics heuristic (Stretch)
- *User* - Researching using OT2 robot
- *Function* - Calculates heuristic based on physics (i.e. prior knowledge)
- *Results* - Heuristic to inform RL agent

7. Use/ modify heuristic to determine what data points to sample 
- *User* - Researcher looking to collect data to actively train the model
- *Function* -  Uses a reinforcement learning technique to determine the most probable samples to obtain the greatest reward
- *Results* - A set of actions such that the next samples are prepared by the OT2 robot


