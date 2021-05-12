1. environment (Class) *(Use cases 1, 2, 3)*

	Methods: init, update_environment, get_constraints

	Attributes: Dictionary of solution information
	
	- **init**:
		- Input: Dataframe with ‘Label’, ‘Concentration’, ‘Volume’
		- Function: Initializes environment class
		- Output: None
	- **update_environment**:
		- Input: Dictionary of actions
		- Function: Calculates environment state given actions. Updates environment attribute.
		- Output: Set current attribute with new calculated ones
	- **check_constraints**:
		- Input: An action
		- Function: Assert if action is allowed by constraints
		- Output: Boolean. True for valid action. 

2. beers_law (Function) *(Use case 4)*

	*Input:* Attenuation coefficient data dataframe, solution dictionary, path length

	*Function:* Apply Beer’s law to hypothetical solution 

	*Output:* Dataframe of ‘wavelength’, ‘absorbance’ 

3. Max_wavelength (Function) *(Use cases 4, 5)*

	*Input:* Dataframe of wavelength and absorbance

	*Function:* Find maximum wavelength absorbed corresponding to hypothetical color of solution

	*Output:* Wavelength as float

4. create_actions (Function) *(Use case 7)*

	*Input:* Batch size, Environment

	*Function:* Propose set of actions using RL algorithm. Check each action is physically possible (using environment object) until a complete set of actions are created

	*Output:* Dictionary of a batch of actions

5. determine_reward (Function) *(Use case 5)*

	*Input:* Desired color/max wavelength , current solution’s color/ max wavelength 

	*Function:* Calculates how close the current color is to the desired color 

	*Output:* Reward as float 

6. update_RL_state (Function) *(Use case 7)*

	*Input:* Set of actions with corresponding regret and uncertainties

	*Function:* Update learning, update heuristic.

	*Output:* Performance / uncertainty of RL model