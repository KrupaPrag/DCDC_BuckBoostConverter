# DCDC_BuckBoostConverter

Data-Driven Model Predictive Control (DDMPC) are considered for controlling a DC to DC Buck-Boost converter with an active load. 

The four control methods are: 
  1. PI Controller 
  2. PPO 
  3. Hybrid 1 (PPO with PI controller)
  4. Hybrid 2 (PPO with filtering mechanism)

The Simulink .slx files contains the environment and the RL agent and the MATLAB .m scripts are used to simulate the respective control methods. 



The .m files only needs to be opened.

The parameters to be set are listed in the second code cell. These include [V_ref, Ts], which respectively are the reference voltage value (state as positive scaler), and the sampling time.

Upon running the .m file, the .slx model will be launched. On the top right of this page, there is a 'Data Inspector' icon which can be launched to see visual representation of the training and validation process. 



### REWARD FUNCTION
To see/edit the reward function and the terminating criterion, open the .slx file. 
This Simulink model has 3 primamry block: 
1. BBC (The buck-boost converter model)
2. SignalProcessing (Within this block there is a signalProcessing function which contains the reward function and terminating criterion)
3. RL Agent 



### PACKAGE REQUIREMENTS
R2021a MATLAB
Packages required: Simulink and Simulink Reinforcement Learning Toolbox 
