clear;
clc;
%%
% GLOBAL PARAMETERS 
% Parameter values
num_episodes = 15;
%%
% CONSTANT
V_source_value = 48; % input voltage
L_inductance = 10e-6; 
C_capacitance = 40e-3;
R_load = 100;
%%
% Sugnal Processing parameters
prev_time = 0;
init_action = 1; 
%%
V_ref = 30; %30,80,110
%%
% PI controller values 
gain_K = 100;
integral_I= 350000;
periodVal = 0.00001;
pw_percent = 50;

%%
Simulation_Time = 3;
sim('DCDC_BBC_PID',Simulation_Time);

