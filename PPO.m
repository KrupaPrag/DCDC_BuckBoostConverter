clear;
clc;
%%
% GLOBAL PARAMETERS 
% Parameter values
num_episodes = 1024;
%%
% Buck Boost Converter Parameters
V_source_value = 48;
L_inductance = 10e-6; 
C_capacitance = 40e-3;
R_load = 100;
%%
% Signal Processing Parameters
prev_time = 0;
init_action = 1; 
stopping_criterion = 500;
error_threshold = 0.1;
%%
Ts = 0.004295;%0.00225;%0.004295;%0.028252;
Tf = 0.3;
V_ref = 80;%30;%80%110;

%%
% RL Parameters
miniBatch_percent = 0.8;
learnRateActor = 0.008;
learnRateCritic= 0.008;
criticLayerSizes= [128 128];
actorLayerSizes= [128 128];
discountFactor= 0.99;
num_epochs= 100;

max_steps = ceil(Tf/Ts);
ExperienceHorisonLength = 75;
ClipFactorVal = 0.2;
EntropyLossWeightVal = 0.01;
MiniBatchSizeVal = ceil(ExperienceHorisonLength*miniBatch_percent); 
NumEpochsVal = 5; 
DiscountFactorVal = 0.99;

%%

% RL Agent

mdl = 'DCDC_BBC_RL';
open_system(mdl)
agentblk = [mdl '/RL Agent'];

numObs = 3; % [v0, e, de/dt]
observationInfo = rlNumericSpec([numObs,1],...
    'LowerLimit',[-inf -inf -inf]',...
    'UpperLimit',[0 V_ref inf]');
observationInfo.Name = 'observations';
observationInfo.Description = 'integrated error, error, and measured height';
numObservations = observationInfo.Dimension(1);

a = [0;1]; 
actionInfo = rlFiniteSetSpec(a);

env = rlSimulinkEnv(mdl,agentblk,observationInfo,actionInfo);

env.ResetFcn = @(in) setVariable(in,'init_action',1);
num_inputs = numObs;        

criticNetwork = [
    imageInputLayer([num_inputs 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(criticLayerSizes(1),'Name','CriticFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(criticLayerSizes(2),'Name','CriticFC2')
    reluLayer('Name','CriticRelu2')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticOpts = rlRepresentationOptions('LearnRate',learnRateCritic,'GradientThreshold',1);


critic = rlValueRepresentation(criticNetwork,observationInfo,'Observation',{'state'},criticOpts);

numAct = numel(actionInfo.Elements);
actorNetwork = [imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(actorLayerSizes(1),'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(actorLayerSizes(2),'Name','ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(numAct,'Name','Action')
    reluLayer('Name','actionProbability')
    ];  

actorOpts = rlRepresentationOptions('LearnRate',learnRateActor,'GradientThreshold',1);


actor = rlStochasticActorRepresentation(actorNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},actorOpts);

agentOpts = rlPPOAgentOptions('ExperienceHorizon',ExperienceHorisonLength,...
                        'ClipFactor',ClipFactorVal,...
                        'EntropyLossWeight',0.02,...
                        'MiniBatchSize',MiniBatchSizeVal,...
                        'NumEpoch',NumEpochsVal,...
                        'AdvantageEstimateMethod','finite-horizon',...
                        'GAEFactor',0.98,...                        
                        'SampleTime',Ts,...
                        'DiscountFactor',DiscountFactorVal);
agent = rlPPOAgent(actor,critic,agentOpts);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes',num_episodes,...
    'MaxStepsPerEpisode',max_steps,...
    'Verbose',true,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',inf,...
    'ScoreAveragingWindowLength',100,...
    'SaveAgentCriteria',"EpisodeReward",...
    'SaveAgentValue',10000);

trainingStats = train(agent,env,trainOpts);