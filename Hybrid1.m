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
stopping_criterion = 500;
error_threshold = 0.01;
%%
Ts = 0.00001;
Tf = 10;
V_ref = 30; %30,80,110
%%
% PI controller values 
gain_K = 100;
integral_I= 40000;
periodVal = 1/10000;
pw_percent = 50;
%%
% Parameters
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
EntropyLossWeightVal = 0.02;
MiniBatchSizeVal = ceil(ExperienceHorisonLength*miniBatch_percent); % <=expectationHorisonVal
NumEpochsVal = 5; 
DiscountFactorVal = 0.99;




%%

neg_Vref = -V_ref;

% mdl = 'DCDC_BBC_RL';
mdl = 'DCDC_BBC_hybrid1';
open_system(mdl)
agentblk = [mdl '/RL Agent'];


numObs = 3; % Action State - change the lowwer and upper limit [v0, e, de/dt, prev_ime, prev_error]
% Observation = [State_t action_{t-1}]^T (current state  and last applied action)
observationInfo = rlNumericSpec([numObs,1],...
    'LowerLimit',[-inf -inf -inf]',...
    'UpperLimit',[0 V_ref inf]');
observationInfo.Name = 'observations';
observationInfo.Description = 'integrated error, error, and measured height';
numObservations = observationInfo.Dimension(1);


a = [0;1]; %The actions that can be applied are binary. 
actionInfo = rlFiniteSetSpec(a);

env = rlSimulinkEnv(mdl,agentblk,observationInfo,actionInfo);

%                                         env.ResetFcn = @(in)localResetFcn(in); %See examples 
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
    'SaveAgentValue',10000);% Save agent with value greater than 

trainingStats = train(agent,env,trainOpts);

