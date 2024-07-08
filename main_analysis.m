%% to do
% create datasets for:
% Violin data
% Achieved unloading
% SPM data

%% Violin plot:
% Load data for gait comfort
for i = 1:length(DataSpatio)
    DataSpatio(i).FeelLevel_reverse = 10 - DataSpatio(i).FeelLevel;
end
xLabel = 'Applied body weight support (% Body weight)';
yLabel = 'Self reported gait comfort, VAS';
groups = [DataSpatio.UnloadLevel];
mean0 = mean([DataSpatio([DataSpatio.UnloadLevel] == 0).FeelLevel_reverse]);
mean10 = mean([DataSpatio([DataSpatio.UnloadLevel] == 10).FeelLevel_reverse]);
mean20 = mean([DataSpatio([DataSpatio.UnloadLevel] == 20).FeelLevel_reverse]);
mean30 = mean([DataSpatio([DataSpatio.UnloadLevel] == 30).FeelLevel_reverse]);
mean40 = mean([DataSpatio([DataSpatio.UnloadLevel] == 40).FeelLevel_reverse]);
mean50 = mean([DataSpatio([DataSpatio.UnloadLevel] == 50).FeelLevel_reverse]);

% Create plot for gait comfort
colorScheme = [0.7,0.7,0.7; 0, 0.4470, 0.7410; 0.9290, 0.6940, 0.1250; 0.4940, 0.1840, 0.5560; 0.4660, 0.6740, 0.1880; 0.6350 0.0780 0.1840];
figure;
xTicks = {'0','10','20','30','40','50'};
daviolinplot([DataSpatio.FeelLevel_reverse], 'groups',groups, 'colors',colorScheme, 'boxcolor','w', 'linkline',1, 'xtlabels', xTicks, 'boxwidth',2, 'violinwidth',1.5,'smoothing',0.5);
hold on
plot([1,2,3,4,5,6],[mean0,mean10,mean20,mean30,mean40,mean50],'.-.', 'LineWidth',1.5)
ylim([0,12])
fontsize(gcf, 12, 'points')
xlabel(xLabel);
ylabel(yLabel);
set(gcf,'position',[1000 500 600 500])



%% Achieved and Target body weight unloading plot
% Robot force data completeness (ID numbers):
% Missing data   : 1
% No-good        : 5
% Ceiling effect?: 3
% Good           : 2,4,6,7,8,9,10,11,12,13,14,15,16,17,18 (15 IDs)

% Convert Voltage to % of total BW
for i = 1:length(DataCycles_selected)
    DataCycles_selected(i).RobotForce_BW = (DataCycles_selected(i).RobotForce / DataCycles_selected(i).MassKg) * 2181.7;
end

% Low-pass Filter RobotForce_BW
SampFreq = 1000;
LP_Band  = 3;
Order    = 1;
for i = 1:length(DataCycles_selected)
    DataCycles_selected(i).RobotForce_BW_filtered = LowPassFilter(double(DataCycles_selected(i).RobotForce_BW), SampFreq, LP_Band, Order);
end

% Function for registering and selecting data of interest, based on DataCycles.
VariableOfInterest = {'RobotForce_BW_filtered'}; % Specify which variable in the DataCycles you wish to perform SPM analysis on
RegisterLength = 101; % Specify length of registration

% Register and select gait cycles
[Y, A, SubjYA] = SPM_RegisterAndSelect(DataCycles_selected,VariableOfInterest,RegisterLength);
Chosen_ones = [2,4,6,7,8,9,10,11,12,13,14,15,16,17,18];

% Create plot

UnloadLevels = [10, 20, 30, 40, 50];
lineWidth = 4;
colorScheme = [0 0.4470 0.7410;0.9290 0.6940 0.1250;0.4940 0.1840 0.5560;0.4660 0.6740 0.1880;0.6350 0.0780 0.1840]; % Color scheme for entire figure, first row is always black for primary timeSeriesData measure.
yLabel = '%Body Weight'; % y axis label for time series plot
GraphTitle = 'Achieved and Target Body Weight Unloading'; % overall figure title
xLabel = '% Stride, initial contact to initial contact'; % x axis label for time series plot
figure;
for i = 1:length(UnloadLevels)
    hold on
    [y,ye]    = deal(mean(Y(A==UnloadLevels(i) & ismember(SubjYA,Chosen_ones),:),1), std(Y(A==UnloadLevels(i) & ismember(SubjYA,Chosen_ones),:),1)); % generates mean and std for each series of data.       
    x         = 0:numel(y)-1;
    plot(x, y, 'color', colorScheme(i,:), 'linewidth',lineWidth); % plots the mean curve
    [y0,y1]   = deal(y+ye, y-ye); % the following lines generate an error cloud based on the std
    [x,y0,y1] = deal( [x(1) x x(end)], [y0(1) y0 y0(end)], [y1(1) y1 y1(end)]);
    [x1,y1]   = deal(fliplr(x), fliplr(y1));
    [X1,Y1]     = deal([x x1], [y0 y1]);
    h         = patch(X1, Y1, 0.7*[1,1,1]);
    yl = yline(UnloadLevels(i),'--', 'LineWidth', 2);
    yl.Color = colorScheme(i,:);
    erAlph = 0.2;
    % plots the error cloud
    set(h, 'FaceColor',colorScheme(i,:), 'FaceAlpha',erAlph, 'EdgeColor','None') 
    hold off
end
legend('10% BWS','','','20% BWS','','','30% BWS','','','40% BWS','','','50% BWS','','','Location','SouthOutside','Orientation','horizontal','NumColumns',3);
fontsize(gcf, 12, 'points')
ylim([0,60])
xlim([0, 100]);
xlabel(xLabel);
xticks([0 20 40 60 80 100])
yticks([0 10 20 30 40 50 60])
ylabel(yLabel);
title(GraphTitle);
set(gcf,'position',[1000 500 600 600])


%% SPM figure
%Manually add the SPM1D toolbox to your workpath

%Specify SPM input: the 9 variables and planes and the register length
VariablesForAnalysis = {'PelvisAngles', 'PelvisAngles', 'PelvisAngles', 'HipAngles', 'HipAngles', 'HipAngles', 'KneeAngles', 'AnkleAngles', 'FootProgressAngles'};
VariablePlane = [1, 2, 3, 1, 2, 3, 1, 1, 1];
RegisterLength = 101; % Specify length of registration

% Perform analyses and create data variables for plotting
% Loops over all specified variables
for p = 1:length(VariablesForAnalysis)

    % Register and select gait cycles for a specified variable
    [Y, A, SubjYA] = SPM_RegisterAndSelect(DataCycles_selected,VariablesForAnalysis(p),RegisterLength,VariablePlane(p));

    % Create the timeSeriesData iterating through unload levels for all participants
    uLevels = unique(A);
    for i = 1:length(uLevels)
        timeSeriesData(:,:,i) = Y(A==uLevels(i),:);
    end
        
    % Perform ANOVA on timeseries data for all 6 unload levels
    alpha      = 0.05; % Alpha corrected for 6 unload levels
    nTests     = 1;
    p_critical = spm1d.util.p_critical_bonf(alpha, nTests);    
    f_anova       = spm1d.stats.anova1rm(Y, A, SubjYA);
    fi_anova      = f_anova.inference(p_critical);

    % Perform SPM ttest on timeSeriesData for 6 unload levels
    alpha      = 0.05; % Alpha corrected for 6 unload levels
    nTests     = 5;
    p_critical = spm1d.util.p_critical_bonf(alpha, nTests);
    for i = 1:length(uLevels)-1
        t(i) = spm1d.stats.ttest_paired(timeSeriesData(:,:,i+1),timeSeriesData(:,:,1));
        ti(i) = t(i).inference(p_critical, 'two_tailed',true);
    end
    
    % Extract cluster values of SPM ANOVA and ttest inference calculations
    if fi_anova.nClusters >= 1
        for o = 1:fi_anova.nClusters
            ClusterCell{o} = fi_anova.clusters{o}.endpoints;
        end
    else
        ClusterCell{1} = [0,0]; % creates an empty cell if no clusters were found
    end
    clusterData{1,:} = ClusterCell;
    clusterCount{1} = fi_anova.nClusters;
    clearvars ClusterCell
    
    for i = 1:length(uLevels)-1
        if ti(i).nClusters >= 1
            for o = 1:ti(i).nClusters
                ClusterCell{o} = ti(i).clusters{o}.endpoints;
            end
        else
            ClusterCell{1} = [0,0]; % creates an empty cell if no clusters were found
        end
        clusterData{i+1,:} = ClusterCell;
        clusterCount{i+1} = ti(i).nClusters;
        clearvars ClusterCell ClusterCounter
    end
    DataStruct(p).Data = timeSeriesData;
    DataStruct(p).Clusters = clusterData;
    DataStruct(p).ClusterCount = clusterCount;
end
for i = 1:length(DataStruct)
    for p = 1:size(DataStruct(i).Data,3)-1
        DataStruct(i).Diff(:,:,p) = abs(DataStruct(i).Data(:,:,p+1) - DataStruct(i).Data(:,:,1));
    end
end

% Plot options
no_deviation_clouds = true; % if true, only basline measure (0%) will have an error cloud.
PlotTitles = {'Pelvis', 'Pelvis', 'Pelvis', 'Hip', 'Hip', 'Hip', 'Knee', 'Ankle', 'Foot Progression'};
PlotYLabels = {'Post.-Ant. Tilt (deg.)', 'Up-Down Obl. (deg.)', 'Ext.-Int. Rot. (deg.)', 'Ext.-Flex. (deg.)', 'Abd.-Add. (deg.)', 'Ext.-Int. Rot. (deg.)', 'Flex. (deg.)', 'Plant.-Dors.Flex. (deg.)', 'Sup.-Pron. (deg.)'};
PlotYLabelsSecondary = {'Abs. Diff.', 'Abs. Diff.', 'Abs. Diff.'};
nParticipantsPlots = [18, 18, 18, 18, 18, 18, 18, 18, 18];
PlotXLabel = {'% Gait cycle'};
LegendText = {'  0% BWS', '10% BWS', '20% BWS', '30% BWS', '40% BWS', '50% BWS', ...
    'SPM ANOVA (p < 0.05)', 'SPM post hoc test: 0% vs 10%', 'SPM post hoc test: 0% vs 20%', 'SPM post hoc test: 0% vs 30%', 'SPM post hoc test: 0% vs 40%', 'SPM post hoc test: 0% vs 50%', ...
    '|Diff. (0%/10%)|(sec.axis)', '|Diff. (0%/20%)|(sec.axis)', '|Diff. (0%/30%)|(sec.axis)', '|Diff. (0%/40%)|(sec.axis)', '|Diff. (0%/50%)|(sec.axis)'};
colorScheme = [0,0,0; 0, 0.4470, 0.7410; 0.9290, 0.6940, 0.1250; 0.4940, 0.1840, 0.5560; 0.4660, 0.6740, 0.1880; 0.6350 0.0780 0.1840];

% Create SPM_Graph of timeSeriesData and Clusters
SPM_GDI_Graph(DataStruct, no_deviation_clouds, PlotTitles, PlotYLabels, PlotYLabelsSecondary, nParticipantsPlots, PlotXLabel, LegendText, colorScheme);



%% Non-linear reg SPM figures with MANOVA
% Matlab Dependencies: SPM1D, fdasrvf matlab package, curve fitting matlab package.
% Python environment dependencies: numpy, scipy, fdasrf, spm1d, nlreg1d,
% skfda, scikit-fda

% Add Matlab folders to Matlab path
addpath(genpath('C:\Users\mlyngpedersen\OneDrive - Syddansk Universitet\FYS PhD\Study 2, Gait analysis\Matlab Scripts'))
addpath(genpath('C:\Users\mlyngpedersen\OneDrive - Syddansk Universitet\FYS PhD\Study 2, Gait analysis\Python Scripts'))

% Set Python 3.9 as active environment
clear classes % clearing classes is needed to refresh non-linear calculations
pyenv('Version', 'C:\Users\mlyngpedersen\AppData\Local\Programs\Python\Python39\python.exe')
% Add Python 3.9 to python system path
path2_39 = 'C:\Users\mlyngpedersen\AppData\Local\Programs\Python\Python39';
if count(py.sys.path, path2_39) == 0
    insert(py.sys.path, int32(0),path2_39)
end

% Add custom python function to python system path
path2py = fileparts(which('nlreg2matlab.py'));
if count(py.sys.path, path2py) == 0
    insert(py.sys.path, int32(0),path2py)
end

% Load and reload cystom python nlreg function
nlregmod = py.importlib.import_module('nlreg2matlab');
py.importlib.reload(nlregmod);
pyenv %reports if environment has been loaded

%Specify SPM input: the 9 variables and planes and the register length
VariablesForAnalysis = {'PelvisAngles', 'PelvisAngles', 'PelvisAngles', 'HipAngles', 'HipAngles', 'HipAngles', 'KneeAngles', 'AnkleAngles', 'FootProgressAngles'};
VariablePlane = [1, 2, 3, 1, 2, 3, 1, 1, 1];
RegisterLength = 101; % Specify length of registration

% Perform analyses and create data variables for plotting
% Loops over all specified variables
for p = 1:length(VariablesForAnalysis)

    % Register and select gait cycles for a specified variable
    [Y, A, SubjYA] = SPM_RegisterAndSelect(DataCycles_selected,VariablesForAnalysis(p),RegisterLength,VariablePlane(p));

    % Non linear registration and displacement field calculations
    Ypy = py.numpy.double(Y);
    Ypy_nlreg_data = nlregmod.get_nlreg_data(Ypy);
    Ypy_nlreg_displacement = nlregmod.get_nlreg_displacement(Ypy);
    Y_nlreg_traj = double(Ypy_nlreg_data+0);
    Y_nlreg_disp = double(Ypy_nlreg_displacement+0);
    Y_nlreg_disp(:,1) = Y_nlreg_disp(:,2);
    Y_nlreg_disp(:,101) = Y_nlreg_disp(:,100);
    Y_nlreg_stacked(:,:,1) = Y_nlreg_traj;
    Y_nlreg_stacked(:,:,2) = Y_nlreg_disp;

    % Create the timeSeriesData iterating through unload levels for all participants
    uLevels = unique(A);
    for i = 1:length(uLevels)
        timeSeriesData(:,:,i) = Y(A==uLevels(i),:);
    end
        
    % Perform ANOVA*lreg on timeseries data for all 6 unload levels
    alpha      = 0.05; % Alpha corrected for 6 unload levels
    nTests     = 1;
    p_critical = spm1d.util.p_critical_bonf(alpha, nTests);    
    f_anova       = spm1d.stats.anova1rm(Y, A, SubjYA);
    fi_anova      = f_anova.inference(p_critical);

    % Perform ANOVA*nlreg on timeseries data for all 6 unload levels
    alpha      = 0.05; % Alpha corrected for 6 unload levels
    nTests     = 1;
    p_critical = spm1d.util.p_critical_bonf(alpha, nTests);    
    f_manova       = spm1d.stats.manova1(Y_nlreg_stacked, A);
    fi_manova      = f_manova.inference(p_critical);

    % Perform ANOVA*Magnitude on timeseries data for all 6 unload levels
    alpha      = 0.05; % Alpha corrected for 6 unload levels
    nTests     = 1;
    p_critical = spm1d.util.p_critical_bonf(alpha, nTests);    
    f_anova_mag       = spm1d.stats.anova1rm(Y_nlreg_traj, A, SubjYA);
    fi_anova_mag      = f_anova_mag.inference(p_critical);

    % Perform ANOVA*Timing on timeseries data for all 6 unload levels
    alpha      = 0.05; % Alpha corrected for 6 unload levels
    nTests     = 1;
    p_critical = spm1d.util.p_critical_bonf(alpha, nTests);    
    f_anova_time       = spm1d.stats.anova1rm(Y_nlreg_disp, A, SubjYA);
    fi_anova_time      = f_anova_time.inference(p_critical);

    % Extract cluster values of lreg and nlreg SPM ANOVA and ttest inference calculations
    if fi_anova.nClusters >= 1
        for o = 1:fi_anova.nClusters
            ClusterCell{o} = fi_anova.clusters{o}.endpoints;
        end
    else
        ClusterCell{1} = [0,0]; % creates an empty cell if no clusters were found
    end
    clusterData{1,:} = ClusterCell;
    clusterCount{1} = fi_anova.nClusters;
    clearvars ClusterCell

    if fi_manova.nClusters >= 1
        for o = 1:fi_manova.nClusters
            ClusterCell{o} = fi_manova.clusters{o}.endpoints;
        end
    else
        ClusterCell{1} = [0,0]; % creates an empty cell if no clusters were found
    end
    clusterData{2,:} = ClusterCell;
    clusterCount{2} = fi_manova.nClusters;
    clearvars ClusterCell

    if fi_anova_mag.nClusters >= 1
        for o = 1:fi_anova_mag.nClusters
            ClusterCell{o} = fi_anova_mag.clusters{o}.endpoints;
        end
    else
        ClusterCell{1} = [0,0]; % creates an empty cell if no clusters were found
    end
    clusterData{3,:} = ClusterCell;
    clusterCount{3} = fi_anova_mag.nClusters;
    clearvars ClusterCell

    if fi_anova_time.nClusters >= 1
        for o = 1:fi_anova_time.nClusters
            ClusterCell{o} = fi_anova_time.clusters{o}.endpoints;
        end
    else
        ClusterCell{1} = [0,0]; % creates an empty cell if no clusters were found
    end
    clusterData{4,:} = ClusterCell;
    clusterCount{4} = fi_anova_time.nClusters;
    clearvars ClusterCell

    % Creates the Datastruct from timeseries data and clusters
    DataStruct(p).Data = timeSeriesData;
    DataStruct(p).Clusters = clusterData;
    DataStruct(p).ClusterCount = clusterCount;
end

% Plot options
no_deviation_clouds = true; % if true, only basline measure (0%) will have an error cloud.
PlotTitles = {'Pelvis', 'Pelvis', 'Pelvis', 'Hip', 'Hip', 'Hip', 'Knee', 'Ankle', 'Foot Progression'};
PlotYLabels = {'Post.-Ant. Tilt (deg.)', 'Up-Down Obl. (deg.)', 'Ext.-Int. Rot. (deg.)', 'Ext.-Flex. (deg.)', 'Abd.-Add. (deg.)', 'Ext.-Int. Rot. (deg.)', 'Flex. (deg.)', 'Plant.-Dors.Flex. (deg.)', 'Sup.-Pron. (deg.)'};
PlotXLabel = {'% Gait cycle'};
LegendText = {'  0% BWS', '10% BWS', '20% BWS', '30% BWS', '40% BWS', '50% BWS', ...
    'SPM ANOVA*lreg (p < 0.05)', 'SPM MANOVA*nlreg', 'SPM ANOVA*Magnitude', 'SPM ANOVA*Timing'};
colorScheme = [0,0,0; 0, 0.4470, 0.7410; 0.9290, 0.6940, 0.1250; 0.4940, 0.1840, 0.5560; 0.4660, 0.6740, 0.1880; 0.6350 0.0780 0.1840];

% Create SPM_Graph of timeSeriesData and Clusters
SPM_GDI_nlreg_Graph(DataStruct, no_deviation_clouds, PlotTitles, PlotYLabels, PlotXLabel, LegendText, colorScheme);


%% Non-linear reg SPM figures without MANOVA
% Matlab Dependencies: SPM1D, fdasrvf matlab package, curve fitting matlab package.
% Python environment dependencies: numpy, scipy, fdasrf, spm1d, nlreg1d,
% skfda, scikit-fda

% Add Matlab folders to Matlab path
addpath(genpath('C:\Users\mlyngpedersen\OneDrive - Syddansk Universitet\FYS PhD\Study 2, Gait analysis\Matlab Scripts'))
addpath(genpath('C:\Users\mlyngpedersen\OneDrive - Syddansk Universitet\FYS PhD\Study 2, Gait analysis\Python Scripts'))

% Set Python 3.9 as active environment
clear classes % clearing classes is needed to refresh non-linear calculations
pyenv('Version', 'C:\Users\mlyngpedersen\AppData\Local\Programs\Python\Python39\python.exe')
% Add Python 3.9 to python system path
path2_39 = 'C:\Users\mlyngpedersen\AppData\Local\Programs\Python\Python39';
if count(py.sys.path, path2_39) == 0
    insert(py.sys.path, int32(0),path2_39)
end

% Add custom python function to python system path
path2py = fileparts(which('nlreg2matlab.py'));
if count(py.sys.path, path2py) == 0
    insert(py.sys.path, int32(0),path2py)
end

% Load and reload cystom python nlreg function
nlregmod = py.importlib.import_module('nlreg2matlab5');
py.importlib.reload(nlregmod);
pyenv %reports if environment has been loaded

%Specify SPM input: the 9 variables and planes and the register length
VariablesForAnalysis = {'PelvisAngles', 'PelvisAngles', 'PelvisAngles', 'HipAngles', 'HipAngles', 'HipAngles', 'KneeAngles', 'AnkleAngles', 'FootProgressAngles'};
VariablePlane = [1, 2, 3, 1, 2, 3, 1, 1, 1];
RegisterLength = 101; % Specify length of registration

% Perform analyses and create data variables for plotting
% Loops over all specified variables
for p = 1:length(VariablesForAnalysis)

    % Register and select gait cycles for a specified variable
    [Y, A, SubjYA] = SPM_RegisterAndSelect(DataCycles_selected,VariablesForAnalysis(p),RegisterLength,VariablePlane(p));

    % Non linear registration and displacement field calculations
    Ypy = py.numpy.double(Y);
    Ypy_nlreg_data = nlregmod.get_nlreg_data(Ypy);
    Ypy_nlreg_displacement = nlregmod.get_nlreg_displacement(Ypy);
    Y_nlreg_traj = double(Ypy_nlreg_data+0);
    Y_nlreg_disp = double(Ypy_nlreg_displacement+0);
    Y_nlreg_disp(:,1) = Y_nlreg_disp(:,2);
    Y_nlreg_disp(:,101) = Y_nlreg_disp(:,100);

    % Create the timeSeriesData iterating through unload levels for all participants
    uLevels = unique(A);
    for i = 1:length(uLevels)
        timeSeriesData(:,:,i) = Y(A==uLevels(i),:);
    end
        
    % Perform ANOVA*lreg on timeseries data for all 6 unload levels
    alpha      = 0.05; % Alpha corrected for 6 unload levels
    nTests     = 1;
    p_critical = spm1d.util.p_critical_bonf(alpha, nTests);    
    f_anova       = spm1d.stats.anova1rm(Y, A, SubjYA);
    fi_anova      = f_anova.inference(p_critical);

    % Perform ANOVA*Magnitude on timeseries data for all 6 unload levels
    alpha      = 0.05; % Alpha corrected for 6 unload levels
    nTests     = 1;
    p_critical = spm1d.util.p_critical_bonf(alpha, nTests);    
    f_anova_mag       = spm1d.stats.anova1rm(Y_nlreg_traj, A, SubjYA);
    fi_anova_mag      = f_anova_mag.inference(p_critical);

    % Perform ANOVA*Timing on timeseries data for all 6 unload levels
    alpha      = 0.05; % Alpha corrected for 6 unload levels
    nTests     = 1;
    p_critical = spm1d.util.p_critical_bonf(alpha, nTests);    
    f_anova_time       = spm1d.stats.anova1rm(Y_nlreg_disp, A, SubjYA);
    fi_anova_time      = f_anova_time.inference(p_critical);

    % Extract cluster values of lreg and nlreg SPM ANOVA and ttest inference calculations
    if fi_anova.nClusters >= 1
        for o = 1:fi_anova.nClusters
            ClusterCell{o} = fi_anova.clusters{o}.endpoints;
        end
    else
        ClusterCell{1} = [0,0]; % creates an empty cell if no clusters were found
    end
    clusterData{1,:} = ClusterCell;
    clusterCount{1} = fi_anova.nClusters;
    clearvars ClusterCell

   if fi_anova_mag.nClusters >= 1
        for o = 1:fi_anova_mag.nClusters
            ClusterCell{o} = fi_anova_mag.clusters{o}.endpoints;
        end
    else
        ClusterCell{1} = [0,0]; % creates an empty cell if no clusters were found
    end
    clusterData{2,:} = ClusterCell;
    clusterCount{2} = fi_anova_mag.nClusters;
    clearvars ClusterCell

    if fi_anova_time.nClusters >= 1
        for o = 1:fi_anova_time.nClusters
            ClusterCell{o} = fi_anova_time.clusters{o}.endpoints;
        end
    else
        ClusterCell{1} = [0,0]; % creates an empty cell if no clusters were found
    end
    clusterData{3,:} = ClusterCell;
    clusterCount{3} = fi_anova_time.nClusters;
    clearvars ClusterCell

    % Creates the Datastruct from timeseries data and clusters
    DataStruct(p).Data = timeSeriesData;
    DataStruct(p).Clusters = clusterData;
    DataStruct(p).ClusterCount = clusterCount;
end

% Plot options
no_deviation_clouds = true; % if true, only basline measure (0%) will have an error cloud.
PlotTitles = {'Pelvis', 'Pelvis', 'Pelvis', 'Hip', 'Hip', 'Hip', 'Knee', 'Ankle', 'Foot Progression'};
PlotYLabels = {'Post.-Ant. Tilt (deg.)', 'Up-Down Obl. (deg.)', 'Ext.-Int. Rot. (deg.)', 'Ext.-Flex. (deg.)', 'Abd.-Add. (deg.)', 'Ext.-Int. Rot. (deg.)', 'Flex. (deg.)', 'Plant.-Dors.Flex. (deg.)', 'Sup.-Pron. (deg.)'};
PlotXLabel = {'% Gait cycle'};
LegendText = {'  0% BWS', '10% BWS', '20% BWS', '30% BWS', '40% BWS', '50% BWS', ...
    'SPM ANOVA (p < 0.05)', 'SPM ANOVA*Magnitude', 'SPM ANOVA*Timing'};
colorScheme = [0,0,0; 0, 0.4470, 0.7410; 0.9290, 0.6940, 0.1250; 0.4940, 0.1840, 0.5560; 0.4660, 0.6740, 0.1880; 0.6350 0.0780 0.1840];

% Create SPM_Graph of timeSeriesData and Clusters
SPM_GDI_nlreg_noManova_Graph(DataStruct, no_deviation_clouds, PlotTitles, PlotYLabels, PlotXLabel, LegendText, colorScheme);
