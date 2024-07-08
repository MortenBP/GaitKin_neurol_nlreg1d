function SPM_GDI_nlreg_Graph(DataStruct, no_deviation_clouds, PlotTitles, PlotYLabels, PlotXLabel, LegendText, colorScheme, lineWidth, errorAlpha)
    arguments
        DataStruct
        no_deviation_clouds = true
        PlotTitles = {'Pelvis', 'Pelvis', 'Pelvis', 'Hip', 'Hip', 'Hip', 'Knee', 'Ankle', 'Foot Progression'};
        PlotYLabels = {'Post.-Ant. Tilt (deg.)', 'Up-Down Obl. (deg.)', 'Ext.-Int. Rot. (deg.)', 'Ext.-Flex. (deg.)', 'Abd.-Add. (deg.)', 'Ext.-Int. Rot. (deg.)', 'Flex. (deg.)', 'Plant.-Dors.Flex. (deg.)', 'Sup.-Pron. (deg.)'};
        PlotXLabel = {'% Gait cycle'};
        LegendText = {'Missing legend text'};
        colorScheme = [0,0,0;turbo(numel(DataStruct))]; % Color scheme for entire figure, first row is always black for primary timeSeriesData measure.
        lineWidth = 2
        errorAlpha = 0.2
    end


% Number of cell variables
[~,~,numLevels] = size(DataStruct(1).Data); % Number of unload levels
numFeatures = numel(DataStruct); % Number of features for analysis

% Scaling values for adjusting size of plot content
% Scalings for Y-axis
timeSeriesHeightScaler = 1;     % *3 plots
subplotScaler = 0.1;            % *12 plots
subplotSpacingScaler = 0.15;    % *5 spaces
topSpacingScaler = 0.0;         % *1 spaces
botSpacingScaler = 0.5;         % *1 spaces
% Scalings for X-axis
PlotWidthScaler = 1.4;          % *3 plots
horizontalSpacingScaler = 0.35; % *2 spaces
LeftSideSpacingScaler = 0.3;    % *1 spaces
RightSideSpacingScaler = 0.1;   % *1 spaces
% Sum of all scalings for overall plot scale
HeightPartitions = timeSeriesHeightScaler*3 + subplotScaler*12 + subplotSpacingScaler*5 + topSpacingScaler + botSpacingScaler; %The height of the 15 rows + spacing
WidthPartitions = PlotWidthScaler*3 + horizontalSpacingScaler*2 + LeftSideSpacingScaler + RightSideSpacingScaler; %The width of the three columns + spacing

% Create spacing variable for subplot
timeSeriesHeight = timeSeriesHeightScaler/HeightPartitions; % Space for time series plot, in percentage of total plot space
subplotHeight = subplotScaler/HeightPartitions; % Height of each subplot
subplotSpacing = subplotSpacingScaler/HeightPartitions; % Space between main plot and subplots
topSpacing = topSpacingScaler/HeightPartitions; % Space at top of figure
bottomSpacing = botSpacingScaler/HeightPartitions; % Space at the bottom of figure

PlotWidth = PlotWidthScaler/WidthPartitions; % Defines the proportion of horizontal space each plot should take
horizontalSpaceing = horizontalSpacingScaler/WidthPartitions; % Defines the amount of space between plots
LeftsideSpacing = LeftSideSpacingScaler/WidthPartitions; % Space at the sides of figure
RightsideSpacing = RightSideSpacingScaler/WidthPartitions; % Space at the sides of figure

%Plot position calculation
PlotWidthCount = 0;
HorizontalSpaceCount = 0;
TimeSeriesHeightCount = 2;
SubplotHeightCount = 12;
SubplotSpacingCount = 5;
featureCount = 1;
for i = 1:3 % number of mean plot rows
    tempPlotWidthCount = PlotWidthCount;
    tempHorizontalSpaceCount = HorizontalSpaceCount;
    for p = 1:3 % number of mean plot collumns
        tempSubplotHeightCount = SubplotHeightCount;
        tempSubplotSpacingCount = SubplotSpacingCount-1;
        PlotPositions(featureCount).meanPlot = [LeftsideSpacing + PlotWidth*tempPlotWidthCount + horizontalSpaceing*tempHorizontalSpaceCount, timeSeriesHeight*TimeSeriesHeightCount + subplotHeight*tempSubplotHeightCount + subplotSpacing*tempSubplotSpacingCount + bottomSpacing, PlotWidth, timeSeriesHeight];
        tempSubplotSpacingCount = tempSubplotSpacingCount-1;
        for o = 1:4 % number of subplot rows for each mean plot
            subPlotPlots{o,:} = [LeftsideSpacing + PlotWidth*tempPlotWidthCount + horizontalSpaceing*tempHorizontalSpaceCount, timeSeriesHeight*TimeSeriesHeightCount + subplotHeight*tempSubplotHeightCount + subplotSpacing*tempSubplotSpacingCount + bottomSpacing, PlotWidth, subplotHeight];
            PlotPositions(featureCount).subPlot = subPlotPlots;
            tempSubplotHeightCount = tempSubplotHeightCount-1;
        end
        tempPlotWidthCount = tempPlotWidthCount+1;
        tempHorizontalSpaceCount = tempHorizontalSpaceCount+1;
        featureCount = featureCount+1;
    end
    TimeSeriesHeightCount = TimeSeriesHeightCount-1;
    SubplotHeightCount = SubplotHeightCount-4;
    SubplotSpacingCount = SubplotSpacingCount-1;
end


% Create a new figure
pixelHeight = 800;
pixelWidth = pixelHeight*(WidthPartitions/HeightPartitions);
figure;
set(gcf,'position',[600 100 pixelWidth pixelHeight])
% Create the time-series subplot
for k = 1:numFeatures
    subplot('Position', PlotPositions(k).meanPlot); % specifies the mean+std subplot position on the plot
    for i = 1:numLevels
        hold on
        [y,ye]    = deal(mean(DataStruct(k).Data(:,:,i),1), std(DataStruct(k).Data(:,:,i),1)); % generates mean and std for each series of data.       
        x         = 0:numel(y)-1;
        hplots = plot(x, y, 'color', colorScheme(i,:), 'linewidth',lineWidth); % plots the mean curve

        % Extracts line data for the legend
        if k == 1 && i == 1
            legendData = hplots;
        elseif k == 1
            legendData(1,length(legendData)+1) = hplots;
        end

        % Create deviation clouds for all trajectories or just for the first
        % (baseline) trajectory if "no_deviation_clods" is false.
        [y0,y1]   = deal(y+ye, y-ye);
        [x,y0,y1] = deal( [x(1) x x(end)], [y0(1) y0 y0(end)], [y1(1) y1 y1(end)]);
        [x1,y1]   = deal(fliplr(x), fliplr(y1));
        [X,Y]     = deal([x x1], [y0 y1]);
        h         = patch(X, Y, 0.7*[1,1,1]);
        erAlph = errorAlpha;
        if no_deviation_clouds && i ~= 1
            erAlph = 0;
        end
        % plots the error cloud
        set(h, 'FaceColor',colorScheme(i,:), 'FaceAlpha',erAlph, 'EdgeColor','None') 
        hold off

        % Sets the ylim and xlim of the plots based on the range of the
        % error cloud from first (baseline) trajectory.
        if i == 1
            ylim([min(Y)-abs(range(Y))*0.1, max(Y)+abs(range(Y))*0.1])
        end
    end
    % Plot the title in the upper left corner of each mean plot using the
    % ylim.
    yL=ylim;
    ypos = yL(2)-range(yL)*0.12;
    text(2,ypos,PlotTitles(k),'HorizontalAlignment','left','VerticalAlignment','bottom')
    
    %Formats the plot
    xlim([0, 100]);
    ylabel(PlotYLabels(k));
    set(gca, 'XTick', []);
    set(gca, 'XTickLabel', []);
    box on
       
    for i = 1:4
        subplot('Position', PlotPositions(k).subPlot{i});
        
        % Plot the black boxes on each subplot
        hold on;
        for j = 1:numel(DataStruct(k).Clusters{i})
            x = [DataStruct(k).Clusters{i}{j}(1), DataStruct(k).Clusters{i}{j}(2), DataStruct(k).Clusters{i}{j}(2), DataStruct(k).Clusters{i}{j}(1)];
            y = [0, 0, 100, 100];
            hpatch = patch(x, y, colorScheme(i,:), 'EdgeColor', 'none');
            box on
            xlabel(PlotXLabel)
            set(gca, 'XTick', [0,20,40,60,80,100]);
            set(gca, 'XTickLabel', [0,20,40,60,80,100]);
                    % Extracts line data for the legend
            if k == 1
                legendData(1,length(legendData)+1) = hpatch;
            end
        end
        hold off;
       
        % Set the x-axis and y-axis limits for the subplots
        xlim([0, 100]);
        ylim([0, 100]);
        
        % Remove y-axis numbers for all subplots
        set(gca, 'YTick', []);
        set(gca, 'YTickLabel', []);

        % Remove x-axis numbers for all subplots except the last one
        if ismember(k,[1,2,3,4,5,6]) || ismember(i, [1,2,3])
            set(gca, 'XTick', []);
            set(gca, 'XTickLabel', []);
            xlabel([])
        end
    end
end

legend(legendData, LegendText, 'Position',[0.49, 0.04, 0.01, 0.01],'Orientation','vertical','NumColumns',4)