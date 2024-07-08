function [Y, A, Subj1] = SPM_RegisterAndSelect(DataCycles, VariableOfInterest, RegisterLength, plane)
    arguments
        DataCycles
        VariableOfInterest
        RegisterLength
        plane = 1 %indicates the plane of trajectories, 1 = sagittal
    end


% Register all gait cycles to the specified length in RegisterLength
for i = 1:length(DataCycles)
    Yr = SPM_RegisterLinear(DataCycles(i).(VariableOfInterest{1})(plane,:),RegisterLength);
    Y(i,[1:length(Yr')]) = Yr'; % Gait cycle data variable
    A(i,:) = DataCycles(i).UnloadLevel; % Unloading level variable
    Subj(i,:) = cellstr(DataCycles(i).Subject); % Subject ID variable
end

% outputs the top correlating gait cycles in variables ready for SPM
% analysis.
[~,~,Subj1] = unique(Subj); % Identifies unique names and outputs in numbers.