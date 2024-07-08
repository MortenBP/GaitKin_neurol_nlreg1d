function Yr = SPM_RegisterLinear(Y, varargin)

if nargin==1
    Q = 101;
else
    Q = varargin{1};
end

if iscell(Y)
    J  = numel(Y);
    Yr = zeros(J,Q);
    for i = 1:J
        Yr(i,:) = SPM_RegisterLinear(Y{i}, Q);
    end
    return
end
    
Q0 = numel(Y);
t0 = [1:Q0]';
t1 = linspace(1, Q0, Q)';
Yr = interp1(t0, Y, t1);

