%__________________________________________________________________________
% Copyright (C) 2022 Todd Pataky



classdef PermuterANOVA2nested_1D < spm1d.stats.nonparam.permuters.APermuterANOVA1DmultiF
    methods
        function [self] = PermuterANOVA2nested_1D(y, A, B)
            self@spm1d.stats.nonparam.permuters.APermuterANOVA1DmultiF(y, A, B)
            self.calc           = spm1d.stats.nonparam.calculators.CalculatorANOVA2nested(self.A, self.B);
            self.nEffects       = 2;
        end
    end
end



