function [SPM] = anova3nested(Y, A, B, C, varargin)
%__________________________________________________________________________
% Copyright (C) 2022 Todd Pataky



design = spm1d.stats.anova.designs.ANOVA3nested(A, B, C);
model  = spm1d.stats.anova.LinearModel(Y, design.X, varargin{:});
model  = model.fit();
SPM    = model.aov(design.contrasts, design.f_terms);
SPM    = spm1d.stats.spm.SPMFList(SPM);
SPM    = spm1d.stats.anova.set_labels(SPM, design);


