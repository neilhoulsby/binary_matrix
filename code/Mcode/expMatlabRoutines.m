function expMatlabRoutines(K, dense_routines)

% Run Experiments written in MATLAB
addpath(genpath('~/work/MatlabToolboxes/stats'))
load('../data/XSparse.mat')

% SPARSE FUNCTIONS
% tic; paNak_m   = VBMF_Nak(Xs, K, true); paNak_m{1}.time = toc;
tic; paNakNoP_m   = VBMF_Nak(Xs, K, false); paNakNoP_m{1}.time = toc;
% paSee_m   = VBMF_See(Xs, K, true);
paSeeNoP_m   = VBMF_See(Xs, K, false);

% DENSE FUNCTIONS
if dense_routines
    % last parameter = optimise_prior?
%     paG_m     = VBMF_G(Xs, K, true);
    paGNoP_m   = VBMF_G(Xs, K, false);
    % last 2 parameters - optimise_prior? local_sparsity?
%     paS_m                 = VBMF_SJe(Xs, K, true, true);
%     paSNoP_m          = VBMF_SJe(Xs, K, false, true);
    paSNoPNoL_m  = VBMF_SJ(Xs, K, false, false);
end

save('output/pas', 'pa*')

end
