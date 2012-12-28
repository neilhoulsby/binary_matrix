function expAnalyseandSave(dense_routines)

load('../data/XSparse.mat')
load('output/pas')
Nprec = 10;
% for i = 1:length(paNak_m)
%     paNak_m{i}    = evaluatePerformance(paNak_m{i}, 'Nak_m',  Xs, Xfulls, Nprec, Nunseen);
% end
for i = 1:length(paNakNoP_m)
    paNakNoP_m{i}    = evaluatePerformance(paNakNoP_m{i}, 'NakNoP_m',  Xs, Xfulls, Nprec, Nunseen);
end
% for i = 1:length(paSee_m)
%     paSee_m{i}    = evaluatePerformance(paSee_m{i}, 'See_m',  Xs, Xfulls, Nprec, Nunseen);
% end
for i = 1:length(paSeeNoP_m)
    paSeeNoP_m{i}    = evaluatePerformance(paSeeNoP_m{i}, 'SeeNoP_m',  Xs, Xfulls, Nprec, Nunseen);
end

if dense_routines
%     for i = 1:length(paG_m)
%         paG_m{i}        = evaluatePerformance(paG_m{i}, 'G_m',  Xs, Xfulls, Nprec, Nunseen);
%     end
    for i = 1:length(paGNoP_m)
        paGNoP_m{i}      = evaluatePerformance(paGNoP_m{i}, 'GNoP_m',  Xs, Xfulls, Nprec, Nunseen);
    end
%     for i = 1:length(paS_m)
%         paS_m{i}                = evaluatePerformance(paS_m{i}, 'S_m',  Xs, Xfulls, Nprec, Nunseen);
%     end
%     for i = 1:length(paSNoO_m)
%         paSNoP_m{i}         = evaluatePerformance(paSNoP_m{i}, 'SNoP_m',  Xs, Xfulls, Nprec, Nunseen);
%     end
    for i = 1:length(paSNoPNoL_m)
        paSNoPNoL_m{i} = evaluatePerformance(paSNoPNoL_m{i}, 'SNoPNoL_m',  Xs, Xfulls, Nprec, Nunseen);
    end
end

% numTSs = load('../CCcode/output/G/paGF_numTSs.txt');
% for i = 1:numTSs
%     paGF_cc{i}          = evaluatePerformance([], 'GF_cc',  Xs, Xfulls, Nprec, Nunseen, i);
% end
numTSs = load('../CCcode/output/GnoPrior/paGF_numTSs.txt');
for i = 1:numTSs
    paGFNoP_cc{i}           = evaluatePerformance([], 'GFNoP_cc',  Xs, Xfulls, Nprec, Nunseen, i);
end
% numTSs = load('../CCcode/output/S/paSFSse_numTSs.txt');
% for i = 1:numTSs
%     paSFSs_cc{i}            = evaluatePerformance([], 'SFSs_cc',  Xs, Xfulls, Nprec, Nunseen, i);
% end
% numTSs = load('../CCcode/output/SnoPrior/paSFSse_numTSs.txt');
% for i = 1:numTSs
%     paSFSsNoP_cc{i}         = evaluatePerformance([], 'SFSsNoP_cc',  Xs, Xfulls, Nprec, Nunseen, i);
% end
numTSs = load('../CCcode/output/SnoPriorNoLocal/paSFSse_numTSs.txt');
for i = 1:numTSs
    paSFSsNoPNoL_cc{i}      = evaluatePerformance([], 'SFSsNoPNoL_cc',  Xs, Xfulls, Nprec, Nunseen, i);
end
% numTSs = load('../Rcode/SVI/biasedSampling/output/numTSs.txt');
% for i = 1:numTSs
%     paSSVIb_R{i}            = evaluatePerformance([], 'SSVIb_R',  Xs, Xfulls, Nprec, Nunseen, i);
% end
% numTSs = load('../Rcode/SVI/biasedSamplingNoPriorTuning/output/numTSs.txt');
% for i = 1:numTSs
%     paSSVIbNoP_R{i}         = evaluatePerformance([], 'SSVIbNoP_R',  Xs, Xfulls, Nprec, Nunseen, i);
% end
numTSs = load('../Rcode/SVI/biasedSamplingNoPriorTuningNoLocalBias/output/numTSs.txt');
for i = 1:numTSs
    paSSVIbNoPNoL_R{i}      = evaluatePerformance([], 'SSVIbNoPNoL_R',  Xs, Xfulls, Nprec, Nunseen, i);
end
% numTSs = load('../Rcode/SVI/noBiasedSampling/output/numTSs.txt');
% for i = 1:numTSs
%     paSSVIub_R{i}       = evaluatePerformance([], 'SSVIub_R',  Xs, Xfulls, Nprec, Nunseen, i);
% end
% numTSs = load('../Rcode/SVI/noBiasedSamplingNoPriorTuning/output/numTSs.txt');
% for i = 1:numTSs
%     paSSVIubNoP_R{i}        = evaluatePerformance([], 'SSVIubNoP_R',  Xs, Xfulls, Nprec, Nunseen, i);
% end
numTSs = load('../Rcode/SVI/noBiasedSamplingNoPriorTuningNoLocalBias/output/numTSs.txt');
for i = 1:numTSs
    paSSVIubNoPNoL_R{i}     = evaluatePerformance([], 'SSVIubNoPNoL_R',  Xs, Xfulls, Nprec, Nunseen, i);
end
% paRand_m   = evaluatePerformance(paRand_m, '',  Xs, Xfulls, Nprec, Nunseen);
% paSVD_m    = evaluatePerformance(paSVD_m, '',  Xs, Xfulls, Nprec, Nunseen);

% Save data
save(sprintf('../results/res_dSet%g_I%g_J%g_%g',dset,I,J,iterNo), ...
    'pa*','Xs','Xfulls','Nunseen','Nprec','dset','iterNo','-v7.3')

end
