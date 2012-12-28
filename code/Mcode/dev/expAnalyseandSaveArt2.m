function expAnalyseandSaveArt2()

load('../data/XSparse.mat')
load('output/pas')

paNak_m    = evaluateTestLL(paNak_m, 'Nak_m',  Xs, Xs_test);
paSee_m    = evaluateTestLL(paSee_m, 'See_m',  Xs, Xs_test);
paG_m      = evaluateTestLL(paG_m, 'G_m',  Xs, Xs_test);
paS_m      = evaluateTestLL(paS_m, 'S_m',  Xs, Xs_test);
paSe_m     = evaluateTestLL(paSe_m, 'Se_m',  Xs, Xs_test);
paGF_cc    = evaluateTestLL([], 'GF_cc',  Xs, Xs_test);
paSFSse_cc = evaluateTestLL([], 'SFSse_cc',  Xs, Xs_test);
paSSVIub_R = evaluateTestLL([], 'SSVIub_R',  Xs, Xs_test);
paSSVIb_R  = evaluateTestLL([], 'SSVIb_R',  Xs, Xs_test);

% Save data
save(sprintf('../results/res_dSet%g_I%g_J%g_%g',dset,I,J,iterNo), ...
    'pa*','Xs','Xs_test','dset','iterNo','-v7.3')

end
