function pa = evaluatePerformance(pa, algorithm,  Xs, Xfulls, Nprec, Nunseen, ts)

% 1 = use simple pass mean through prior
% 2 = integrate Gaussian approx to post against sigmoid
%       -> do integral approximately using method in
%          Mackay - Evidence framework applied to Classification Networks

if strcmp(algorithm, 'GF_cc')
    dir = '../CCcode/output/G';
    tmp = load(sprintf('%s/paGF_TS%g_mA.txt', dir, ts));
    pa.meansA = tmp;
    tmp = load(sprintf('%s/paGF_TS%g_mS.txt', dir, ts));
    pa.meansS = tmp';
    tmp = load(sprintf('%s/paGF_TS%g_vA.txt', dir, ts));
    pa.varsA = tmp;
    tmp = load(sprintf('%s/paGF_TS%g_vS.txt', dir, ts));
    pa.varsS = tmp';
    tmp = load(sprintf('%s/paGF_TS%g_pvA.txt', dir, ts));
    pa.va = tmp;
    tmp = load(sprintf('%s/paGF_TS%g_pvS.txt', dir, ts));
    pa.vs = tmp';
    pa.vx   = load(sprintf('%s/paGF_TS%g_vx.txt', dir, ts));
    pa.C    = load(sprintf('%s/paGF_TS%g_cost.txt', dir, ts));
    pa.time = load(sprintf('%s/paGF_TS%g_time.txt', dir, ts));
elseif strcmp(algorithm, 'GFNoP_cc')
    dir = '../CCcode/output/GnoPrior';
    tmp = load(sprintf('%s/paGF_TS%g_mA.txt', dir, ts));
    pa.meansA = tmp;
    tmp = load(sprintf('%s/paGF_TS%g_mS.txt', dir, ts));
    pa.meansS = tmp';
    tmp = load(sprintf('%s/paGF_TS%g_vA.txt', dir, ts));
    pa.varsA = tmp;
    tmp = load(sprintf('%s/paGF_TS%g_vS.txt', dir, ts));
    pa.varsS = tmp';
    tmp = load(sprintf('%s/paGF_TS%g_pvA.txt', dir, ts));
    pa.va = tmp;
    tmp = load(sprintf('%s/paGF_TS%g_pvS.txt', dir, ts));
    pa.vs = tmp';
    pa.vx   = load(sprintf('%s/paGF_TS%g_vx.txt', dir, ts));
    pa.C    = load(sprintf('%s/paGF_TS%g_cost.txt', dir, ts));
    pa.time = load(sprintf('%s/paGF_TS%g_time.txt', dir, ts));
elseif strcmp(algorithm, 'SFSs_cc')
    dir = '../CCcode/output/S';
    tmp = load(sprintf('%s/paSFSse_TS%g_mA.txt', dir, ts));
    pa.meansA = tmp;
    tmp = load(sprintf('%s/paSFSse_TS%g_mS.txt', dir, ts));
    pa.meansS = tmp';
    tmp = load(sprintf('%s/paSFSse_TS%g_vA.txt', dir, ts));
    pa.varsA = tmp;
    tmp = load(sprintf('%s/paSFSse_TS%g_vS.txt', dir, ts));
    pa.varsS = tmp';
    tmp = load(sprintf('%s/paSFSse_TS%g_pvA.txt', dir, ts));
    pa.va = tmp;
    tmp = load(sprintf('%s/paSFSse_TS%g_pvS.txt', dir, ts));
    pa.vs = tmp';
    pa.meanb = load(sprintf('%s/paSFSse_TS%g_mb.txt', dir, ts));
    pa.varb = load(sprintf('%s/paSFSse_TS%g_vb.txt', dir, ts));
    pa.vb = load(sprintf('%s/paSFSse_TS%g_pvb.txt', dir, ts));
    pa.C    = load(sprintf('%s/paSFSse_TS%g_cost.txt', dir, ts));
    pa.time = load(sprintf('%s/paSFSse_TS%g_time.txt', dir, ts));
    pa.NSSrows = load(sprintf('%s/paSFSse_TS%g_NSSrows.txt', dir, ts));
    pa.NSScols = load(sprintf('%s/paSFSse_TS%g_NSScols.txt', dir, ts));
    pa.NSSmatrix = load(sprintf('%s/paSFSse_TS%g_NSSmatrix.txt', dir, ts));
    pa.Niters = load(sprintf('%s/paSFSse_TS%g_Niters.txt', dir, ts));
elseif strcmp(algorithm, 'SFSsNoP_cc')
    dir = '../CCcode/output/SnoPrior';
    tmp = load(sprintf('%s/paSFSse_TS%g_mA.txt', dir, ts));
    pa.meansA = tmp;
    tmp = load(sprintf('%s/paSFSse_TS%g_mS.txt', dir, ts));
    pa.meansS = tmp';
    tmp = load(sprintf('%s/paSFSse_TS%g_vA.txt', dir, ts));
    pa.varsA = tmp;
    tmp = load(sprintf('%s/paSFSse_TS%g_vS.txt', dir, ts));
    pa.varsS = tmp';
    tmp = load(sprintf('%s/paSFSse_TS%g_pvA.txt', dir, ts));
    pa.va = tmp;
    tmp = load(sprintf('%s/paSFSse_TS%g_pvS.txt', dir, ts));
    pa.vs = tmp';
    pa.meanb = load(sprintf('%s/paSFSse_TS%g_mb.txt', dir, ts));
    pa.varb = load(sprintf('%s/paSFSse_TS%g_vb.txt', dir, ts));
    pa.vb = load(sprintf('%s/paSFSse_TS%g_pvb.txt', dir, ts));
    pa.C    = load(sprintf('%s/paSFSse_TS%g_cost.txt', dir, ts));
    pa.time = load(sprintf('%s/paSFSse_TS%g_time.txt', dir, ts));
    pa.NSSrows = load(sprintf('%s/paSFSse_TS%g_NSSrows.txt', dir, ts));
    pa.NSScols = load(sprintf('%s/paSFSse_TS%g_NSScols.txt', dir, ts));
    pa.NSSmatrix = load(sprintf('%s/paSFSse_TS%g_NSSmatrix.txt', dir, ts));
    pa.Niters = load(sprintf('%s/paSFSse_TS%g_Niters.txt', dir, ts));
elseif strcmp(algorithm, 'SFSsNoPNoL_cc')
    dir = '../CCcode/output/SnoPriorNoLocal';
    tmp = load(sprintf('%s/paSFSse_TS%g_mA.txt', dir, ts));
    pa.meansA = tmp;
    tmp = load(sprintf('%s/paSFSse_TS%g_mS.txt', dir, ts));
    pa.meansS = tmp';
    tmp = load(sprintf('%s/paSFSse_TS%g_vA.txt', dir, ts));
    pa.varsA = tmp;
    tmp = load(sprintf('%s/paSFSse_TS%g_vS.txt', dir, ts));
    pa.varsS = tmp';
    tmp = load(sprintf('%s/paSFSse_TS%g_pvA.txt', dir, ts));
    pa.va = tmp;
    tmp = load(sprintf('%s/paSFSse_TS%g_pvS.txt', dir, ts));
    pa.vs = tmp';
    pa.meanb = load(sprintf('%s/paSFSse_TS%g_mb.txt', dir, ts));
    pa.varb = load(sprintf('%s/paSFSse_TS%g_vb.txt', dir, ts));
    pa.vb = load(sprintf('%s/paSFSse_TS%g_pvb.txt', dir, ts));
    pa.C    = load(sprintf('%s/paSFSse_TS%g_cost.txt', dir, ts));
    pa.time = load(sprintf('%s/paSFSse_TS%g_time.txt', dir, ts));
    pa.NSSrows = load(sprintf('%s/paSFSse_TS%g_NSSrows.txt', dir, ts));
    pa.NSScols = load(sprintf('%s/paSFSse_TS%g_NSScols.txt', dir, ts));
    pa.NSSmatrix = load(sprintf('%s/paSFSse_TS%g_NSSmatrix.txt', dir, ts));
    pa.Niters = load(sprintf('%s/paSFSse_TS%g_Niters.txt', dir, ts));
elseif strcmp(algorithm, 'SSVIb_R')
    dir = '../Rcode/SVI/biasedSampling/output';
    tmp = load(sprintf('%s/mA_TS%g.txt', dir, ts));
    pa.meansA = tmp;
    tmp = load(sprintf('%s/mS_TS%g.txt', dir, ts));
    pa.meansS = tmp';
    tmp = load(sprintf('%s/vA_TS%g.txt', dir, ts));
    pa.varsA = tmp;
    tmp = load(sprintf('%s/vS_TS%g.txt', dir, ts));
    pa.varsS = tmp';
    pa.meanb = load(sprintf('%s/mb_TS%g.txt', dir, ts));
    pa.varb =  load(sprintf('%s/vb_TS%g.txt', dir, ts));
    tmp = load(sprintf('%s/pvA_TS%g.txt', dir, ts));
    pa.va = tmp';
    tmp = load(sprintf('%s/pvS_TS%g.txt', dir, ts));
    pa.vs = tmp;
    tmp = load(sprintf('%s/pvb_TS%g.txt', dir, ts));
    pa.vb = tmp;
    pa.time = load(sprintf('%s/time_TS%g.txt', dir, ts));
elseif strcmp(algorithm, 'SSVIbNoP_R')
    dir = '../Rcode/SVI/biasedSamplingNoPriorTuning/output';
    tmp = load(sprintf('%s/mA_TS%g.txt', dir, ts));
    pa.meansA = tmp;
    tmp = load(sprintf('%s/mS_TS%g.txt', dir, ts));
    pa.meansS = tmp';
    tmp = load(sprintf('%s/vA_TS%g.txt', dir, ts));
    pa.varsA = tmp;
    tmp = load(sprintf('%s/vS_TS%g.txt', dir, ts));
    pa.varsS = tmp';
    pa.meanb = load(sprintf('%s/mb_TS%g.txt', dir, ts));
    pa.varb =  load(sprintf('%s/vb_TS%g.txt', dir, ts));
    tmp = load(sprintf('%s/pvA_TS%g.txt', dir, ts));
    pa.va = tmp';
    tmp = load(sprintf('%s/pvS_TS%g.txt', dir, ts));
    pa.vs = tmp;
    tmp = load(sprintf('%s/pvb_TS%g.txt', dir, ts));
    pa.vb = tmp;
    pa.time = load(sprintf('%s/time_TS%g.txt', dir, ts));
elseif strcmp(algorithm, 'SSVIbNoPNoL_R')
    dir = '../Rcode/SVI/biasedSamplingNoPriorTuningNoLocalBias/output';
    tmp = load(sprintf('%s/mA_TS%g.txt', dir, ts));
    pa.meansA = tmp;
    tmp = load(sprintf('%s/mS_TS%g.txt', dir, ts));
    pa.meansS = tmp';
    tmp = load(sprintf('%s/vA_TS%g.txt', dir, ts));
    pa.varsA = tmp;
    tmp = load(sprintf('%s/vS_TS%g.txt', dir, ts));
    pa.varsS = tmp';
    pa.meanb = load(sprintf('%s/mb_TS%g.txt', dir, ts));
    pa.varb =  load(sprintf('%s/vb_TS%g.txt', dir, ts));
    tmp = load(sprintf('%s/pvA_TS%g.txt', dir, ts));
    pa.va = tmp';
    tmp = load(sprintf('%s/pvS_TS%g.txt', dir, ts));
    pa.vs = tmp;
    tmp = load(sprintf('%s/pvb_TS%g.txt', dir, ts));
    pa.vb = tmp;
    pa.time = load(sprintf('%s/time_TS%g.txt', dir, ts));
elseif strcmp(algorithm, 'SSVIub_R')
    dir = '../Rcode/SVI/noBiasedSampling/output';
    tmp = load(sprintf('%s/mA_TS%g.txt', dir, ts));
    pa.meansA = tmp;
    tmp = load(sprintf('%s/mS_TS%g.txt', dir, ts));
    pa.meansS = tmp';
    tmp = load(sprintf('%s/vA_TS%g.txt', dir, ts));
    pa.varsA = tmp;
    tmp = load(sprintf('%s/vS_TS%g.txt', dir, ts));
    pa.varsS = tmp';
    pa.meanb = load(sprintf('%s/mb_TS%g.txt', dir, ts));
    pa.varb =  load(sprintf('%s/vb_TS%g.txt', dir, ts));
    tmp = load(sprintf('%s/pvA_TS%g.txt', dir, ts));
    pa.va = tmp';
    tmp = load(sprintf('%s/pvS_TS%g.txt', dir, ts));
    pa.vs = tmp;
    tmp = load(sprintf('%s/pvb_TS%g.txt', dir, ts));
    pa.vb = tmp;
    pa.time = load(sprintf('%s/time_TS%g.txt', dir, ts));
elseif strcmp(algorithm, 'SSVIubNoP_R')
    dir = '../Rcode/SVI/noBiasedSamplingNoPriorTuning/output';
    tmp = load(sprintf('%s/mA_TS%g.txt', dir, ts));
    pa.meansA = tmp;
    tmp = load(sprintf('%s/mS_TS%g.txt', dir, ts));
    pa.meansS = tmp';
    tmp = load(sprintf('%s/vA_TS%g.txt', dir, ts));
    pa.varsA = tmp;
    tmp = load(sprintf('%s/vS_TS%g.txt', dir, ts));
    pa.varsS = tmp';
    pa.meanb = load(sprintf('%s/mb_TS%g.txt', dir, ts));
    pa.varb =  load(sprintf('%s/vb_TS%g.txt', dir, ts));
    tmp = load(sprintf('%s/pvA_TS%g.txt', dir, ts));
    pa.va = tmp';
    tmp = load(sprintf('%s/pvS_TS%g.txt', dir, ts));
    pa.vs = tmp;
    tmp = load(sprintf('%s/pvb_TS%g.txt', dir, ts));
    pa.vb = tmp;
    pa.time = load(sprintf('%s/time_TS%g.txt', dir, ts));
elseif strcmp(algorithm, 'SSVIubNoPNoL_R')
    dir = '../Rcode/SVI/noBiasedSamplingNoPriorTuningNoLocalBias/output';
    tmp = load(sprintf('%s/mA_TS%g.txt', dir, ts));
    pa.meansA = tmp;
    tmp = load(sprintf('%s/mS_TS%g.txt', dir, ts));
    pa.meansS = tmp';
    tmp = load(sprintf('%s/vA_TS%g.txt', dir, ts));
    pa.varsA = tmp;
    tmp = load(sprintf('%s/vS_TS%g.txt', dir, ts));
    pa.varsS = tmp';
    pa.meanb = load(sprintf('%s/mb_TS%g.txt', dir, ts));
    pa.varb =  load(sprintf('%s/vb_TS%g.txt', dir, ts));
    tmp = load(sprintf('%s/pvA_TS%g.txt', dir, ts));
    pa.va = tmp';
    tmp = load(sprintf('%s/pvS_TS%g.txt', dir, ts));
    pa.vs = tmp;
    tmp = load(sprintf('%s/pvb_TS%g.txt', dir, ts));
    pa.vb = tmp;
    pa.time = load(sprintf('%s/time_TS%g.txt', dir, ts));
elseif strcmp(algorithm, 'SSVIb_R')
    dir = '../Rcode/SVI/biasedSampling/output';
    tmp = load(sprintf('%s/mA_TS%g.txt', dir, ts));
    pa.meansA = tmp;
    tmp = load(sprintf('%s/mS_TS%g.txt', dir, ts));
    pa.meansS = tmp';
    tmp = load(sprintf('%s/vA_TS%g.txt', dir, ts));
    pa.varsA = tmp;
    tmp = load(sprintf('%s/vS_TS%g.txt', dir, ts));
    pa.varsS = tmp';
    pa.meanb = load(sprintf('%s/mb_TS%g.txt', dir, ts));
    pa.varb =  load(sprintf('%s/vb_TS%g.txt', dir, ts));
    tmp = load(sprintf('%s/pvA_TS%g.txt', dir, ts));
    pa.va = tmp';
    tmp = load(sprintf('%s/pvS_TS%g.txt', dir, ts));
    pa.vs = tmp;
    tmp = load(sprintf('%s/pvb_TS%g.txt', dir, ts));
    pa.vb = tmp;
    pa.time = load(sprintf('%s/time_TS%g.txt', dir, ts));
end

if ~ (regexpcmp(algorithm, 'G*') ||...
        regexpcmp(algorithm, 'Nak*') ||...
        regexpcmp(algorithm, 'See*'))
    Pmeans = pa.meansA * pa.meansS + pa.meanb;
    % some methods don't return variance - fill with zeros,
    %    i.e. assume a delta on the posterior mean.
    Pvars  = pa.meansA.^2 * pa.varsS + pa.varsA * pa.meansS.^2 + pa.varsA * pa.varsS ...
        + pa.varb;
else
    Pmeans = pa.meansA * pa.meansS;
    Pvars  = pa.meansA.^2 * pa.varsS + pa.varsA * pa.meansS.^2 + pa.varsA * pa.varsS;
end


% EVALUATE PRECISION AND RECALL
Pall = Pmeans .* (1 ./ sqrt(1 + 0.125 * pi * Pvars));
Pall = 1 ./ (1 + exp(-Pall));
I = size(pa.meansA, 1);
J = size(pa.meansS, 2);
Nblocks = 1;
Iblock = I / Nblocks;
precs = zeros(Nblocks, Nprec);
recs = zeros(Nblocks, Nprec);
for ibl = 1 : Nblocks
    Xs_tmp = Xs((ibl-1)*Iblock+1 : ibl*Iblock, :);
    Xfulls_tmp = Xfulls((ibl-1)*Iblock+1 : ibl*Iblock, :);
    P = Pall((ibl-1)*Iblock+1 : ibl*Iblock, :);
    for n = 1:Nprec
        T = zeros(Iblock,n);
        P = P - (Xs_tmp == 1) * 1e10;
        [~, ri] = sort(P, 2, 'descend');
        for it = 1:Iblock
            T(it,:) = Xfulls_tmp(it,ri(it,1:n));
        end
        precs(ibl, n) = mean(T(:)==1);
        recs(ibl, n)  = sum(T(:)==1) / (I*Nunseen);
    end
end
pa.prec = mean(precs, 1);
pa.rec  = mean(recs, 1);

% EVALUATE THE TRUE FREE ENERGY
% TODO: for Nak (x \in {0, 1} not {-1, 1})
%       and Seeg (only have lat fun evaluated at few points).

X = full(Xs) * 2 - 1;
if regexpcmp(algorithm, 'G*')
    pa.C = LB_G(pa, X);
elseif strcmp(algorithm, 'Nak_m') ||...
        strcmp(algorithm, 'NakNoP_m') ||...
        strcmp(algorithm, 'See_m') ||...
        strcmp(algorithm, 'SeeNoP_m')
    pa.C = NaN;
else
    pa.C = LB_SJ_computeZb(pa, X, true, true);
end

end
