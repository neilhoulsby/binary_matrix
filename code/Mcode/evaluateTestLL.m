function pa = evaluateTestLL(pa, algorithm,  Xs, Xs_test)

% 1 = use simple pass mean through prior
% 2 = integrate Gaussian approx to post against sigmoid
%       -> do integral approximately using method in 
%          Mackay - Evidence framework applied to Classification Networks

if strcmp(algorithm, 'GF_cc') 
    tmp = load('../CCcode/output/paGF_mA.txt');
    pa.meansA = tmp;
    tmp = load('../CCcode/output/paGF_mS.txt');
    pa.meansS = tmp';
    tmp = load('../CCcode/output/paGF_vA.txt');
    pa.varsA = tmp;
    tmp = load('../CCcode/output/paGF_vS.txt');
    pa.varsS = tmp';
    tmp = load('../CCcode/output/paGF_pvA.txt');
    pa.va = tmp;
    tmp = load('../CCcode/output/paGF_pvS.txt');
    pa.vs = tmp';
    pa.vx   = load('../CCcode/output/paGF_vx.txt');
    pa.C    = load('../CCcode/output/paGF_cost.txt');
    pa.time = load('../CCcode/output/paGF_time.txt');
elseif strcmp(algorithm, 'SFSse_cc') 
    tmp = load('../CCcode/output/paSFSse_mA.txt');
    pa.meansA = tmp;
    tmp = load('../CCcode/output/paSFSse_mS.txt');
    pa.meansS = tmp';
    tmp = load('../CCcode/output/paSFSse_vA.txt');
    pa.varsA = tmp;
    tmp = load('../CCcode/output/paSFSse_vS.txt');
    pa.varsS = tmp';
    tmp = load('../CCcode/output/paSFSse_pvA.txt');
    pa.va = tmp;
    tmp = load('../CCcode/output/paSFSse_pvS.txt');
    pa.vs = tmp';
    pa.meanb = load('../CCcode/output/paSFSse_mb.txt');
    pa.varb = load('../CCcode/output/paSFSse_vb.txt');
    pa.vb = load('../CCcode/output/paSFSse_pvb.txt');
    pa.C    = load('../CCcode/output/paSFSse_cost.txt');
    pa.time = load('../CCcode/output/paSFSse_time.txt');
    pa.NSSrows = load('../CCcode/output/paSFSse_NSSrows.txt');
    pa.NSScols = load('../CCcode/output/paSFSse_NSScols.txt');
    pa.NSSmatrix = load('../CCcode/output/paSFSse_NSSmatrix.txt');
    pa.Niters = load('../CCcode/output/paSFSse_Niters.txt');
elseif strcmp(algorithm, 'SSVIub_R') 
    tmp = load('../Rcode/SVI/noBiasedSampling/output/mA.txt');
    pa.meansA = tmp;
    tmp = load('../Rcode/SVI/noBiasedSampling/output/mS.txt');
    pa.meansS = tmp';
    tmp = load('../Rcode/SVI/noBiasedSampling/output/vA.txt');
    pa.varsA = tmp;
    tmp = load('../Rcode/SVI/noBiasedSampling/output/vS.txt');
    pa.varsS = tmp';
    pa.meanb = load('../Rcode/SVI/noBiasedSampling/output/mb.txt');
    pa.varb =  load('../Rcode/SVI/noBiasedSampling/output/vb.txt');
    tmp = load('../Rcode/SVI/noBiasedSampling/output/pvA.txt');
    pa.va = tmp';
    tmp = load('../Rcode/SVI/noBiasedSampling/output/pvS.txt');
    pa.vs = tmp;
    tmp = load('../Rcode/SVI/noBiasedSampling/output/pvb.txt');
    pa.vb = tmp;
elseif strcmp(algorithm, 'SSVIb_R') 
    tmp = load('../Rcode/SVI/biasedSampling/output/mA.txt');
    pa.meansA = tmp;
    tmp = load('../Rcode/SVI/biasedSampling/output/mS.txt');
    pa.meansS = tmp';
    tmp = load('../Rcode/SVI/biasedSampling/output/vA.txt');
    pa.varsA = tmp;
    tmp = load('../Rcode/SVI/biasedSampling/output/vS.txt');
    pa.varsS = tmp';
    pa.meanb = load('../Rcode/SVI/biasedSampling/output/mb.txt');
    pa.varb =  load('../Rcode/SVI/biasedSampling/output/vb.txt');
    tmp = load('../Rcode/SVI/biasedSampling/output/pvA.txt');
    pa.va = tmp';
    tmp = load('../Rcode/SVI/biasedSampling/output/pvS.txt');
    pa.vs = tmp;
    tmp = load('../Rcode/SVI/biasedSampling/output/pvb.txt');
    pa.vb = tmp;
end

if ~ (strcmp(algorithm, 'Nak_m') ||...
        strcmp(algorithm, 'See_m') ||...
        strcmp(algorithm, 'G_m')||...
        strcmp(algorithm, 'GF_cc') )
    Pmeans = pa.meansA * pa.meansS + pa.meanb;
    Pvars  = pa.meansA.^2 * pa.varsS + pa.varsA * pa.meansS.^2 + pa.varsA * pa.varsS ...
        + pa.varb;
else
    Pmeans = pa.meansA * pa.meansS;
    Pvars  = pa.meansA.^2 * pa.varsS + pa.varsA * pa.meansS.^2 + pa.varsA * pa.varsS;
end
% Integrate sgmoid vs. Gaussian (approx) posterior - approximately.
P = Pmeans .* (1 ./ sqrt(1 + 0.125 * pi * Pvars));
P = 1 ./ (1 + exp(-P));  % CHECK THIS!!

pa.eLLtrain = exp(mean(mean( log(P) .* (Xs == 1) +...
    log(1 - P) .* (Xs ~= 1) )));
pa.eLLtest  = exp(mean(mean( log(P) .* (Xs_test == 1) +...
    log(1 - P) .* (Xs_test ~= 1) )));
pa.Ptrain = mean(mean( P .* (Xs == 1) +...
    (1 - P) .* (Xs ~= 1) ));
pa.Ptest  = mean(mean( P .* (Xs_test == 1) +...
    (1 - P) .* (Xs_test ~= 1) ));
pred = P > 0.5;
pa.PCtrain = mean(mean( pred == Xs ));
pa.PCtest  = mean(mean( pred == Xs_test ));


if strcmp(algorithm, 'Nak_m')
    X = full(Xs); % Nak aims for 0,1 (check Seeg)  
else
    X = full(Xs) * 2 - 1;
end
if  strcmp(algorithm, 'G_m') ||...
        strcmp(algorithm, 'GF_cc') ||...
        strcmp(algorithm, 'Nak_m')
    pa.C = LB_G(pa, X); 
elseif strcmp(algorithm, 'S_m') ||...
        strcmp(algorithm, 'Se_m') ||...
        strcmp(algorithm, 'SFSSe_cc') ||...
        strcmp(algorithm, 'SSVIub_R') ||...
        strcmp(algorithm, 'SSVIb_R') ||...
        strcmp(algorithm, 'See_m')
    pa.C = LB_SJ(pa, X); 
end
        

end
