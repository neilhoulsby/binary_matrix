function paout = VBMF_G(Xs, K, optimise_prior)

% G = Gaussian likelihood

% main for inplementing variational matrix factorisation w/ Gaussian noise
% X = input matrix
% Xseen = binary matrix 'mask' indicating which parks of X are observed
% K = number of auxiliary dimensions

% lik: p(xij|A,S) = N(xij| ai.*a.j, vx)
% priors: p(A) = prod_ik N(aik,0,1)
%         p(S) = prod_kj N(s_kj,0,vsk) - each row has diff variance
% var: Q(A) = prod_ik N(a_ik|a^bar_ik|a^til_ik)
%      Q(S) = prod_ik N(s_ik|s^bar_ik|s^til_ik)

% Unit test derivatives (either by num calc grad or check either side of update) 
%    - OK: dCdA, dCdS, vx, vs, varsA, varsS

X = full(Xs); X = X*2 - 1;
Xseen = ones(size(X));              % currently unused

[I J] = size(X);
N = sum(sum(Xseen));
if K > I || K > J; error('K too large!'); end;
pa.I = I; pa.J = J; pa.K = K;
pa.X = X;

% Learning params
gamma = 0.5;
alpha = 1;
epsilon = 1e-4;

% Prior parameters (all prior means are zero).
pa.va = ones(1, K);
pa.vs = ones(K, 1);

% Initialise posterior - load in parameters for means
mA     = load('../initialisation/mA_init.mat');
mS     = load('../initialisation/mS_init.mat');
vx     = load('../initialisation/vx_init.mat');
pa.meansA = mA.meansA; pa.meansS = mS.meansS; pa.vx = vx.vx;
pa.varsA  = ones(size(pa.meansA));
pa.varsS  = ones(size(pa.meansS));

% Eval variational lower bound
C = LB_G(pa);

iter = 1;
converged = false;
timeSlot = 1;
tic;
while( iter < 3 || (~converged  && iter < 250) )
    
    % Update variances of Q, A then S
    % e.g. varA S = KxJ, repmat -> KxJxI, .*Sxeen (1xJxI) for all K, sum over J, 
    % permute+invert -> varsA I*K  
    pa.varsA = ( bsxfun(@plus, 1./pa.va, permute(sum(bsxfun(@times, repmat((pa.meansS.^2 + pa.varsS)./pa.vx, [1 1 I]), ...
        permute(Xseen,[3 2 1])), 2), [3 1 2])) ).^-1;
    pa.varsS = ( bsxfun(@plus, 1./pa.vs, permute(sum(bsxfun(@times, repmat((pa.meansA.^2 + pa.varsA)./pa.vx, [1 1 J]), ...
        permute(Xseen,[1 3 2])), 1), [2 3 1])) ).^-1;    
    
    % Update means of Q, A then S
    dCdA   = bsxfun(@rdivide, pa.meansA, pa.va) + ( -(Xseen.*(X - pa.meansA*pa.meansS))*pa.meansS' + ...
        sum(bsxfun(@times, bsxfun(@times, pa.meansA, permute(pa.varsS, [3 1 2])), permute(Xseen, [1 3 2])), 3) )./pa.vx;
    dCdS   = bsxfun(@rdivide, pa.meansS, pa.vs) + ( -pa.meansA'*((X - pa.meansA*pa.meansS).*Xseen) + ...
        sum(bsxfun(@times, bsxfun(@times, permute(pa.varsA, [2 3 1]), pa.meansS), permute(Xseen, [3 2 1])), 3) )./pa.vx;
    d2CdA2 = pa.varsA.^-1;
    d2CdS2 = pa.varsS.^-1;

    pa.meansA = pa.meansA - gamma.*(d2CdA2.^-alpha).*dCdA;
    pa.meansS = pa.meansS - gamma.*(d2CdS2.^-alpha).*dCdS;
    
    % Update variances of lik and Q(S)
    pa.vx = SS( (X - pa.meansA*pa.meansS).^2 + pa.varsA*(pa.meansS.^2) + ...
        (pa.meansA.^2)*pa.varsS + pa.varsA*pa.varsS ) / N;

    if optimise_prior
        pa.va = sum((pa.meansA.^2 + pa.varsA), 1) / I;
%         pa.vs = sum((pa.meansS.^2 + pa.varsS), 2) / J;
    end
    
    % Eval variational lower bound
    Cold = C;
    C = LB_G(pa);    
    
    % Check for converegence
    if abs(C - Cold) < epsilon*abs(C)
       converged = true; 
    end

    % Tune learning parameters
    if C < Cold
        gamma = gamma * 1.1;
    else
        gamma = gamma / 2;
    end
    
    if mod(iter, 10) == 0
        disp(sprintf('G: %g, iter: %g',C, iter))

        paout{timeSlot}.X = Xs;
        paout{timeSlot}.meansA = pa.meansA;
        paout{timeSlot}.meansS = pa.meansS;
        paout{timeSlot}.varsA = pa.varsA;
        paout{timeSlot}.varsS = pa.varsS;
        paout{timeSlot}.va = pa.va;
        paout{timeSlot}.vs = pa.vs;
        paout{timeSlot}.K = K;
        paout{timeSlot}.Niters = iter;
        paout{timeSlot}.C = C;
        paout{timeSlot}.time = toc;
        paout{timeSlot}.vx = pa.vx;
        paout{timeSlot}.converged = converged;
        timeSlot = timeSlot + 1;
    end
    
    % Iterate
    iter = iter + 1;
end

paout{timeSlot}.X = Xs;
paout{timeSlot}.meansA = pa.meansA;
paout{timeSlot}.meansS = pa.meansS;
paout{timeSlot}.varsA = pa.varsA;
paout{timeSlot}.varsS = pa.varsS;
paout{timeSlot}.va = pa.va;
paout{timeSlot}.vs = pa.vs;
paout{timeSlot}.K = K;
paout{timeSlot}.Niters = iter;
paout{timeSlot}.C = C;
paout{timeSlot}.time = toc;
paout{timeSlot}.vx = pa.vx;
paout{timeSlot}.converged = converged;
end




