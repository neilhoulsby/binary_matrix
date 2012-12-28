function paout = VBMF_SJ(Xs, K, optimise_prior, local_sparsity)

% S = sigmoid
% J = Jaakkola approx
% LSp = add in global and local sparsity terms

% NEED TO CHECK pa.Xseen in update for Z - dont think it matter those Zs that
% dont contribute to C can have an arb gradient - remove pa.Xseen I think

% main for implementing variational matrix factorisation of a binary matrix
% using a sigmoid likelihood function
% X = input matrix
% pa.Xseen = binary matrix 'mask' indicating which parks of X are observed
% K = number of auxiliary dimensions

% lik: p(xij|A,S) = sigma(xij.* ai*sj)
% priors: p(A) = prod_ik N(aik,0,1)
%         p(S) = prod_kj N(s_kj,0,vsk) - each row has diff variance
% var: Q(A) = prod_ik N(a_ik|a^bar_ik|a^til_ik)
%      Q(S) = prod_ik N(s_ik|s^bar_ik|s^til_ik)
%
% CheckGrad: good: all
if local_sparsity; K = K + 2; end
X = full(Xs); X = X*2 - 1;
pa.Xseen = ones(size(X));
[I J] = size(X);
if K > I || K > J; error('K too large!'); end;
pa.I = I; pa.J = J; pa.K = K;
pa.X = X;

% Learning params
gamma = 0.5;     % gradient
alpha = 1;       % curvature
epsilon = 1e-5;  % convergence

% Prior parameters (all prior means are zero).
pa.va = ones(1, K);
pa.vs = ones(K, 1);
pa.vb = 1e10;
if local_sparsity
    pa.va(K - 1) = 1;      % fixed at 1, if have a prior mean of 1, then make v small.
    pa.va(K)     = 1e10;   % local sparsity parameter free to move.
    pa.vs(K)     = 1;
    pa.vs(K - 1) = 1e10;
end

% Initialise posterior - load in parameters for means
mA  = load('../initialisation/mA_init.mat');
mS  = load('../initialisation/mS_init.mat');
mb  = load('../initialisation/mb_init.mat');
pa.meansA = mA.meansA; pa.meansS = mS.meansS; pa.meanb = mb.meanb;
pa.varsA  = ones(size(pa.meansA));
pa.varsS  = ones(size(pa.meansS));
pa.varb   = 1;
if local_sparsity
    pa.meansA(:, K - 1) = ones(I, 1);
    pa.meansA(:, K)     = zeros(I, 1);
    pa.meansS(K, :)     = ones(1, J);
    pa.meansS(K - 1,:)  = zeros(1, J);
    pa.varsA(:, K - 1)  = 1e-10 * ones(I, 1);
    pa.varsA(:, K)      = ones(I, 1);
    pa.varsS(K, :)      = 1e-10 * ones(1, J);
    pa.varsS(K - 1, :)  = ones(1, J);
end


% Calculate initiatial z_{ij} (variational parameter)
pa.Z =  sqrt( (pa.meansA.^2)*pa.varsS + pa.varsA*(pa.meansS.^2) + pa.varsA*pa.varsS + (pa.meansA*pa.meansS).^2 ...
    + 2.*pa.meanb.*(pa.meansA*pa.meansS) + pa.meanb^2 + pa.varb );

% Recompute a better starting b
pa.lambdaZ = lambda(pa.Z);
sum_lambdaZ = sum(sum(pa.Xseen .* pa.lambdaZ));
pa.varb  = ( 1/pa.vb - 2 * sum_lambdaZ )^(-1);
pa.meanb = sum(sum( pa.Xseen .* (0.5 * X + 2 * pa.lambdaZ .* (pa.meansA*pa.meansS)) )) / ...
    (1/pa.vb - 2 * sum_lambdaZ);

% Eval initial variational lower bound
C = LB_SJ(pa);

timeSlot = 1;
iter = 1;
converged = false;
tic;
while( iter < 3 || (~converged  && iter < 100) )
    
    % Update variational parameter
    pa.Z = sqrt( (pa.meansA.^2)*pa.varsS + pa.varsA*(pa.meansS.^2) + pa.varsA*pa.varsS + (pa.meansA*pa.meansS).^2 ...
        + 2.*pa.meanb.*(pa.meansA*pa.meansS) + pa.meanb^2 + pa.varb);
    pa.lambdaZ = lambda(pa.Z);
    
    % Update variances of Q, A then S
    % e.g. varA = I*K <- S = KxJ, permute 1*K*J, Z = I*J, permute I*1*J , product and sum over J
    pa.varsA = (bsxfun(@minus, 1./pa.va, 2 .* sum( bsxfun(@times, permute(pa.Xseen, [1 3 2]), ...
        bsxfun(@times, permute(pa.lambdaZ, [1 3 2]), permute(pa.meansS.^2 + pa.varsS, [3 1 2]))), 3))).^(-1);
    pa.varsS = (bsxfun(@minus, 1./pa.vs, 2 .* sum( bsxfun(@times, permute(pa.Xseen, [3 2 1]), ...
        bsxfun(@times, permute(pa.lambdaZ, [3 2 1]), permute(pa.meansA.^2 + pa.varsA, [2 3 1]))), 3))).^(-1);
    if local_sparsity
        pa.varsA(:,K-1) = 1e-10*ones(I,1);
        pa.varsS(K,:)   = 1e-10*ones(1,J);
    end
    
    % Update means of Q, A then S
    dCdA   = bsxfun(@rdivide,pa.meansA,pa.va) - sum( bsxfun(@times, permute(pa.Xseen, [1 3 2]), ...
        bsxfun(@times, 0.5 .* permute(pa.X, [1 3 2]), permute(pa.meansS, [3 1 2])) ...
        + 2.*bsxfun(@times, permute(pa.lambdaZ, [1 3 2]), bsxfun(@times, pa.meansA, permute(pa.varsS, [3 1 2])) ...
        + bsxfun(@times, permute(pa.meansS, [3 1 2]), permute(pa.meansA*pa.meansS, [1 3 2])) ...
        + bsxfun(@times, repmat(pa.meanb, [I 1 1]), permute(pa.meansS, [3 1 2])) )), 3);
    %     d2CdA2 = pa.varsA.^-1;
    d2CdA2 = bsxfun(@minus, 1./pa.va, 2 .* sum( bsxfun(@times, permute(pa.Xseen, [1 3 2]), ...
        bsxfun(@times, permute(pa.lambdaZ, [1 3 2]), permute(pa.meansS.^2+pa.varsS, [3 1 2]))), 3));
    
    dCdS   = bsxfun(@rdivide,pa.meansS,pa.vs) - sum( bsxfun(@times, permute(pa.Xseen, [3 2 1]), ...
        bsxfun(@times, 0.5 .* permute(pa.X, [3 2 1]), permute(pa.meansA, [2 3 1])) ...
        + 2.*bsxfun(@times, permute(pa.lambdaZ, [3 2 1]), bsxfun(@times, permute(pa.varsA, [2 3 1]), pa.meansS) ...
        + bsxfun(@times, permute(pa.meansA, [2 3 1]), permute(pa.meansA*pa.meansS, [3 2 1])) ...
        + bsxfun(@times, repmat(pa.meanb, [1 J 1]), permute(pa.meansA, [2 3 1])) )), 3);
    %     d2CdS2  pa.varsS.^-1;
    d2CdS2 = bsxfun(@minus, 1./pa.vs, 2 .* sum( bsxfun(@times, permute(pa.Xseen, [3 2 1]), ...
        bsxfun(@times, permute(pa.lambdaZ, [3 2 1]), permute(pa.meansA.^2+pa.varsA, [2 3 1]))), 3));
    
    pa.meansA = pa.meansA - gamma.*(d2CdA2.^-alpha).*dCdA;
    pa.meansS = pa.meansS - gamma.*(d2CdS2.^-alpha).*dCdS;
    if local_sparsity
        pa.meansA(:,K-1) = ones(I,1);
        pa.meansS(K,:)   = ones(1,J);
    end
    
    % Update prior variances of Q(A), maybe update Q(S) also
    if optimise_prior
        pa.va = sum((pa.meansA.^2 + pa.varsA), 1) ./ I;
        %         pa.vs = sum((pa.meansS.^2 + pa.varsS), 2) ./ J;
        if local_sparsity
            % leave alone param corresponding to the row of ones, and
            % one corresponding to local sparsity
            pa.va(K - 1) = 1;
            pa.va(K)     = 1e10;
            %             pa.vs(K)     = 1e10;
            %             pa.vs(K - 1) = 1e-10;
        end
    end
    
    % Update sparsity parameter
    pa.lambdaZ = lambda(pa.Z);
    sum_lambdaZ = sum(sum(pa.Xseen .* pa.lambdaZ));
    pa.varb  = ( 1/pa.vb - 2 * sum_lambdaZ )^(-1);   % a little wrong, numerics?
    pa.meanb = sum(sum( pa.Xseen .* (0.5 * pa.X + 2 * pa.lambdaZ .* (pa.meansA*pa.meansS)) )) / ...
        (1/pa.vb - 2 * sum_lambdaZ);
    
    % Eval variational lower bound
    Cold = C;
    [C] = LB_SJ(pa);
    
    % Check for converegence
    if mod(iter, 10) == 0
        disp(sprintf('SJ: %g, iter: %g',C, iter))
        paout{timeSlot}.X = Xs;
        paout{timeSlot}.meansA = pa.meansA;
        paout{timeSlot}.meansS = pa.meansS;
        paout{timeSlot}.varsA = pa.varsA;
        paout{timeSlot}.varsS = pa.varsS;
        paout{timeSlot}.meanb = pa.meanb;
        paout{timeSlot}.varb = pa.varb;
        paout{timeSlot}.va = pa.va;
        paout{timeSlot}.vs = pa.vs;
        paout{timeSlot}.vb = pa.vb;
        paout{timeSlot}.K = K;
        paout{timeSlot}.Niters = iter;
        paout{timeSlot}.C = C;
        paout{timeSlot}.time = toc;
        timeSlot = timeSlot + 1;
    end
    if abs(C - Cold) < epsilon*abs(C)
        converged = true;
    end
    % Tune learning parameters
    if C < Cold
        gamma = gamma * 1.1;
    else
        gamma = gamma / 2;
    end
    % Iterate
    iter = iter + 1;
    
end

paout{timeSlot}.X = Xs;
paout{timeSlot}.meansA = pa.meansA;
paout{timeSlot}.meansS = pa.meansS;
paout{timeSlot}.varsA = pa.varsA;
paout{timeSlot}.varsS = pa.varsS;
paout{timeSlot}.meanb = pa.meanb;
paout{timeSlot}.varb = pa.varb;
paout{timeSlot}.va = pa.va;
paout{timeSlot}.vs = pa.vs;
paout{timeSlot}.vb = pa.vb;
paout{timeSlot}.K = K;
paout{timeSlot}.Niters = iter;
paout{timeSlot}.C = C;
paout{timeSlot}.time = toc;

end

function res = sigma(Z)
res = 1 ./ (1 + exp(-Z));
end

function res = lambda(Z)
res = (0.5 - sigma(Z)) ./ (2 .* Z);
end
