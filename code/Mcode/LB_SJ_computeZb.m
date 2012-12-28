function [C] = LB_SJ_computeZb(pa, X, compute_Z, use_b)
% LB_SJ - Evaluates the Variational Free Energy
%         (Sigmoid likelihood, Jaakkola approximation, Sparsity parameter).

if nargin < 2
    X = pa.X;
end

if use_b
    varsA    = pa.varsA;
    varsS    = pa.varsS;
    meansA   = pa.meansA;
    meansS   = pa.meansS;
    meanb    = pa.meanb;
    varb     = pa.varb;
    vs       = pa.vs;
    va       = pa.va;
    vb       = pa.vb;
    
    if compute_Z
        pa.Z = sqrt( (pa.meansA.^2)*pa.varsS + pa.varsA*(pa.meansS.^2) + pa.varsA*pa.varsS + (pa.meansA*pa.meansS).^2 ...
            + 2.*pa.meanb.*(pa.meansA*pa.meansS) + pa.meanb^2 + pa.varb );
    end
    Z = pa.Z;
    lambdaZ = lambda(Z);
    sigmaZ  = sigma(Z);
    
    Cij  =  - 0.5 * (X .* (meansA*meansS + meanb) ) + 0.5 .* Z - log(sigmaZ) - lambdaZ ...
        .* ( (meansA.^2)*varsS + varsA*(meansS.^2) + varsA*varsS + (meansA*meansS).^2 ...
        + 2 .* meanb .* (meansA*meansS) + meanb^2 + varb - Z.^2 );
    Cik  = 0.5 .* bsxfun(@rdivide, (meansA.^2 + varsA), va) - 0.5 .* log(bsxfun(@rdivide, varsA, va)) - 0.5;
    Ckj  = 0.5 .* bsxfun(@rdivide, (meansS.^2 + varsS), vs) - 0.5 .* log(bsxfun(@rdivide, varsS, vs)) - 0.5;
    Cb   = 0.5 * (meanb^2 + varb) / vb - 0.5 * log(varb / vb) - 0.5;
    C    = sum(sum(Cij)) + sum(sum(Cik)) + sum(sum(Ckj)) + Cb;
else
    varsA    = pa.varsA;
    varsS    = pa.varsS;
    meansA   = pa.meansA;
    meansS   = pa.meansS;
    vs       = pa.vs;
    va       = pa.va;
    
    if compute_Z
        pa.Z = sqrt( (pa.meansA.^2)*pa.varsS + pa.varsA*(pa.meansS.^2) + pa.varsA*pa.varsS + (pa.meansA*pa.meansS).^2);
    end
    Z = pa.Z;
    lambdaZ = lambda(Z);
    sigmaZ  = sigma(Z);
    
    Cij  =  - 0.5 * (X .* (meansA*meansS) ) + 0.5 .* Z - log(sigmaZ) - lambdaZ ...
        .* ( (meansA.^2)*varsS + varsA*(meansS.^2) + varsA*varsS + (meansA*meansS).^2 ...
        - Z.^2 );
    Cik  = 0.5 .* bsxfun(@rdivide, (meansA.^2 + varsA), va) - 0.5 .* log(bsxfun(@rdivide, varsA, va)) - 0.5;
    Ckj  = 0.5 .* bsxfun(@rdivide, (meansS.^2 + varsS), vs) - 0.5 .* log(bsxfun(@rdivide, varsS, vs)) - 0.5;
    C    = sum(sum(Cij)) + sum(sum(Cik)) + sum(sum(Ckj));
end

end

function res = sigma(Z)
res = 1 ./ (1 + exp(-Z));
end

function res = lambda(Z)
res = (0.5 - sigma(Z)) ./ (2 .* Z);
end
