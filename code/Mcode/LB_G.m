function [C] = LB_G(pa, X)

    if nargin < 2
        X = pa.X;
    end
    Xseen = ones(size(X));
    
    Cij  =  ((X - pa.meansA*pa.meansS).^2 +  (pa.meansA.^2)*pa.varsS ...
        + pa.varsA*(pa.meansS.^2) + pa.varsA*pa.varsS )./(2*pa.vx) + 0.5*log(2*pi*pa.vx);
    Cik  = bsxfun(@rdivide, (pa.meansA.^2 + pa.varsA), 2.*pa.va) - log(bsxfun(@rdivide, pa.varsA, pa.va)) ./ 2 - 0.5;
    Ckj  = bsxfun(@rdivide, (pa.meansS.^2 + pa.varsS), 2.*pa.vs) - log(bsxfun(@rdivide, pa.varsS, pa.vs)) ./ 2 - 0.5;
    C    = SS(Cij.*Xseen) + SS(Cik) + SS(Ckj); 
end