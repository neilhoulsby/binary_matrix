function [dC d2C] = ND_SJ(pa)
    % Calculates gradients at current point.
    perturb = 1e-5;

    % calc cost at operating point
    C =  LB_SJ(pa);
    
    % perturb one of the parameters and calc costs above and below
    pa.meansA(1,1) = pa.meansA(1,1) + perturb;
    Cu =  LB_SJ(pa);
    pa.meansA(1,1) = pa.meansA(1,1) - 2*perturb;
    Cl =  LB_SJ(pa);  
    
    % calc numerical derivative
    dC = (Cu - Cl) / (2*perturb);

    % calc numerical 2nd derivative (could be noisy)
    if nargout > 1
       dCu = (Cu - C) / perturb;
       dCl = (C - Cl) / perturb;
       d2C = (dCu - dCl) / perturb;
    end
end

