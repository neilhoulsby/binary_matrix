function [dC d2C] = ND_G(pa)
    % Calculates gradients at current point.
    perturb = 1e-6;
  
    % calc cost at operating point
    C =  LB_G(pa);
    
    % perturb one of the parameters and calc costs above and below
    pa.vx(1,1) = pa.vx(1,1) + perturb;
    Cu =  LB_G(pa);
    pa.vx(1,1) = pa.vx(1,1) - 2*perturb;
    Cl =  LB_G(pa);  
    
    % calc numerical derivative
    dC = (Cu - Cl) / (2*perturb);

    % calc numerical 2nd derivative (could be noisy)
    if nargout > 1
       dCu = (Cu - C) / perturb;
       dCl = (C - Cl) / perturb;
       d2C = (dCu - dCl) / perturb;
    end
end

