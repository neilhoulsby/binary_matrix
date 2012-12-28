function [pa] = VBMF_Nak(Xs, K, optimise_prior)

% notation
% Nak = nakajima

% VBMFNAKAJIMA: Gaussian analytic solution.
%   Assumes 1. spherical likelihood, 2. zero mean, const variance priors

% two probs, 1 a factor of two seems to be mising from the preds
%            2 the one of the means is going very large - not being
%            regularised, although may not effect preds

% order of increasing cost n=ones,val; or n=1s,ans; n=all,val; n=all,abs

% assume WLOG I < J also n.b. in pa{1}per B = A and A = S.

% CHECK INPUTTING 0/1s:

[Ltemp Mtemp] = size(Xs);         % L = I, M = J
if Ltemp > Mtemp
    Xst = Xs';
else
    Xst = Xs;
end
[L M] = size(Xst); 
n = 1;

% prior hyperparams
ca2 = 1 * ones(K,1);                           % va
cs2 = 1 * ones(K,1);                           % vs

[U S V] = svds(Xst, K);    % all the computation is here - for spa{1}rse replace with svds
gammas = diag(S(1:K,1:K));
U = U(:,1:K);
V = V(:,1:K);
residuals = Xst - U*S(1:K,1:K)*V';
s2 = var(residuals(:));            % vx N.B. can re-estimate

% learn prior parameters?
if optimise_prior
    c2 = (gammas.^2 - (L + M)*s2/n + sqrt((gammas.^2  - (L + M)*s2/n).^2 - 4*L*M*s2^2/n^2))...
        / (2 * L * M);
    gammaUnderline = (sqrt(L)+sqrt(M)) * sqrt(s2) / sqrt(n);
    c2(gammas < gammaUnderline) = 0;
    cs2 = c2 ./ ca2;
end


% coefficients
etas2 = ( 1 - (s2*L)./(n*gammas.^2) ) .* ( 1 - (s2*M)./(n*gammas.^2) ) .* gammas.^2; 
cfs3 =  (L-M)^2 *gammas / (L*M);
cfs2 =  - ( cfs3.*gammas + (L^2+M^2)*etas2 ./ (L*M) + 2*s2^2 ./ (n^2*ca2.*cs2) );
cfs0 = ( etas2 - s2^2 ./ (n^2*ca2.*cfs2) ).^2;
cfs1 =  cfs3 .* sqrt(cfs0);
gammaHats = zeros(K,1);
for k = 1:K
    solus = roots([1 cfs3(k) cfs2(k) cfs1(k) cfs0(k)]);
    % dont want imag solutions - numerics can mean v small imaginary bit.
    solus(abs(imag(solus))>1e-3) = -1e10;   
    solus = real(solus);
    solus = sort(solus, 'descend');
    gammaHats(k)  = solus(2);
end
gammaTils = sqrt( (L+M)*s2/(2*n) + (s2^2)./(2*n^2*ca2.*cs2) + ...
    sqrt( ((L+M)*s2/(2*n) + (s2^2)./(2*n^2*ca2.*cs2)).^2 - L*M*s2^2 / n^2 ) );

gammaHatsVB = gammaHats;
gammaHatsVB(gammas <= gammaTils) = 0;
if optimise_prior
    Dels = M * log(n*gammas.*gammaHatsVB / (M*s2) + 1) + ...
        L * log(n*gammas.*gammaHatsVB / (L*s2) + 1) + ...
        (n / s2) * (-2*gammas.*gammaHatsVB + L*M*c2);
    gammaHatsVB(gammas <= gammaUnderline) = 0;  % Corollary 5  .* (Dels > 0)
    gammaHatsVB(Dels > 0) = 0;                  % Corollary 5
end

% threshold eta for this bit
etas2 = etas2 .* (gammas > gammaTils) + ( s2./(n*sqrt(ca2).*sqrt(cs2)) ) .* (gammas <= gammaTils);
deltas = ( n*(M-L)*(gammas - gammaHats) ...
    + sqrt( n^2 * (M-L)^2 * (gammas-gammaHatsVB).^2 + 4*s2^2*L*M./(ca2.*cs2) )  ) ...
    ./ (2*s2*M*ca2.^(-1));
meansA = bsxfun(@times, sqrt(gammaHatsVB.*deltas.^-1)', U);
meansS = bsxfun(@times, sqrt(gammaHatsVB.*deltas), V');
varsA = repmat( ( -(n*etas2 + s2*(M-L)) ...
    + sqrt( (n*etas2 + s2*(M-L)).^2 + 4*L*n*s2*etas2 ) )' ...
    ./ (2*n*L*( gammaHatsVB.*deltas.^(-1) + n^(-1)*s2*ca2.^(-1) ) )', [L 1]);
varsS = repmat( (( -(n*etas2 - s2*(M-L)) ...
    + sqrt( (n*etas2 - s2*(M-L)).^2 + 4*M*n*s2*etas2 ) ) ...
    ./ (2*n*M*( gammaHatsVB.*deltas.^(-1) + n^(-1)*s2*cs2.^(-1) )) ), [1 M]);
pa{1}.gammaHatsVB = gammaHatsVB;

% (AS)' = S'A'
% pa{1}.P = U*diag(gammaHats)*V';
if Ltemp > Mtemp 
   pa{1}.meansA = meansS';
   pa{1}.meansS = meansA';
   pa{1}.varsA = varsS';
   pa{1}.varsS = varsA';
else
    pa{1}.meansA = meansA;
    pa{1}.meansS = meansS;
    pa{1}.varsA = varsA;
    pa{1}.varsS = varsS;       
end
pa{1}.X = Xs;
pa{1}.Niters = 1;
pa{1}.vx = s2;
pa{1}.va = ca2';
pa{1}.vs = cs2;

end



