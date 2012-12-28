function [pa] = VBMF_See(Xs, K, prior_learning)
% VBMFAISTATS: implementation of AISTATS paper by Bouchard.

% See = Seeger, bouchard AISTATS paper, 
% S   = spares version (mostly zeros in matrix)

% CHECK INPUTTING 0/1s:
if sum(unique(Xs))~=1 error('for Ais, matrix must be 0/1'); end

[I J] = size(Xs);
Xinds = zeros(sum(sum(Xs==1)), 2);
[Xinds(:,1) Xinds(:,2)]  = find(Xs==1);

epsilon = 1e-5;

% Prior variance
va = 1;
vs = 1;  % constant across whole matrix,
%  - these provide bounds on solu, they come together when prior
%    variance is tended to infinity

% likelihood param
[U S V] = svds(Xs ,K);    % for sparse replace with svds
U = U(:,1:K);
V = V(:,1:K);
residuals = Xs - U*S*V';
vx = var(residuals(:));        % vx

% post approx
Y = Xs;
meansA = 1e-6 * ones(I,K);
meansS = 1e-6 * ones(K,J);
tmp = zeros(length(Xinds), 1);
for xit = 1:length(Xinds)
    i = Xinds(xit, 1); j = Xinds(xit, 2); 
    tmp(xit) = meansA(i,:) * meansS(:,j);
end
P = sparse(Xinds(:,1), Xinds(:,2), tmp);
sparseOnes = sparse( Xinds(:,1), Xinds(:,2), ones(size(Xinds, 1), 1) );
if (size(P,1)~=I || size(P,2)~=J) % do slowly if have an empty row or col
   P = sparse(I,J);
   sparseOnes = sparse(I,J);
   for xit = 1:length(Xinds)
       i = Xinds(xit, 1); j = Xinds(xit, 2);
       P(i,j) = meansA(i,:) * meansS(:,j);
       sparseOnes(i,j) = 1;
   end
end
converged = false;
iter = 0;
timeSlot = 1;
tic;
while ~converged  && iter < 100
    Z = P;
    Ytil = (Z - 4 * (sparseOnes ./ (sparseOnes + exp(-Z)) - Y));
    paTemp = VBMF_Nak(Ytil, K, prior_learning);
    Pold = P;
    tmp = zeros(length(Xinds), 1);
    for xit = 1:length(Xinds)
        i = Xinds(xit, 1); j = Xinds(xit, 2);
        tmp(xit) = paTemp{1}.meansA(i,:) * paTemp{1}.meansS(:,j);
    end
    P = sparse(Xinds(:,1), Xinds(:,2), tmp);
    if (size(P,1)~=I || size(P,2)~=J) % do slowly if have an empty row or col
        P = sparse(I,J);
        for xit = 1:length(Xinds)
            i = Xinds(xit, 1); j = Xinds(xit, 2);
            P(i,j) = paTemp{1}.meansA(i,:) * paTemp{1}.meansS(:,j);
        end
    end
    
    if mean(abs(P(:)-Pold(:))) < epsilon
       converged = true; 
    end
    iter = iter+1;
    disp(mean(abs(P(:)-Pold(:))))
    if mod(iter, 5) == 0
        pa{timeSlot}.meansA = paTemp{1}.meansA;
        pa{timeSlot}.meansS = paTemp{1}.meansS;
        pa{timeSlot}.varsA = paTemp{1}.varsA;
        pa{timeSlot}.varsS = paTemp{1}.varsS;
        pa{timeSlot}.Niters = iter;
        pa{timeSlot}.time = toc;
        pa{timeSlot}.converged = converged;
        timeSlot = timeSlot + 1;
    end
end
pa{timeSlot}.meansA = paTemp{1}.meansA;
pa{timeSlot}.meansS = paTemp{1}.meansS;
pa{timeSlot}.varsA = paTemp{1}.varsA;
pa{timeSlot}.varsS = paTemp{1}.varsS;
pa{timeSlot}.Niters = iter;
pa{timeSlot}.time = toc;
pa{timeSlot}.converged = converged;

end
