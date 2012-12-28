function initialiseSVDsparse(Xs, K)

[I J] = size(Xs);
if K > I || K > J; error('K too large!'); end;

% Initialise
[U S V] = svds(Xs, K);
meansA = U(:,1:K)*sqrt(S(1:K,1:K));
meansS = sqrt(S(1:K,1:K))*V(:,1:K)';
% varsA = ones(size(meansA));
% varsS = ones(size(meansS));
residuals = Xs - meansA*meansS;
vx = var(residuals(:));
sparseness = SS(Xs==1) / numel(Xs);
meanb = -log(sparseness^-1 - 1);

save('../initialisation/mA_init', 'meansA')
save('../initialisation/mS_init', 'meansS')
save('../initialisation/mb_init', 'meanb')
save('../initialisation/vx_init', 'vx')

dlmwrite('../initialisation/mA_init.txt', meansA, 'delimiter', ',')
dlmwrite('../initialisation/mS_init.txt', meansS', 'delimiter', ',')
dlmwrite('../initialisation/mb_init.txt', meanb)
dlmwrite('../initialisation/vx_init.txt', vx)

end