function [ pa ] = SVDMFSparse( Xs, K )
%SVDMFSparse SVD mat fac.
tic;
[U S V] = svds(Xs,K);
U = U(:,1:K);
S = S(1:K,1:K);
V = V(:,1:K);
pa.S = S;
pa.meansA = U * S;
pa.meansS = V';
pa.time = toc;
pa.Niters = 1;
end

