function expSetupArt2(dset, iterNo, I, J)
% different evaluation for artificial data.
% dset = 2 always.
% Generates two matrices from the same parameters.
% Evaluates with test LL.


% SEED
% RandStream.setDefaultStream(RandStream('mt19937ar', 'Seed', 1e6*iterNo));
RandStream.setDefaultStream(RandStream('mt19937ar', 'Seed', 100*sum(clock)));
addpath(genpath('~/work/MatlabToolboxes/stats'))

% Create/Load Data - large matrix make in blocks

dset = 2;
K = 5;
Nblocks = 10;
Ibl = I / Nblocks;
Xs = [];
Xs_test = [];
S = randn(K,J);
for iblocks = 1:Nblocks
    A = randn(Ibl,K);
    b = -6;
    P = A*S + b;
    X = rand(size(P)) < sigma(P) + 0;
    Xtest = rand(size(P)) < sigma(P) + 0;
    
    Xs_tmp = sparse(X);
    Xs = [Xs; Xs_tmp];
    Xs_test_tmp = sparse(Xtest);
    Xs_test = [Xs_test; Xs_test_tmp];
end
fprintf('Data has dimensions: %g, %g\n', size(Xs,1), size(Xs,2))

% save matrix and parameters.
save('../data/XSparse.mat', 'Xs', 'Xs_test', 'I', 'J', ...
    'dset', 'iterNo')

% save X in format for C++ code.
list = find(Xs==1);
XSparse(:, 1) = mod(list-1, I);
XSparse(:, 2) = floor((list-1)/I);
dlmwrite('../data/XSparse.txt', XSparse, 'delimiter', ',')

 % Initialise mean parameters using SVD so that it is the same for all methods.
initialiseSVDsparse(Xs, K);

end


