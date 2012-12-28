function expSetup(dset, iterNo, I, J, K)

% SEED
% RandStream.setDefaultStream(RandStream('mt19937ar', 'Seed', 1e6*iterNo));
RandStream.setDefaultStream(RandStream('mt19937ar', 'Seed', 100*sum(clock)));
addpath(genpath('~/work/MatlabToolboxes/stats'))

Nunseen = 1;

% Create/Load Data - large matrix make in blocks
if (dset == 1)
    Kt = 5;
    Nblocks = 100;
    Ibl = I / Nblocks;
    Xs = [];
    Xfulls = [];
    S = randn(Kt,J);
    for iblocks = 1:Nblocks
        A = randn(Ibl,Kt);
        b = -6;
        X = A*S + b;
        X = rand(size(X)) < sigma(X) + 0;
        % Leave out Nunseen of the +1s in each row, rank the -1s, calc prec@M.
        Xfull = X;
        for i = 1:Ibl
            ind_ones = find(Xfull(i,:)==1);
            % Check have more +1s than Nunseen.
            delete_ones = randsample(ind_ones, min(Nunseen,length(ind_ones)));
            X(i,delete_ones) = 0;
        end
        Xs_tmp = sparse(X);
        Xfulls_tmp = sparse(Xfull);
        Xs = [Xs; Xs_tmp];
        Xfulls = [Xfulls; Xfulls_tmp];
    end
elseif (dset == 3)
    load('~/work/binMatFac/code/original/data/sparseMatrix.txt')
    [I J] = size(sparseMatrix);
    Xs = sparse(sparseMatrix);
    Xfulls = Xs;
    for i = 1:I
        ind_ones = find(Xfulls(i,:)==1);
        % Check have more +1s than Nunseen.
        delete_ones = randsample(ind_ones, min(Nunseen,length(ind_ones)));
        Xs(i,delete_ones) = 0;
    end
elseif (dset == 4)
    load('~/work/binMatFac/code/original/data/T10I4D100K.mat')
    [I J] = size(sparseMatrix);
    X = (sparseMatrix == 1) + 0;
    Xs = sparse(X);
    Xfulls = Xs;
    for i = 1:I
        ind_ones = find(Xfulls(i,:)==1);
        % Check have more +1s than Nunseen.
        delete_ones = randsample(ind_ones, min(Nunseen,length(ind_ones)));
        X(i,delete_ones) = 0;
    end
    Xs = sparse(X);
elseif (dset == 5)
    load('~/work/binMatFac/code/original/data/retail.mat')
    [I J] = size(sparseMatrix); % 9157 x 1599 (min 3 purch per customer)
    Xfulls = sparseMatrix;
    X = (full(sparseMatrix) == 1) + 0;
    for i = 1:I
        ind_ones = find(Xfulls(i,:)==1);
        % Check have more +1s than Nunseen.
        delete_ones = randsample(ind_ones, min(Nunseen,length(ind_ones)));
        X(i,delete_ones) = 0;
    end
    Xs = sparse(X);
elseif (dset == 6)
    load('~/work/binMatFac/code/original/data/wikiVote.mat')
    [I J] = size(sparseMatrix);  % 4800 x 4800 (min 2 links per node)
    Xfulls = sparseMatrix;
    X = (full(sparseMatrix) == 1) + 0;
    for i = 1:I
        ind_ones = find(Xfulls(i,:)==1);
        % Check have more +1s than Nunseen.
        delete_ones = randsample(ind_ones, min(Nunseen,length(ind_ones)));
        X(i,delete_ones) = 0;
    end
    Xs = sparse(X);
elseif (dset == 7)
    load('~/work/binMatFac/code/original/data/likes_data_I73572_J21187.mat')
    [I J] = size(Xs);
    Xfulls = Xs;
    for i = 1:I
        if mod(i, 1000) == 0
            disp(i)
        end
        ind_ones = find(Xfulls(i,:)==1);
        % Check have more +1s than Nunseen.
        delete_ones = randsample(ind_ones, min(Nunseen,length(ind_ones)));
        Xs(i,delete_ones) = 0;
    end
end
fprintf('Data has dimensions: %g, %g\n', size(Xs,1), size(Xs,2))

% save matrix and parameters.
save('../data/XSparse.mat', 'Xs', 'Xfulls', 'I', 'J', 'Nunseen', ...
    'dset', 'iterNo')

% save X in format for C++ code.
list = find(Xs==1);
XSparse(:, 1) = mod(list-1, I);
XSparse(:, 2) = floor((list-1)/I);
dlmwrite('../data/XSparse.txt', XSparse, 'delimiter', ',')

 % Initialise mean parameters using SVD so that it is the same for all methods.
initialiseSVDsparse(Xs, K);

end


