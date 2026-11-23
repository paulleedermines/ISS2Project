% Train a variable-order Markov model on the Hawaiian training set and
% forecast the test set through the Symbol Machine. Hyperparameters were
% chosen using the held-out training suffix in checkpoint_validation.
checkpointRoot = fileparts(mfilename('fullpath'));
previousPath = path;
pathCleanup = onCleanup(@() path(previousPath));
addpath(fileparts(checkpointRoot));
rng(0);
data = load('sequence_Hawaiian_train.mat', 'sequence');
train = double(data.sequence(:));
maxOrder = 15;
scale = 32;
baseCounts = accumarray(train, 1, [9 1])';
base = (baseCounts + 0.5) / (sum(baseCounts) + 4.5);
tables = cell(maxOrder, 1);
keys = zeros(numel(train), 1);
for order = 1:maxOrder
    keys(order+1:end) = keys(order+1:end) + ...
        (train(1:end-order)-1) * 9^(order-1);
    tables{order} = sparse(keys(order+1:end)+1, train(order+1:end), ...
        1, 9^order, 9);
end
n = initializeSymbolMachineF26('sequence_Hawaiian_test.mat', 0);
history = zeros(n, 1);
for t = 1:n
    probs = base;
    key = 0;
    for order = 1:min(maxOrder,t-1)
        key = key + (history(t-order)-1) * 9^(order-1);
        counts = full(tables{order}(key+1,:));
        alpha = scale * max(1, sum(counts > 0));
        probs = (counts + alpha * probs) / (sum(counts) + alpha);
    end
    [history(t), ~] = symbolMachineF26(probs);
end
reportSymbolMachineF26;
clear pathCleanup;
