function results = checkpoint_validation()
% Select a variable-order Markov predictor using TRAINING data only.
% No test sequence is loaded or examined by this experiment.
root = fileparts(mfilename('fullpath'));
oldFolder = pwd;
oldPath = path;
cleanupFolder = onCleanup(@() cd(oldFolder)); %#ok<NASGU>
cleanupPath = onCleanup(@() path(oldPath)); %#ok<NASGU>
addpath(fileparts(root));
cd(root);
data = load('sequence_Hawaiian_train.mat', 'sequence');
sequence = double(data.sequence(:));
split = floor(0.8 * numel(sequence));
train = sequence(1:split);
validation = sequence(split+1:end);
maxOrder = 15;
scales = [0.125 0.25 0.5 1 2 4 8 16 32 64];
discounts = [0.1 0.25 0.5 0.75 0.9];
fprintf('Training prefix: %d; validation suffix: %d symbols.\n', ...
    numel(train), numel(validation));
baseCounts = accumarray(train, 1, [9 1])';
base = (baseCounts + 0.5) / (sum(baseCounts) + 4.5);
trueIndices = sub2ind([numel(validation) 9], ...
    (1:numel(validation))', validation);
trueProb = repmat(base(validation)', 1, numel(scales) + numel(discounts));
trainKeys = zeros(numel(train), 1);
validationKeys = zeros(numel(validation), 1);
scores = zeros(maxOrder, numel(scales) + numel(discounts));
for order = 1:maxOrder
    trainKeys(order+1:end) = trainKeys(order+1:end) + ...
        (train(1:end-order)-1) * 9^(order-1);
    validationKeys(order+1:end) = validationKeys(order+1:end) + ...
        (validation(1:end-order)-1) * 9^(order-1);
    counts = sparse(trainKeys(order+1:end)+1, train(order+1:end), ...
        1, 9^order, 9);
    localCounts = zeros(numel(validation), 9);
    localCounts(order+1:end, :) = full(counts(validationKeys(order+1:end)+1,:));
    totals = sum(localCounts, 2);
    distinct = sum(localCounts > 0, 2);
    observedCounts = localCounts(trueIndices);
    for candidate = 1:numel(scales)
        alpha = scales(candidate) * max(1, distinct);
        trueProb(:,candidate) = (observedCounts + alpha .* ...
            trueProb(:,candidate)) ./ (totals + alpha);
    end
    for candidate = 1:numel(discounts)
        column = numel(scales) + candidate;
        discount = discounts(candidate);
        seen = totals > 0;
        trueProb(seen,column) = (max(observedCounts(seen)-discount,0) + ...
            discount * distinct(seen) .* trueProb(seen,column)) ./ totals(seen);
    end
    scores(order,:) = mean(-log2(trueProb), 1);
    fprintf('Order %2d, best validation %.6f bits/symbol\n', ...
        order, min(scores(order,:)));
end
results.scores = scores;
results.scales = scales;
results.discounts = discounts;
results.split = split;
[results.bestScore, index] = min(scores(:));
[results.bestOrder, column] = ind2sub(size(scores), index);
if column <= numel(scales)
    results.method = 'witten-bell';
    results.parameter = scales(column);
else
    results.method = 'absolute-discount';
    results.parameter = discounts(column-numel(scales));
end
fprintf('Selected: order %d, %s %.3g, validation %.6f bits/symbol.\n', ...
    results.bestOrder, results.method, results.parameter, results.bestScore);
save('checkpoint_validation.mat', 'results');
end


