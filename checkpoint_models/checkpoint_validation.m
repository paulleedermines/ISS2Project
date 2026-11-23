function results = checkpoint_validation()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MARKOV CHAIN PROJECT CHECKPOINT - MODEL VALIDATION
%%% Compare Markov orders and smoothing using TRAINING data only.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use the first 80% of the Hawaiian training sequence to learn symbol
% counts. Use the last 20% to compare forecasts for symbols that were not
% used to learn those counts. This last part is our validation sequence.
% The testing sequence is not loaded or used to choose the model.
%
% The output results records every score and the best model settings.

%% SET UP THE MATLAB PATH AND WORKING FOLDER

root = fileparts(mfilename('fullpath'));
oldFolder = pwd;
oldPath = path;
cleanupFolder = onCleanup(@() cd(oldFolder));
cleanupPath = onCleanup(@() path(oldPath));
addpath(fileparts(root));
cd(root);

%% LOAD AND SPLIT THE TRAINING DATA

data = load('sequence_Hawaiian_train.mat', 'sequence');
sequence = double(data.sequence(:));
split = floor(0.8 * length(sequence));
train = sequence(1:split);
validation = sequence(split+1:end);
trainLength = length(train);
validationLength = length(validation);
fprintf('Training prefix: %d; validation suffix: %d symbols.\n', ...
    trainLength, validationLength);

%% CHOOSE THE MODEL SETTINGS TO COMPARE

% For every order, try two ways of mixing in a lower-order forecast:
%   1) Scaled Witten-Bell-style smoothing, with several scales.
%   2) Absolute discounting, with several choices of discount.
% Each column of scores will correspond to one of these choices.
maxOrder = 15;
scales = [0.125 0.25 0.5 1 2 4 8 16 32 64];
discounts = [0.1 0.25 0.5 0.75 0.9];
numberOfCandidates = length(scales) + length(discounts);

%% TRAIN THE ZEROTH-ORDER MARKOV MODEL

% Add half a count to each symbol before normalizing to a pmf.
symbolCounts = zeros(1,9);
for symbol = 1:9
    symbolCounts(symbol) = sum(train == symbol);
end
estimatedPMF = (symbolCounts + 0.5) / (sum(symbolCounts) + 4.5);

% To compute the penalty, we only need the probability assigned to the
% symbol that actually occurred. Start every candidate at order zero.
correctSymbolProbs = zeros(validationLength,numberOfCandidates);
for ii = 1:validationLength
    correctSymbolProbs(ii,:) = estimatedPMF(validation(ii));
end

%% TRAIN AND VALIDATE EACH HIGHER-ORDER MODEL

% Store each history as a base-9 number, with the most recent symbol in
% the ones place. Subtract 1 to use digits 0 through 8; add 1 for a row.
% Sparse count matrices avoid storing rows for histories never observed.
trainHistoryRows = zeros(trainLength,1);
validationHistoryRows = zeros(validationLength,1);
scores = zeros(maxOrder,numberOfCandidates);
for order = 1:maxOrder
    % Add one older symbol to each history, then count its next symbols.
    trainHistoryRows(order+1:end) = trainHistoryRows(order+1:end) + ...
        (train(1:end-order)-1) * 9^(order-1);
    validationHistoryRows(order+1:end) = validationHistoryRows(order+1:end) + ...
        (validation(1:end-order)-1) * 9^(order-1);
    symbolCounts = sparse(trainHistoryRows(order+1:end)+1, ...
        train(order+1:end), ...
        1, 9^order, 9);

    % Look up training counts for each validation history. The first
    % 'order' symbols have too little history, so keep their earlier pmfs.
    localCounts = zeros(validationLength,9);
    localCounts(order+1:end,:) = ...
        full(symbolCounts(validationHistoryRows(order+1:end)+1,:));
    totals = sum(localCounts, 2);
    distinct = sum(localCounts > 0, 2);
    observedCounts = zeros(validationLength,1);
    for ii = 1:validationLength
        observedCounts(ii) = localCounts(ii,validation(ii));
    end

    % [-- Scaled Witten-Bell-style smoothing --]
    % Use alpha extra counts, distributed according to the previous pmf.
    for candidate = 1:length(scales)
        alpha = scales(candidate) * max(1, distinct);
        correctSymbolProbs(:,candidate) = (observedCounts + alpha .* ...
            correctSymbolProbs(:,candidate)) ./ (totals + alpha);
    end

    % [-- Absolute discounting --]
    % Remove a fixed amount from every nonzero count, then give that
    % probability to the previous pmf. Unseen histories keep that pmf.
    for candidate = 1:length(discounts)
        column = length(scales) + candidate;
        discount = discounts(candidate);
        seen = totals > 0;
        correctSymbolProbs(seen,column) = ...
            (max(observedCounts(seen)-discount,0) + ...
            discount * distinct(seen) .* correctSymbolProbs(seen,column)) ...
            ./ totals(seen);
    end

    % The average penalty in bits per symbol is smaller for better models.
    scores(order,:) = mean(-log2(correctSymbolProbs), 1);
    fprintf('Order %2d, best validation %.6f bits/symbol\n', ...
        order, min(scores(order,:)));
end

%% SELECT AND SAVE THE BEST MODEL SETTINGS

results.scores = scores;
results.scales = scales;
results.discounts = discounts;
results.split = split;
% MATLAB searches columns first, keeping the original tie-breaking order.
[results.bestScore, index] = min(scores(:));
[results.bestOrder, column] = ind2sub(size(scores), index);
if column <= length(scales)
    results.method = 'witten-bell';
    results.parameter = scales(column);
else
    results.method = 'absolute-discount';
    results.parameter = discounts(column-length(scales));
end
fprintf('Selected: order %d, %s %.3g, validation %.6f bits/symbol.\n', ...
    results.bestOrder, results.method, results.parameter, results.bestScore);
save('checkpoint_validation.mat', 'results');
% The onCleanup objects restore the caller's folder and path on return.
end
