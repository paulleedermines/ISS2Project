%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MARKOV CHAIN PROJECT CHECKPOINT - FINAL FORECAST
%%% Train on Hawaiian training data, then use the Symbol Machine.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Like the first- and second-order examples in markovChainCheckpointF26,
% this model counts the symbols that follow each sequence of past symbols.
% Here we use orders 0 through 15. We start with a zeroth-order forecast,
% then update it using successively longer histories.
%
% checkpoint_validation chose order 15 and smoothing scale 32 using
% only a held-out part of the TRAINING sequence.

%% SET UP THE MATLAB PATH

% Find the original Symbol Machine functions and data in the parent folder.
% Restore the caller's path when this script finishes.
checkpointRoot = fileparts(mfilename('fullpath'));
previousPath = path;
pathCleanup = onCleanup(@() path(previousPath));
addpath(fileparts(checkpointRoot));
rng(0);

%% LOAD THE TRAINING DATA AND CHOOSE THE MODEL SETTINGS

data = load('sequence_Hawaiian_train.mat', 'sequence');
sequence = double(data.sequence(:)); % Use a column vector of training symbols
sequenceLength = length(sequence);
maxOrder = 15;
scale = 32;

%% TRAIN THE ZEROTH-ORDER MARKOV MODEL

% Count how often each symbol appears. Add half a count to each symbol
% so that every symbol has a positive probability.
symbolCounts = zeros(1,9);
for symbol = 1:9
    symbolCounts(symbol) = sum(sequence == symbol);
end
estimatedPMF = (symbolCounts + 0.5) / (sum(symbolCounts) + 4.5);

%% TRAIN THE HIGHER-ORDER MARKOV MODELS

% Each row holds the nine counts for one history of past symbols.
% A history is stored as a base-9 number: the most recent symbol is in
% the ones place, the next is in the nines place, and so on. Subtract 1
% from each symbol to get digits 0 through 8, then add 1 for MATLAB rows.
%
% A cell array keeps one count matrix per order. Sparse matrices store
% only nonzero counts because most long histories never occur.
countTables = cell(maxOrder,1);
historyRows = zeros(sequenceLength,1);
for order = 1:maxOrder
    historyRows(order+1:end) = historyRows(order+1:end) + ...
        (sequence(1:end-order)-1) * 9^(order-1);
    % Repeated row/column pairs are added together by sparse.
    countTables{order} = sparse(historyRows(order+1:end)+1, ...
        sequence(order+1:end), ...
        1, 9^order, 9);
end

%% FORECAST THE TESTING DATA WITH THE SYMBOL MACHINE

% The test sequence is revealed one symbol at a time by the machine.
% The first forecast uses estimatedPMF because no past symbols are known.
sequenceLength = initializeSymbolMachineF26('sequence_Hawaiian_test.mat',0);
pastSymbols = zeros(sequenceLength,1);
for ii = 1:sequenceLength
    probs = estimatedPMF;
    historyRow = 0;
    for order = 1:min(maxOrder,ii-1)
        historyRow = historyRow + (pastSymbols(ii-order)-1) * 9^(order-1);
        symbolCounts = full(countTables{order}(historyRow+1,:));

        % Mix the counts for this history with the lower-order forecast.
        % More smoothing keeps the forecast closer to the lower order.
        distinctSymbols = sum(symbolCounts > 0);
        alpha = scale * max(1,distinctSymbols);
        probs = (symbolCounts + alpha * probs) / (sum(symbolCounts) + alpha);
    end
    [symbol,penalty] = symbolMachineF26(probs);
    pastSymbols(ii) = symbol;
end
reportSymbolMachineF26;

%% RESTORE THE MATLAB PATH

clear pathCleanup;
