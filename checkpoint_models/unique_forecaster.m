function results = unique_forecaster()
% UNIQUE_FORECASTER  Categorical periodic-clock forecasting experiment.
% Run: results = unique_forecaster;
% No toolbox is required. Original Symbol Machine files are unmodified.
%
% Model: estimate P(symbol=k | index modulo P). No previous symbol values
% enter a prediction. This is a categorical seasonal density, rather than
% a transition table indexed by recent symbols or a linear lag regression.
% It assumes the test file continues the training file's sample-index clock.
% The files supply no timestamps, so this is an assumption, not metadata.
%
% Select period and Dirichlet strength using only the final 30% of TRAIN.
% Refit on all TRAIN; issue each TEST probability before calling the Symbol
% Machine. All baselines use the same complete train/test files and score.

root = fileparts(mfilename('fullpath'));
oldFolder = pwd;
oldPath = path;
cleanupFolder = onCleanup(@() cd(oldFolder)); %#ok<NASGU>
cleanupPath = onCleanup(@() path(oldPath)); %#ok<NASGU>
addpath(fileparts(root));
cd(root);
rng(0, 'twister');
trainFile = 'sequence_ElecDemand_train.mat';
testFile = 'sequence_ElecDemand_test.mat';
loaded = load(trainFile, 'sequence');
train = double(loaded.sequence(:));
assert(all(ismember(train, 1:9)), 'Training symbols must be in 1:9.');
n = numel(train);
cut = floor(0.7*n);
periods = 2:min(100, floor(cut/3));
strengths = [0.1 1 3 10 30 100];
floorMix = 0.01;
validation = zeros(numel(periods)*numel(strengths), 3);
row = 0;
bestLoss = Inf;
for period = periods
    for strength = strengths
        template = fitClock(train(1:cut), period, strength, floorMix);
        loss = scoreClock(template, train(cut+1:end), cut);
        row = row+1;
        validation(row,:) = [period strength loss];
        if loss < bestLoss
            bestLoss = loss;
            bestPeriod = period;
            bestStrength = strength;
        end
    end
end
selection = array2table(validation, 'VariableNames', ...
    {'Period','PriorStrength','ValidationBitsPerSymbol'});
selection = sortrows(selection, 'ValidationBitsPerSymbol');
writetable(selection, 'unique_forecaster_validation.csv');
fprintf('TRAIN ONLY: %d fit / %d validation; period=%d, strength=%g, validation=%.9f bits/symbol.\n', ...
    cut, n-cut, bestPeriod, bestStrength, bestLoss);

template = fitClock(train, bestPeriod, bestStrength, floorMix);
mcCounts = ones(9,9,9); % assignment's add-one MC(2)
for t = 3:n
    mcCounts(train(t-2),train(t-1),train(t)) = ...
        mcCounts(train(t-2),train(t-1),train(t))+1;
end
mc = mcCounts ./ sum(mcCounts,3);
ar2 = fitAR(train,2);
ar4 = fitAR(train,4);

models = {'Uniform','MC2_add_one','AR2_conditional_OLS','AR4_conditional_OLS','Categorical_clock'};
bps = zeros(numel(models),1);
totals = zeros(numel(models),1);
lengths = zeros(numel(models),1);
losses = cell(numel(models),1);
for model = 1:numel(models)
    count = initializeSymbolMachineF26(testFile,0);
    history = zeros(count,1);
    loss = zeros(count,1);
    for t = 1:count
        switch model
            case 1
                probabilities = ones(1,9)/9;
            case 2
                if t <= 2
                    probabilities = ones(1,9)/9;
                else
                    probabilities = reshape(mc(history(t-2),history(t-1),:),1,9);
                end
            case {3,4}
                if model == 3, ar = ar2; else, ar = ar4; end
                if t <= ar.order
                    probabilities = ones(1,9)/9;
                else
                    x = [1; history(t-1:-1:t-ar.order)];
                    mu = ar.beta'*x;
                    logWeights = -((1:9)-mu).^2/(2*ar.variance);
                    weights = exp(logWeights-max(logWeights));
                    probabilities = weights/sum(weights);
                end
            case 5
                phase = mod(n+t-1,bestPeriod)+1;
                probabilities = template(phase,:);
        end
        assert(all(isfinite(probabilities)) && all(probabilities>0) && ...
            abs(sum(probabilities)-1)<1e-10, 'Invalid forecast.');
        [observed, loss(t)] = symbolMachineF26(probabilities);
        history(t) = double(observed); % revealed only AFTER forecast
    end
    fprintf('\n%s\n', models{model});
    reportSymbolMachineF26;
    bps(model) = mean(loss);
    totals(model) = sum(loss);
    lengths(model) = count;
    losses{model} = loss;
end
results = table(string(models(:)),lengths,totals,bps, ...
    'VariableNames',{'Model','TestSymbols','TotalBits','BitsPerSymbol'});
writetable(results, 'unique_forecaster_results.csv');
disp(results);
% Paired chronological blocks describe variability; they are not iid CIs.
blockSize = 50;
blocks = floor(lengths(end)/blockSize);
blockResults = zeros(blocks,4);
for b = 1:blocks
    ix = (b-1)*blockSize+(1:blockSize);
    blockResults(b,:) = [ix(1) ix(end) ...
        mean(losses{2}(ix)-losses{5}(ix)) ...
        mean(losses{3}(ix)-losses{5}(ix))];
end
writetable(array2table(blockResults,'VariableNames', ...
    {'FirstSymbol','LastSymbol','MC2MinusClock','AR2MinusClock'}), ...
    'unique_forecaster_blocks.csv');
% Store exact selected model and fitting details for an auditable rerun.
save('unique_forecaster_model.mat','template','bestPeriod','bestStrength', ...
    'floorMix','cut','trainFile','testFile','ar2','ar4','results','losses');
end

function template = fitClock(sequence,period,strength,floorMix)
counts = zeros(period,9);
for t = 1:numel(sequence)
    phase = mod(t-1,period)+1;
    counts(phase,sequence(t)) = counts(phase,sequence(t))+1;
end
globalCounts = accumarray(sequence,1,[9 1])';
prior = (globalCounts+1)/(numel(sequence)+9);
template = (counts+strength*prior)./(sum(counts,2)+strength);
template = (1-floorMix)*template + floorMix/9;
end

function loss = scoreClock(template,sequence,offset)
period = size(template,1);
phases = mod(offset+(0:numel(sequence)-1)',period)+1;
probabilities = template(sub2ind(size(template),phases,sequence));
loss = mean(-log2(probabilities));
end

function model = fitAR(sequence,order)
% Conditional Gaussian maximum likelihood = least squares after warmup.
% This is a transparent AR baseline, not MATLAB arima's exact likelihood.
n = numel(sequence);
x = ones(n-order,order+1);
for lag = 1:order
    x(:,lag+1) = sequence(order+1-lag:n-lag);
end
y = sequence(order+1:n);
model.beta = x\y;
model.variance = max(mean((y-x*model.beta).^2),1e-6);
model.order = order;
end
