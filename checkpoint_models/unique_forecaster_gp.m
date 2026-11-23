function results = unique_forecaster_gp()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% GAUSSIAN PROCESS FORECASTING CHECKPOINT
%%% Train on electricity demand data, then compare five forecasts.
%%% Uses the original Symbol Machine F26 and the provided datasets.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A Gaussian process (GP) models how values at different TIMES are related.
% Nearby times can have similar values, and a periodic pattern can repeat.
% Each forecast uses all symbols revealed so far through these time
% relationships. We do not fit a finite list of symbol lags or a table of
% symbol transitions. The nonperiodic kernel has no finite Markov order.
%
% We choose the GP settings using TRAINING data only. When testing begins,
% we start with an empty history. Each forecast goes to the Symbol Machine
% BEFORE it reveals the next test symbol. Revealed symbols update the GP
% forecast, but they never change the selected kernel settings.

%% SETUP AND TRAINING DATA

% Find the original datasets and Symbol Machine in the parent folder.
% Restore the caller's folder and MATLAB path when this function finishes.
root = fileparts(mfilename('fullpath'));
oldFolder = pwd;
oldPath = path;
cleanupFolder = onCleanup(@() cd(oldFolder));
cleanupPath = onCleanup(@() path(oldPath));
addpath(fileparts(root));
cd(root);
rng(0,'twister');

trainFile = 'sequence_ElecDemand_train.mat';
testFile = 'sequence_ElecDemand_test.mat';
loaded = load(trainFile,'sequence'); % OK because this is TRAINING data
trainingSequence = double(loaded.sequence(:));

%% TRAIN THE GAUSSIAN PROCESS

% Use the first 70% of the training sequence to estimate the mean and
% variance. Use the remaining 30% to compare possible kernel settings.
% Keep the order in time: earlier symbols come before later symbols.
cut = floor(0.7*numel(trainingSequence));
fitSequence = trainingSequence(1:cut);
validationSequence = trainingSequence(cut+1:end);

% This list of candidate settings was fixed before examining GP test results.
% lengthscale controls how quickly nearby times lose their relationship.
% noiseRatio adds independent noise, period sets the repeating time scale,
% and periodicWeight balances the nearby-time and repeating components.
lengthscales = [1 2 4 8];
noiseRatios = [0.01 0.03 0.1 0.3];
periods = [12 24 48];
periodicWeights = [0 0.5 0.9];
numberOfSettings = numel(lengthscales)*numel(noiseRatios)* ...
    numel(periods)*numel(periodicWeights);
selection = zeros(numberOfSettings,5);
row = 0;
bestLoss = Inf;

for lengthscale = lengthscales
    for noiseRatio = noiseRatios
        for period = periods
            for periodicWeight = periodicWeights
                config = struct('lengthscale',lengthscale, ...
                    'noiseRatio',noiseRatio,'period',period, ...
                    'periodicWeight',periodicWeight, ...
                    'mean',mean(fitSequence), ...
                    'signalVariance',var(fitSequence,1));

                % Validation starts with an empty history, just as testing will.
                loss = validationLoss(validationSequence,config);
                row = row + 1;
                selection(row,:) = [lengthscale noiseRatio period ...
                    periodicWeight loss];
                if loss < bestLoss
                    bestLoss = loss;
                    bestConfig = config;
                end
            end
        end
    end
end

selection = array2table(selection,'VariableNames', ...
    {'Lengthscale','NoiseRatio','Period','PeriodicWeight','ValidationBitsPerSymbol'});
writetable(sortrows(selection,'ValidationBitsPerSymbol'), ...
    'unique_forecaster_gp_validation.csv');
fprintf('GP training-only selection: %.9f validation bits/symbol\n',bestLoss);
disp(bestConfig);

% Keep the selected kernel settings. Now estimate the mean and variance
% again using ALL training symbols before moving to the testing dataset.
bestConfig.mean = mean(trainingSequence);
bestConfig.signalVariance = var(trainingSequence,1);
checkSequentialGP(trainingSequence(1:30),bestConfig);

%% TRAIN THE COMPARISON MODELS

% Match the AR checkpoint: estimate second- and fourth-order models with
% the Econometrics Toolbox. All model fitting happens before testing.
assert(exist('arima','file')==2, ...
    'Econometrics Toolbox is required for assignment AR baselines.');
ar2 = estimate(arima(2,0,0),trainingSequence,'Display','off');
ar4 = estimate(arima(4,0,0),trainingSequence,'Display','off');

% Match the second-order Markov checkpoint. Count each possible next
% symbol after each pair of past symbols. Starting at 1 prevents zeros.
symbolCounts = ones(9,9,9);
for ii = 3:numel(trainingSequence)
    past2 = trainingSequence(ii-2);
    past1 = trainingSequence(ii-1);
    currentSymbol = trainingSequence(ii);
    symbolCounts(past2,past1,currentSymbol) = ...
        symbolCounts(past2,past1,currentSymbol) + 1;
end
probMatrix = symbolCounts./sum(symbolCounts,3);
% For each (past2,past1) pair, the nine next-symbol probabilities sum to 1.

% Save each model's penalties for the final comparison table and plots.
models = {'Uniform','MC2_add_one','AR2_arima','AR4_arima','Gaussian_process'};
totals = zeros(numel(models),1);
bps = zeros(numel(models),1);
lengths = zeros(numel(models),1);
losses = cell(numel(models),1);

%% UNIFORM FORECAST

% This baseline assigns probability 1/9 to every symbol, at every step.
sequenceLength = initializeSymbolMachineF26(testFile,0);
probs = ones(1,9)/9;
penalties = zeros(sequenceLength,1);
for ii = 1:sequenceLength
    checkProbabilities(probs);
    [~,penalty] = symbolMachineF26(probs); % The uniform forecast needs no history.
    penalties(ii) = penalty;
end
fprintf('\n%s\n',models{1});
reportSymbolMachineF26;
totals(1) = sum(penalties);
bps(1) = mean(penalties);
lengths(1) = sequenceLength;
losses{1} = penalties;

%% SECOND-ORDER MARKOV CHAIN FORECAST

% Start a fresh test run. The first two symbols receive uniform forecasts
% because we have not yet seen the pair needed by a second-order model.
sequenceLength = initializeSymbolMachineF26(testFile,0);
history = zeros(sequenceLength,1);
penalties = zeros(sequenceLength,1);
for ii = 1:sequenceLength
    if ii <= 2
        probs = ones(1,9)/9;
    else
        past2 = history(ii-2);
        past1 = history(ii-1);
        probs = reshape(probMatrix(past2,past1,:),1,9);
    end
    checkProbabilities(probs);
    [symbol,penalty] = symbolMachineF26(probs);
    history(ii) = double(symbol);
    penalties(ii) = penalty;
end
fprintf('\n%s\n',models{2});
reportSymbolMachineF26;
totals(2) = sum(penalties);
bps(2) = mean(penalties);
lengths(2) = sequenceLength;
losses{2} = penalties;

%% SECOND-ORDER AUTOREGRESSIVE FORECAST

% As in the AR checkpoint, forecast from the last two revealed symbols.
% Turn the Gaussian prediction into nine probabilities that sum to 1.
sequenceLength = initializeSymbolMachineF26(testFile,0);
history = zeros(sequenceLength,1);
penalties = zeros(sequenceLength,1);
for ii = 1:sequenceLength
    if ii <= 2
        probs = ones(1,9)/9;
    else
        [predictedSym,~,V] = forecast(ar2,1,history(ii-2:ii-1));
        probs = gaussianPMF(predictedSym,V);
    end
    checkProbabilities(probs);
    [symbol,penalty] = symbolMachineF26(probs);
    history(ii) = double(symbol);
    penalties(ii) = penalty;
end
fprintf('\n%s\n',models{3});
reportSymbolMachineF26;
totals(3) = sum(penalties);
bps(3) = mean(penalties);
lengths(3) = sequenceLength;
losses{3} = penalties;

%% FOURTH-ORDER AUTOREGRESSIVE FORECAST

% The fourth-order model needs four revealed symbols before it can use
% forecast(). Give uniform forecasts until that history is available.
sequenceLength = initializeSymbolMachineF26(testFile,0);
history = zeros(sequenceLength,1);
penalties = zeros(sequenceLength,1);
for ii = 1:sequenceLength
    if ii <= 4
        probs = ones(1,9)/9;
    else
        [predictedSym,~,V] = forecast(ar4,1,history(ii-4:ii-1));
        probs = gaussianPMF(predictedSym,V);
    end
    checkProbabilities(probs);
    [symbol,penalty] = symbolMachineF26(probs);
    history(ii) = double(symbol);
    penalties(ii) = penalty;
end
fprintf('\n%s\n',models{4});
reportSymbolMachineF26;
totals(4) = sum(penalties);
bps(4) = mean(penalties);
lengths(4) = sequenceLength;
losses{4} = penalties;

%% GAUSSIAN PROCESS FORECAST

% Start with no test history. L describes relationships between TIMES,
% so we can compute it before any test symbols are revealed. Its leading
% block does not depend on future times; the training-data check verifies this.
sequenceLength = initializeSymbolMachineF26(testFile,0);
history = zeros(sequenceLength,1);
penalties = zeros(sequenceLength,1);
L = covarianceFactor(sequenceLength,bestConfig);
innovations = zeros(sequenceLength,1);

for ii = 1:sequenceLength
    % Use only the prediction errors of symbols already revealed.
    % For ii = 1, this sum is empty and predictedSym is the training mean.
    predictedSym = bestConfig.mean + L(ii,1:ii-1)*innovations(1:ii-1);
    V = L(ii,ii)^2;
    probs = gaussianPMF(predictedSym,V);
    checkProbabilities(probs);
    [symbol,penalty] = symbolMachineF26(probs);
    history(ii) = double(symbol);
    penalties(ii) = penalty;

    % Only after revealing this symbol can we use its normalized error
    % (called an innovation) to help forecast later symbols.
    innovations(ii) = (history(ii)-predictedSym)/L(ii,ii);
end
fprintf('\n%s\n',models{5});
reportSymbolMachineF26;
totals(5) = sum(penalties);
bps(5) = mean(penalties);
lengths(5) = sequenceLength;
losses{5} = penalties;

%% COMPARE AND SAVE RESULTS

% Lower bits per symbol means a better probability forecast.
results = table(string(models(:)),lengths,totals,bps, ...
    'VariableNames',{'Model','TestSymbols','TotalBits','BitsPerSymbol'});
writetable(results,'unique_forecaster_gp_results.csv');
disp(results);

% Compare average penalties over consecutive blocks of 50 test symbols.
% Positive differences mean that the GP used fewer bits than that baseline.
blockSize = 50;
blockCount = floor(sequenceLength/blockSize);
blockResults = zeros(blockCount,5);
for block = 1:blockCount
    indices = (block-1)*blockSize + (1:blockSize);
    blockResults(block,:) = [indices(1) indices(end) ...
        mean(losses{2}(indices)-losses{5}(indices)) ...
        mean(losses{3}(indices)-losses{5}(indices)) ...
        mean(losses{4}(indices)-losses{5}(indices))];
end
writetable(array2table(blockResults,'VariableNames', ...
    {'FirstSymbol','LastSymbol','MC2MinusGP','AR2MinusGP','AR4MinusGP'}), ...
    'unique_forecaster_gp_blocks.csv');
save('unique_forecaster_gp_model.mat','bestConfig','cut','selection', ...
    'results','losses','ar2','ar4');

figure('Visible','off','Color','w');
tiledlayout(1,2,'TileSpacing','compact');
nexttile;
bar(bps);
grid on;
xticklabels({'Uniform','MC(2)','AR(2)','AR(4)','GP'});
ylabel('Bits per symbol (lower is better)');
title('ElecDemand: all 500 held-out symbols');
nexttile;
hold on;
for model = 2:5
    plot(cumsum(losses{model})./(1:sequenceLength)','LineWidth',1.5);
end
grid on;
xlabel('Revealed test symbols');
ylabel('Cumulative bits per symbol');
legend({'MC(2)','AR(2)','AR(4)','GP'},'Location','northeast');
title('Causal forecasts, reset test history');
exportgraphics(gcf,'unique_forecaster_gp_comparison.png','Resolution',160);
close(gcf);
end

%% HELPER FUNCTIONS

function loss = validationLoss(sequence,config)
% Try one set of GP settings on the training-only validation sequence.
% Begin with no history, forecast, then reveal each validation symbol.
L = covarianceFactor(numel(sequence),config);
innovations = zeros(numel(sequence),1);
loss = 0;
for ii = 1:numel(sequence)
    predictedSym = config.mean + L(ii,1:ii-1)*innovations(1:ii-1);
    probs = gaussianPMF(predictedSym,L(ii,ii)^2);
    loss = loss - log2(probs(sequence(ii)));
    innovations(ii) = (sequence(ii)-predictedSym)/L(ii,ii);
end
loss = loss/numel(sequence);
end

function L = covarianceFactor(sequenceLength,config)
% Build the GP covariance matrix K from distances between input times.
% The squared-exponential term relates nearby times. The periodic term
% relates times separated by a repeating pattern. The diagonal adds noise.
distance = (1:sequenceLength)' - (1:sequenceLength);
rbf = exp(-0.5*(distance/config.lengthscale).^2);
periodic = exp(-2*sin(pi*distance/config.period).^2);
K = config.signalVariance*((1-config.periodicWeight)*rbf + ...
    config.periodicWeight*periodic + config.noiseRatio*eye(sequenceLength));

% Cholesky factorization gives K = L*L'. Because L is lower triangular,
% row ii can use earlier prediction errors without using future symbols.
L = chol(K,'lower');
end

function probs = gaussianPMF(predictedSym,V)
% Sample the Gaussian at symbols 1 through 9, then normalize the samples.
% This gives the same normalized shape as normpdf(1:9,predictedSym,sqrt(V)).
% Subtracting the largest log weight avoids numerical underflow. The tiny
% numeric floor prevents zeros and is shared by the GP and AR forecasts.
logWeights = -((1:9)-predictedSym).^2/(2*max(V,1e-10));
weights = exp(logWeights-max(logWeights));
probs = max(weights,realmin);
probs = probs/sum(probs);
end

function checkProbabilities(probs)
% Check that each forecast is a valid, positive probability vector.
assert(all(isfinite(probs)) && all(probs>0) && ...
    abs(sum(probs)-1)<1e-10,'Invalid PMF.');
end

function checkSequentialGP(sequence,config)
% Check the efficient prediction-error calculation against the direct
% Gaussian conditional mean and variance on a short training sequence.
sequenceLength = numel(sequence);
L = covarianceFactor(sequenceLength,config);
K = L*L';
innovations = zeros(sequenceLength,1);
largestError = 0;

for ii = 1:sequenceLength
    sequentialMean = config.mean + L(ii,1:ii-1)*innovations(1:ii-1);
    sequentialVariance = L(ii,ii)^2;
    if ii == 1
        directMean = config.mean;
        directVariance = K(1,1);
    else
        past = 1:ii-1;
        directMean = config.mean + ...
            K(ii,past)*(K(past,past)\(sequence(past)-config.mean));
        directVariance = K(ii,ii) - ...
            K(ii,past)*(K(past,past)\K(past,ii));
    end
    largestError = max([largestError abs(sequentialMean-directMean) ...
        abs(sequentialVariance-directVariance)]);
    innovations(ii) = (sequence(ii)-sequentialMean)/L(ii,ii);
end
assert(largestError<1e-9, ...
    'Sequential GP failed independent posterior check.');

% Adding future input times must not change the earlier covariance factor.
% This confirms that precomputing L does not use future observed symbols.
prefix = covarianceFactor(15,config);
assert(max(abs(prefix-L(1:15,1:15)),[],'all')<1e-10, ...
    'Future covariance entries changed the prefix forecast.');
fprintf('GP causal/direct-posterior check passed; max error %.3g.\n',largestError);
end
