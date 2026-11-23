function results = unique_forecaster_gp()
% UNIQUE_FORECASTER_GP  Exact causal Gaussian-process symbol forecasting.
% A GP places a joint Gaussian distribution over values indexed by TIME.
% Its covariance is a squared-exponential plus smooth periodic kernel.
% No finite list of symbol lags or symbol-transition table is fitted.
% The RBF term has no finite-order Markov property: conditioning uses all
% revealed observations through a dense covariance factor. A GP can be
% represented using infinite past regressions, as can many stochastic
% processes; this is not a claim that conditional probability lacks history.
%
% Test history is reset, so no train/test calendar or phase alignment is
% assumed. Hyperparameters use ONLY a chronological TRAIN holdout. Test
% observations update GP conditioning, never kernel parameters. Every
% forecast is issued before its symbol is revealed by the Symbol Machine.
root = fileparts(mfilename('fullpath'));
oldFolder = pwd; oldPath = path;
cleanupFolder = onCleanup(@() cd(oldFolder)); %#ok<NASGU>
cleanupPath = onCleanup(@() path(oldPath)); %#ok<NASGU>
addpath(fileparts(root));
cd(root); rng(0,'twister');
trainFile = 'sequence_ElecDemand_train.mat';
testFile = 'sequence_ElecDemand_test.mat';
loaded = load(trainFile,'sequence');
train = double(loaded.sequence(:));
cut = floor(0.7*numel(train));
fit = train(1:cut); validation = train(cut+1:end);
% This bounded grid was fixed before examining GP test performance.
lengthscales = [1 2 4 8];
noiseRatios = [0.01 0.03 0.1 0.3];
periods = [12 24 48];
periodicWeights = [0 0.5 0.9];
selection = zeros(numel(lengthscales)*numel(noiseRatios)* ...
    numel(periods)*numel(periodicWeights),5);
row = 0; bestLoss = Inf;
for ell = lengthscales
    for noiseRatio = noiseRatios
        for period = periods
            for periodicWeight = periodicWeights
                config = struct('lengthscale',ell,'noiseRatio',noiseRatio, ...
                    'period',period,'periodicWeight',periodicWeight, ...
                    'mean',mean(fit),'signalVariance',var(fit,1));
                loss = validationLoss(validation,config);
                row = row+1;
                selection(row,:) = [ell noiseRatio period periodicWeight loss];
                if loss < bestLoss
                    bestLoss = loss; bestConfig = config;
                end
            end
        end
    end
end
selection = array2table(selection,'VariableNames', ...
    {'Lengthscale','NoiseRatio','Period','PeriodicWeight','ValidationBitsPerSymbol'});
writetable(sortrows(selection,'ValidationBitsPerSymbol'),'unique_forecaster_gp_validation.csv');
fprintf('GP training-only selection: %.9f validation bits/symbol\n',bestLoss);
disp(bestConfig);
bestConfig.mean = mean(train);
bestConfig.signalVariance = var(train,1);
checkSequentialGP(train(1:30),bestConfig);

% Econometrics Toolbox reproduces the AR assignment's estimate(arima(...))
% Gaussian forecasts, rather than substituting conditional OLS estimates.
assert(exist('arima','file')==2,'Econometrics Toolbox is required for assignment AR baselines.');
ar2 = estimate(arima(2,0,0),train,'Display','off');
ar4 = estimate(arima(4,0,0),train,'Display','off');
mcCounts = ones(9,9,9);
for t = 3:numel(train)
    mcCounts(train(t-2),train(t-1),train(t)) = ...
        mcCounts(train(t-2),train(t-1),train(t))+1;
end
mc = mcCounts./sum(mcCounts,3);
models = {'Uniform','MC2_add_one','AR2_arima','AR4_arima','Gaussian_process'};
totals = zeros(numel(models),1); bps = totals; lengths = totals;
losses = cell(numel(models),1);
for model = 1:numel(models)
    count = initializeSymbolMachineF26(testFile,0);
    history = zeros(count,1); loss = history;
    if model==5
        L = covarianceFactor(count,bestConfig);
        innovations = zeros(count,1);
    end
    for t = 1:count
        switch model
            case 1
                probabilities = ones(1,9)/9;
            case 2
                if t<=2, probabilities = ones(1,9)/9;
                else, probabilities = reshape(mc(history(t-2),history(t-1),:),1,9); end
            case {3,4}
                if model==3, order=2; ar=ar2; else, order=4; ar=ar4; end
                if t<=order
                    probabilities = ones(1,9)/9;
                else
                    % Actual assignment forecast() API; only revealed test history.
                    [mu,~,variance] = forecast(ar,1,history(t-order:t-1));
                    probabilities = gaussianPMF(mu,variance);
                end
            case 5
                mu = bestConfig.mean + L(t,1:t-1)*innovations(1:t-1);
                variance = L(t,t)^2;
                probabilities = gaussianPMF(mu,variance);
        end
        assert(all(isfinite(probabilities)) && all(probabilities>0) && ...
            abs(sum(probabilities)-1)<1e-10,'Invalid PMF.');
        [observed,loss(t)] = symbolMachineF26(probabilities);
        history(t) = double(observed);
        if model==5
            innovations(t) = (history(t)-mu)/L(t,t);
        end
    end
    fprintf('\n%s\n',models{model}); reportSymbolMachineF26;
    totals(model) = sum(loss); bps(model)=mean(loss); lengths(model)=count;
    losses{model}=loss;
end
results = table(string(models(:)),lengths,totals,bps, ...
    'VariableNames',{'Model','TestSymbols','TotalBits','BitsPerSymbol'});
writetable(results,'unique_forecaster_gp_results.csv'); disp(results);
blockSize=50; blockCount=floor(count/blockSize); blockResults=zeros(blockCount,5);
for b=1:blockCount
    ix=(b-1)*blockSize+(1:blockSize);
    blockResults(b,:)=[ix(1) ix(end) mean(losses{2}(ix)-losses{5}(ix)) ...
        mean(losses{3}(ix)-losses{5}(ix)) mean(losses{4}(ix)-losses{5}(ix))];
end
writetable(array2table(blockResults,'VariableNames', ...
    {'FirstSymbol','LastSymbol','MC2MinusGP','AR2MinusGP','AR4MinusGP'}), ...
    'unique_forecaster_gp_blocks.csv');
save('unique_forecaster_gp_model.mat','bestConfig','cut','selection', ...
    'results','losses','ar2','ar4');
figure('Visible','off','Color','w');
tiledlayout(1,2,'TileSpacing','compact');
nexttile; bar(bps); grid on;
xticklabels({'Uniform','MC(2)','AR(2)','AR(4)','GP'});
ylabel('Bits per symbol (lower is better)');
title('ElecDemand: all 500 held-out symbols');
nexttile; hold on;
for m=2:5
    plot(cumsum(losses{m})./(1:count)','LineWidth',1.5);
end
grid on; xlabel('Revealed test symbols'); ylabel('Cumulative bits per symbol');
legend({'MC(2)','AR(2)','AR(4)','GP'},'Location','northeast');
title('Causal forecasts, reset test history');
exportgraphics(gcf,'unique_forecaster_gp_comparison.png','Resolution',160);
close(gcf);
end

function checkSequentialGP(sequence,config)
% Independent direct GP posterior confirms the efficient innovations method.
n=numel(sequence); L=covarianceFactor(n,config); K=L*L';
z=zeros(n,1); largestError=0;
for t=1:n
    sequentialMean=config.mean+L(t,1:t-1)*z(1:t-1);
    sequentialVariance=L(t,t)^2;
    if t==1
        directMean=config.mean; directVariance=K(1,1);
    else
        past=1:t-1;
        directMean=config.mean+K(t,past)*(K(past,past)\(sequence(past)-config.mean));
        directVariance=K(t,t)-K(t,past)*(K(past,past)\K(past,t));
    end
    largestError=max([largestError abs(sequentialMean-directMean) ...
        abs(sequentialVariance-directVariance)]);
    z(t)=(sequence(t)-sequentialMean)/L(t,t);
end
assert(largestError<1e-9,'Sequential GP failed independent posterior check.');
% Future input times can be precomputed: Cholesky's leading block is causal.
prefix=covarianceFactor(15,config);
assert(max(abs(prefix-L(1:15,1:15)),[],'all')<1e-10, ...
    'Future covariance entries changed the prefix forecast.');
fprintf('GP causal/direct-posterior check passed; max error %.3g.\n',largestError);
end

function loss=validationLoss(sequence,config)
L=covarianceFactor(numel(sequence),config);
innovations=zeros(numel(sequence),1); loss=0;
for t=1:numel(sequence)
    mu=config.mean+L(t,1:t-1)*innovations(1:t-1);
    pmf=gaussianPMF(mu,L(t,t)^2);
    loss=loss-log2(pmf(sequence(t)));
    innovations(t)=(sequence(t)-mu)/L(t,t);
end
loss=loss/numel(sequence);
end

function L=covarianceFactor(n,config)
distance=(1:n)'-(1:n);
rbf=exp(-0.5*(distance/config.lengthscale).^2);
periodic=exp(-2*sin(pi*distance/config.period).^2);
K=config.signalVariance*((1-config.periodicWeight)*rbf + ...
    config.periodicWeight*periodic + config.noiseRatio*eye(n));
L=chol(K,'lower');
end

function pmf=gaussianPMF(mu,variance)
logWeights=-((1:9)-mu).^2/(2*max(variance,1e-10));
weights=exp(logWeights-max(logWeights));
% Numeric floor only; shared by GP and assignment AR forecasts.
pmf=max(weights,realmin); pmf=pmf/sum(pmf);
end
