# Forecasting checkpoint

This folder contains two forecasting experiments: a refined Markov chain for
Hawaiian text and a Gaussian-process model for electricity demand. Both use
the original Symbol Machine and datasets in the parent `ISS2Project` folder.
Keep this folder inside `ISS2Project`; the scripts locate those files there.

## Requirements and setup

- MATLAB. These scripts were tested with MATLAB R2026a.
- Econometrics Toolbox for the Gaussian-process experiment's AR comparison
  models. The refined Markov and clock experiments do not require it.
- The parent folder must contain the provided `sequence_*.mat` files and
  `initializeSymbolMachineF26.m`, `symbolMachineF26.m`, and
  `reportSymbolMachineF26.m`.

In MATLAB, set the current folder to `ISS2Project`, then run:

```matlab
cd('checkpoint_models')
```

Run the commands below from that folder. Lower penalty, measured in bits per
symbol, means better probabilistic forecasts.

## Refined Markov model

Run the Hawaiian test forecast:

```matlab
checkpoint_final_markov
```

The script trains on `sequence_Hawaiian_train.mat`, forecasts every symbol in
`sequence_Hawaiian_test.mat`, and prints the Symbol Machine report. The
expected penalty is **1.4672 bits/symbol** for 5,654 test symbols. The saved
report is `checkpoint_markov_report.txt`; running the script prints a report
but does not overwrite that text file.

The model uses context orders up to 15 and a smoothing scale of 32. To repeat
the training-only search that selected these settings, run:

```matlab
validation = checkpoint_validation;
```

The search uses the first 80% of the training sequence for fitting and the
remaining 20% for validation. It prints the selected settings and writes
`checkpoint_validation.mat`. The final Markov script has its selected settings
written into the script; running validation does not change them automatically.

## Gaussian-process experiment

Run the electricity-demand forecast and its baseline comparisons:

```matlab
results = unique_forecaster_gp;
```

The function selects its settings using a held-out portion of the training
data, then runs the Gaussian process, an add-one second-order Markov model,
and MATLAB AR(2) and AR(4) models on the complete test sequence. It prints a
Symbol Machine report for each model and returns a table. Expected test
penalties are:

| Model | Bits/symbol |
| --- | ---: |
| Gaussian process | 0.9585 |
| Markov, order 2 | 1.8098 |
| AR, order 2 | 1.4543 |
| AR, order 4 | 1.3387 |

The function writes or replaces `unique_forecaster_gp_results.csv`, the
validation and block comparison CSV files, `unique_forecaster_gp_model.mat`,
and `unique_forecaster_gp_comparison.png`. The previously captured complete
Symbol Machine output is in `unique_forecaster_gp_verified_run.log`. For the
method, comparison details, and limitations, see `unique_forecaster_report.txt`.

The Gaussian-process result is **preliminary**: an earlier model was tested
on the same electricity test set before this framework was chosen. A fresh
test dataset would be needed for independent confirmation.

## Earlier clock experiment

`unique_forecaster.m` contains an earlier categorical clock model. Run it with
`results = unique_forecaster;` to reproduce that comparison. It performed
worse than the baselines on the electricity test data and assumes the test
sequence continues the training sequence's clock phase. It writes or replaces
its own `unique_forecaster_...` results files.
