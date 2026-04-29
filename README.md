# Analysis_HW7

Streamflow forecasting for HWRS564b Homework 7. This repository fits a simple historical streamflow model, generates a 5-day forecast for the Verde River, and saves validation and forecast plots in the `Outputs/` folder.

## Overview

The workflow is split across three Python scripts and one shell script:

- `train_model.py` downloads historical streamflow data, fits a selected model, and can validate the model on a set of test period data.
- `generate_forecast.py` loads the saved model and produces a 5-day forecast starting on a user-specified date.
- `forecast_functions.py` contains the shared helper functions used by both scripts.
- `run_workflow.sh` connects the two Python scripts into one workflow.

There are currently three model options:

- `longterm_avg` - one mean streamflow value for the full training period.
- `monthly_avg` - a separate mean streamflow value for each calendar month.
- `randomized` - monthly averages with random noise added to create a more variable forecast.

## Requirements

The scripts use the following third-party packages:

- `numpy`
- `pandas`
- `matplotlib`
- `hf_hydrodata`

You also need valid HydroFrame credentials because the scripts call the HydroFrame API to download streamflow data.

See Environment.yml file.

## Quick Start

1. Make sure the workflow script is executable:

	```bash
	chmod +x run_workflow.sh
	```

2. Edit `run_workflow.sh` to set the gauge ID, date range, forecast date, and model settings you want to use.

3. Run the workflow:

	```bash
	./run_workflow.sh
	```

4. Enter your HydroFrame email and PIN when prompted.

The script will train or load the model, run validation if enabled, and then generate the 5-day forecast.


## Outputs

Generated files are written to the `Outputs/` directory:

- `Outputs/validation_plot.png` - validation plot comparing observed and predicted streamflow.
- `Outputs/forecast_plot.png` - 5-day forecast plot with recent observed flow.

The trained model is saved as `saved_model.pkl` in the repository root.

## Notes

- The repository is configured around the Verde River gauge with USGS site ID `09506000`.
