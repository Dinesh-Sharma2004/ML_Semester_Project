# Regime-Aware Time Series Forecasting System

A regime-aware forecasting system for large-scale telemetry streams built with Python, LightGBM, Scikit-learn, FastAPI, and a lightweight frontend dashboard.

The project models non-stationary sensor behavior by first detecting latent operating regimes with a Gaussian Mixture Model (GMM), then routing each sample to a specialized LightGBM regressor. A global baseline model is kept for comparison, and the backend reports MAE/RMSE along with regime-wise error analysis.

## Highlights

- Processes large telemetry datasets with memory-aware loading and `float32` downcasting.
- Engineers rolling temporal features for next-step forecasting.
- Uses a 4-regime GMM for unsupervised regime detection.
- Trains 1 global LightGBM model plus 4 regime-specific LightGBM models.
- Supports online adaptation with buffered partial model updates.
- Exposes asynchronous FastAPI endpoints for training, evaluation, simulation, and monitoring.
- Includes a frontend dashboard for adaptive vs monolithic model comparison.

## ML Architecture

### 1. Data ingestion

The backend loads raw telemetry CSVs from `backend/data/` and:

- selects valid sensor columns,
- parses and sorts timestamps,
- drops invalid rows,
- downcasts sensor values to `float32` for lower memory usage.

### 2. Feature engineering

For each time step, the pipeline computes:

- `roll_mean`
- `roll_var`
- `trend`

The target is the next-step aggregated sensor signal, making this a supervised next-step forecasting problem.

### 3. Regime detection

A Gaussian Mixture Model clusters the engineered feature space into 4 latent operating regimes. Each input sample is assigned to the most likely regime before prediction.

### 4. Forecasting

The system trains:

- 1 global LightGBM regressor as a monolithic baseline
- 4 regime-specific LightGBM regressors for adaptive forecasting

At inference time, the backend compares:

- adaptive prediction from the detected regime model
- monolithic prediction from the global baseline

### 5. Online adaptation

During simulation, each regime maintains a rolling sample buffer and periodically refreshes its LightGBM model using recent observations to better handle drift and changing operating conditions.

## Tech Stack

- Python
- FastAPI
- LightGBM
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Vanilla HTML/CSS/JavaScript frontend

## Repository Structure

```text
.
├── backend/
│   ├── main.py
│   ├── data_loader.py
│   ├── clustering.py
│   ├── model_manager.py
│   ├── models/
│   ├── plots/
│   └── data/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── requirements.txt
```

## Datasets

This repository does not include the raw telemetry CSV files because they are too large for GitHub. The backend expects datasets such as:

- `backend/data/dataset_train.csv`
- `backend/data/MetroPT2.csv`

A dataset link reference is available in `backend/data/dataset_links.txt`.

Based on the local project files used during development, the data footprint is approximately:

- `dataset_train.csv`: ~10.77M rows
- `MetroPT2.csv`: ~7.12M rows
- 21 columns per dataset

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Dinesh-Sharma2004/ML_Semester_Project.git
cd ML_Semester_Project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add datasets

Place the required CSV files inside:

```text
backend/data/
```

### 5. Run the backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Run the frontend

Open `frontend/index.html` in a browser, or serve it with a simple static server.

Example:

```bash
cd frontend
python -m http.server 5500
```

Then open:

```text
http://localhost:5500
```

## API Overview

### `POST /run_dataset_processing`

Starts background training, validation, and simulation.

Example payload:

```json
{
  "train_dataset": "train",
  "test_dataset": "metro",
  "window_size": 10,
  "val_split": 0.15,
  "max_train_rows": 20000,
  "max_test_rows": 500
}
```

### `GET /processing_status`

Returns current background-processing status and validation metrics when available.

### `GET /get_simulation_step/{step_id}`

Returns prediction details for a single simulation step.

### `GET /get_simulation_data`

Returns actual values and adaptive vs monolithic predictions for plotting.

### `GET /status`

Returns current regime centroids, model initialization state, and validation metrics.

## Evaluation

The backend computes:

- MAE
- RMSE
- regime-wise adaptive MAE
- regime-wise monolithic MAE

It also generates:

- time-series plots
- regime-cluster plots
- error-comparison plots

## Hugging Face Deployment

This project is a good fit for a Hugging Face Spaces demo by hosting:

- the frontend as the user-facing dashboard
- the backend logic through a lightweight inference wrapper

Recommended deployment options:

- **Gradio Space** if you want to convert the current dashboard into a single Python app
- **Docker Space** if you want to preserve the FastAPI backend plus static frontend structure

For production-like Spaces deployment:

- exclude raw training datasets from the repository
- load sample demo data or lightweight cached artifacts instead
- keep trained model artifacts small enough for repository limits
- expose only the simulation/inference path in the public demo

## Resume-Friendly Summary

Built an end-to-end machine learning pipeline for regime-aware forecasting on large-scale telemetry data using GMM-based latent state detection and LightGBM-based adaptive regression. Engineered temporal features, implemented online model adaptation, and deployed an API-driven simulation workflow with per-regime MAE/RMSE monitoring.

## Future Improvements

- Add automated tests for the backend pipeline
- Log experiment configurations and metrics
- Persist run metadata for reproducibility
- Add a Dockerfile for one-command deployment
- Create a Hugging Face Spaces-ready demo entrypoint

## Author

Dinesh Sharma

- GitHub: https://github.com/Dinesh-Sharma2004
