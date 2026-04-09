from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import List, Dict, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import random
import traceback
from fastapi.responses import JSONResponse
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from data_loader import prepare_training_data
from clustering import GMMClusterer
from model_manager import RegimeModelManager

app = FastAPI(title="Cluster-Adaptive Forecasting Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
N_REGIMES = 4
FEATURE_DIM = 3
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

clusterer = GMMClusterer(k=N_REGIMES)
manager = RegimeModelManager(n_regimes=N_REGIMES, feature_dim=FEATURE_DIM)

SIMULATION_RESULTS: List[Dict] = []
SIMULATION_METRICS: Dict = {}
PROCESSING_STATUS = "idle"
PROCESSING_ERROR_DETAIL = ""

class RunRequest(BaseModel):
    train_dataset: str = "train"
    test_dataset: str = "metro"
    window_size: int = 10
    val_split: float = 0.15
    max_train_rows: Optional[int] = Field(None)
    max_test_rows: Optional[int] = Field(None)

class StatusResponse(BaseModel):
    status: str
    error_detail: Optional[str] = None
    steps_available: int = 0
    validation_metrics: Optional[Dict] = None

class SimulationDataPoint(BaseModel):
    time_step: int
    actual_value: float
    monolithic_prediction: float
    adaptive_prediction: float

class SimulationDataResponse(BaseModel):
    data: List[SimulationDataPoint]
    total_points: int

def calculate_final_regime_metrics(results: List[Dict]) -> Dict:
    regime_errors = [[] for _ in range(N_REGIMES)]
    mono_errors = [[] for _ in range(N_REGIMES)]
    counts = [0] * N_REGIMES
    for step in results:
        regime = step["detected_regime"]
        if 0 <= regime < N_REGIMES:
            regime_errors[regime].append(step["adaptive_error"])
            mono_errors[regime].append(step["monolithic_error"])
            counts[regime] += 1
    final_metrics = {"adaptive_mae_per_regime": [], "monolithic_mae_per_regime": [], "samples_per_regime": counts}
    for i in range(N_REGIMES):
        ada_mae = np.mean(regime_errors[i]) if regime_errors[i] else 0.0
        mono_mae = np.mean(mono_errors[i]) if mono_errors[i] else 0.0
        final_metrics["adaptive_mae_per_regime"].append(float(ada_mae))
        final_metrics["monolithic_mae_per_regime"].append(float(mono_mae))
    return final_metrics

def save_time_series_plot(results: List[Dict], filename: str = "timeseries_plot.png"):
    if not results: return
    print(f"[Plot] Generating {filename}...")
    plt.figure(figsize=(12, 4))
    steps = [r["time_step"] for r in results]
    actual = [r["true_value"] for r in results]
    plt.plot(steps, actual, label='Actual Value', color='#21808D', linewidth=1.5)
    plt.title('Time Series of Actual Values (Test Set Sample)')
    plt.xlabel('Time Step (Sampled)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()
    print(f"[Plot] Saved {filename}")

def save_cluster_plot(results: List[Dict], centroids: np.ndarray, filename: str = "cluster_plot.png"):
    if not results: return
    print(f"[Plot] Generating {filename}...")
    plt.figure(figsize=(8, 6))
    features = np.array([r["features"] for r in results]).astype('float32')
    regimes = np.array([r["detected_regime"] for r in results])
    x_feat = features[:, 0]; y_feat = features[:, 2]
    colors = ['#2563EB', '#D97706', "#169A46", '#DC2626']
    for i in range(N_REGIMES):
        mask = (regimes == i)
        if np.any(mask):
            plt.scatter(x_feat[mask], y_feat[mask], label=f'Regime {i}', color=colors[i], alpha=0.6, s=15)
    if centroids is not None and centroids.shape[0] == N_REGIMES:
         for i in range(N_REGIMES):
             if centroids.shape[1] > 2:
                 plt.scatter(centroids[i, 0], centroids[i, 2], marker='X', s=150, color=colors[i], edgecolor='black', linewidth=1.5, label=f'Centroid {i}')
             else:
                 plt.scatter(centroids[i, 0], centroids[i, 1], marker='X', s=150, color=colors[i], edgecolor='black', linewidth=1.5, label=f'Centroid {i}')
    plt.title('Regime Clustering (Feature Space: Roll Mean vs Trend)')
    plt.xlabel('Rolling Mean'); plt.ylabel('Trend')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename)); plt.close(); print(f"[Plot] Saved {filename}")

def save_error_comparison_plot(final_metrics: Dict, filename: str = "error_comparison_plot.png"):
    if not final_metrics: return
    print(f"[Plot] Generating {filename}...")
    adaptive_maes = final_metrics["adaptive_mae_per_regime"]; mono_maes = final_metrics["monolithic_mae_per_regime"]
    regime_labels = [f'Regime {i}' for i in range(N_REGIMES)]; x = np.arange(N_REGIMES); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, adaptive_maes, width, label='Adaptive MAE', color='#21808D')
    rects2 = ax.bar(x + width/2, mono_maes, width, label='Monolithic MAE', color='grey', alpha=0.7)
    ax.set_ylabel('Mean Absolute Error (MAE)'); ax.set_title('MAE Comparison by Regime (Test Set Sample)')
    ax.set_xticks(x); ax.set_xticklabels(regime_labels); ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.2f'); ax.bar_label(rects2, padding=3, fmt='%.2f')
    fig.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, filename)); plt.close(); print(f"[Plot] Saved {filename}")


def run_evaluation(X: np.ndarray, y: np.ndarray) -> Dict:
    print(f"Running evaluation on {len(X)} samples...")
    start_time = time.time(); adaptive_preds = []; mono_preds = []
    if not manager.global_initialized: print("Warning: Global model not initialized during evaluation.")
    for i in range(len(X)):
        x = X[i]; regime = clusterer.assign(x); y_adaptive = manager.predict(regime, x)
        y_mono = manager.predict(-1, x) if manager.global_initialized else y_adaptive
        adaptive_preds.append(y_adaptive); mono_preds.append(y_mono)
    y_true = y
    adaptive_mae = mean_absolute_error(y_true, adaptive_preds) if len(adaptive_preds) > 0 else 0.0
    monolithic_mae = mean_absolute_error(y_true, mono_preds) if len(mono_preds) > 0 else 0.0
    adaptive_rmse = np.sqrt(mean_squared_error(y_true, adaptive_preds)) if len(adaptive_preds) > 0 else 0.0
    monolithic_rmse = np.sqrt(mean_squared_error(y_true, mono_preds)) if len(mono_preds) > 0 else 0.0
    improvement_mae = monolithic_mae - adaptive_mae
    metrics = {"adaptive_mae": float(adaptive_mae),"adaptive_rmse": float(adaptive_rmse),"monolithic_mae": float(monolithic_mae),"monolithic_rmse": float(monolithic_rmse),"improvement_mae": float(improvement_mae)}
    end_time = time.time(); print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
    return metrics

def _run_processing_task(req: RunRequest):
    global SIMULATION_RESULTS, SIMULATION_METRICS, clusterer, manager, PROCESSING_STATUS, PROCESSING_ERROR_DETAIL
    try:
        start_time_total = time.time(); print("[Task] Starting background processing..."); PROCESSING_STATUS = "running"; PROCESSING_ERROR_DETAIL = ""; SIMULATION_RESULTS = []; SIMULATION_METRICS = {}
        start_time = time.time(); print(f"[Task] Loading training data from: {req.train_dataset} (max_rows={req.max_train_rows})...")
        X_full_train, y_full_train, meta_train = prepare_training_data(name=req.train_dataset, window=req.window_size, max_rows=req.max_train_rows)
        print(f"[Task] Loaded {len(X_full_train)} training samples in {time.time() - start_time:.2f}s.")
        start_time = time.time(); print(f"[Task] Loading full test data from: {req.test_dataset} (will sample later)...")
        X_test_full, y_test_full, meta_test = prepare_training_data(name=req.test_dataset, window=req.window_size, max_rows=None)
        print(f"[Task] Loaded {len(X_test_full)} potential test samples in {time.time() - start_time:.2f}s.")
        meta = meta_train; meta['features'] = ["roll_mean", "roll_var", "trend"]
        print("[Task] Splitting training data & sampling test data..."); n_train_samples = len(X_full_train); val_idx = int(n_train_samples * (1 - req.val_split))
        X_train, y_train = X_full_train[:val_idx], y_full_train[:val_idx]; X_val, y_val = X_full_train[val_idx:], y_full_train[val_idx:]
        n_test_full = len(X_test_full)
        if req.max_test_rows is not None and req.max_test_rows < n_test_full:
            sample_size = min(req.max_test_rows, n_test_full); random_indices = random.sample(range(n_test_full), sample_size)
            X_test = X_test_full[random_indices]; y_test = y_test_full[random_indices]; print(f"[Task] Using {len(X_test)} randomly sampled test rows.")
        else: X_test = X_test_full; y_test = y_test_full; print(f"[Task] Using all {len(X_test)} available test rows.")
        meta['train_dataset'] = req.train_dataset; meta['test_dataset'] = req.test_dataset; meta['n_train'] = int(len(X_train)); meta['n_val'] = int(len(X_val)); meta['n_test'] = int(len(X_test))
        if meta['n_test'] < 10 or meta['n_train'] < 10: raise ValueError("Test or train set too small after processing/sampling.")
 
        print("[Task] Initializing models..."); clusterer = GMMClusterer(k=N_REGIMES); manager = RegimeModelManager(n_regimes=N_REGIMES, feature_dim=FEATURE_DIM)
        start_time = time.time(); print("[Task] Training global model..."); manager.train_global(X_train, y_train); print(f"[Task] Global model training took {time.time() - start_time:.2f}s.")
        start_time = time.time(); print("[Task] Training clusterer..."); clusterer.fit_batch(X_train); print(f"[Task] Clusterer training took {time.time() - start_time:.2f}s.")
        if clusterer.is_fitted:
            print("[Task] Assigning training data & training regime models..."); start_time_assign = time.time(); train_assignments = clusterer.model.predict(X_train); print(f"[Task] Assignment took {time.time() - start_time_assign:.2f}s.")
            start_time_regime = time.time(); manager.train_regime_models_batch(X_train, y_train, train_assignments); print(f"[Task] Regime model training took {time.time() - start_time_regime:.2f}s.")
        else: print("WARNING: Clusterer failed to fit..."); manager.regime_initialized = [False] * N_REGIMES
        manager.save_models()

        print("[Task] Validating models..."); SIMULATION_METRICS = run_evaluation(X_val, y_val); print(f"[Task] Validation Metrics: {SIMULATION_METRICS}")

        print(f"[Task] Processing test set ({len(X_test)} steps)..."); start_time_sim = time.time(); temp_results = []
        for i in range(len(X_test)):
            x = X_test[i]; y_true = y_test[i]; regime = clusterer.assign(x); y_adaptive = manager.predict(regime, x); y_mono = manager.predict(-1, x)
            step_data = {"time_step": int(i), "features": x.tolist(), "true_value": float(y_true), "detected_regime": int(regime),"prediction_adaptive": float(y_adaptive), "prediction_monolithic": float(y_mono), "adaptive_error": float(abs(y_true - y_adaptive)), "monolithic_error": float(abs(y_true - y_mono))}
            temp_results.append(step_data)
            manager.partial_update(regime, x, y_true)
            if (i + 1) % 1000 == 0: elapsed = time.time() - start_time_sim; rate = (i + 1) / elapsed if elapsed > 0 else 0; print(f"  [Task] Processed step {i+1}/{len(X_test)} ({rate:.0f} steps/sec)")
        
        print("[Task] Generating and saving plots..."); final_regime_metrics = calculate_final_regime_metrics(temp_results)
        save_time_series_plot(temp_results); save_cluster_plot(temp_results, clusterer.centroids); save_error_comparison_plot(final_regime_metrics); print("[Task] Plots saved.")
        
        SIMULATION_RESULTS = temp_results; print(f"[Task] Test set simulation completed in {time.time() - start_time_sim:.2f} seconds."); print(f"[Task] Total processing time: {time.time() - start_time_total:.2f} seconds.")
        PROCESSING_STATUS = "complete"
    except Exception as e: print(f"!!! ERROR during background processing: {e}"); print(traceback.format_exc()); PROCESSING_STATUS = "error"; PROCESSING_ERROR_DETAIL = str(e)


@app.post("/run_dataset_processing")
async def start_dataset_processing(req: RunRequest, background_tasks: BackgroundTasks):
    """
    Starts the long data processing task (using GMM + LGBM) in the background
    and returns an estimated time.
    """
    try:
        print(f"[DEBUG] Received request with data: {req.dict()}")
    except Exception as e:
        print(f"[DEBUG] Received invalid request: {e}")
        raise HTTPException(status_code=400, detail="Invalid request body. Could not parse.")

    global PROCESSING_STATUS, PROCESSING_ERROR_DETAIL
    if PROCESSING_STATUS == "running":
        print("[DEBUG] Rejected: Processing is already in progress.")
        raise HTTPException(status_code=400, detail="Processing is already in progress.")

    train_row_factor = 1.0 / 100000; test_row_factor = 0.5 / 100000; base_time = 10
    est_train_time = (req.max_train_rows * train_row_factor) if req.max_train_rows else 60
    est_test_time = (req.max_test_rows * test_row_factor) if req.max_test_rows else 10
    estimated_seconds = max(15, int(base_time + est_train_time + est_test_time))

    PROCESSING_STATUS = "idle"; PROCESSING_ERROR_DETAIL = ""
    background_tasks.add_task(_run_processing_task, req)
    print(f"Added processing task to background. Estimated time: {estimated_seconds} seconds.")
    return {"status": "processing_started", "estimated_seconds": estimated_seconds}

@app.get("/processing_status", response_model=StatusResponse)
async def get_processing_status():
    """Poll this endpoint to check if the background task is done."""
    return StatusResponse(
        status=PROCESSING_STATUS,
        error_detail=PROCESSING_ERROR_DETAIL,
        steps_available=len(SIMULATION_RESULTS),
        validation_metrics=SIMULATION_METRICS if PROCESSING_STATUS == "complete" else None
    )

@app.get("/get_simulation_step/{step_id}")
def get_simulation_step(step_id: int):
    if PROCESSING_STATUS != "complete": raise HTTPException(status_code=400, detail=f"Processing not complete. Status: {PROCESSING_STATUS}")
    if not SIMULATION_RESULTS or step_id < 0 or step_id >= len(SIMULATION_RESULTS): raise HTTPException(status_code=404, detail="Step not found.")
    return SIMULATION_RESULTS[step_id]

@app.get("/get_simulation_data", response_model=SimulationDataResponse)
async def get_simulation_data():
    """
    Returns the core simulation data (actual vs predictions) for plotting.
    """
    if PROCESSING_STATUS != "complete": raise HTTPException(status_code=400, detail=f"Processing not complete. Status: {PROCESSING_STATUS}")
    if not SIMULATION_RESULTS: raise HTTPException(status_code=404, detail="No simulation results available.")
    data_points = [SimulationDataPoint(time_step=step["time_step"], actual_value=step["true_value"], monolithic_prediction=step["prediction_monolithic"], adaptive_prediction=step["prediction_adaptive"]) for step in SIMULATION_RESULTS]
    response_data = SimulationDataResponse(data=data_points, total_points=len(data_points))
    return response_data

@app.get("/status")
def get_current_status():
    """
    Returns current cluster centroids (static GMM) and model status.
    """
    centroids_list = clusterer.centroids.tolist() if clusterer and hasattr(clusterer, 'centroids') and clusterer.centroids is not None and clusterer.centroids.size > 0 else []
    return {
        "n_regimes": N_REGIMES,
        "centroids": centroids_list,
        "regime_initialized": manager.regime_initialized if manager else [False]*N_REGIMES,
        "global_initialized": manager.global_initialized if manager else False,
        "simulation_steps_available": len(SIMULATION_RESULTS),
        "validation_metrics": SIMULATION_METRICS
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)