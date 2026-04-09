
const API_BASE_URL = "http://localhost:8000"; 
const numRegimes = 4;
const maxBufferSize = 100;
const POLLING_INTERVAL_MS = 2000; 
    
let running = false;
let timeStep = 0;
let totalSteps = 0;
let currentRegime = 0;
let transitionCount = 0;
let dataBuffer = [];
let adaptiveErrors = [];
let monolithicErrors = [];
let regimeErrors = Array(numRegimes).fill(null).map(() => ({ adaptive: [], monolithic: [] }));
let regimeSamples = Array(numRegimes).fill(0);
let backendCentroids = [];
let processingPollInterval = null; 
let estimatedTimeRemaining = 0;
let countdownInterval = null;
let timeSeriesChart = null;
let clusterChart = null;
let errorChart = null;
let statusSpan;
let processingInfoDiv;
let startBtn;
let stopBtn;
let resetBtn;


function log(message, isError = false) {
    const logContainer = document.getElementById('logContainer');
    if (!logContainer) return; 
    const entry = document.createElement('div');
    entry.className = 'log-entry' + (isError ? ' transition' : '');
    entry.textContent = `[t=${timeStep}] ${message}`;
    logContainer.insertBefore(entry, logContainer.firstChild);
    if (logContainer.children.length > 100) logContainer.removeChild(logContainer.lastChild);
}

async function apiCall(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    try {
        const res = await fetch(url, options);
        if (!res.ok) {
            let errText = await res.text();
            try { errText = JSON.parse(errText).detail || errText; } catch(e) {}
            throw new Error(errText || `HTTP ${res.status}`);
        }
        const contentType = res.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
             return await res.json();
        } else {
             return null;
        }
    } catch (err) {
        log(`[API ERROR] ${options.method || 'GET'} ${endpoint}: ${err.message}`, true);
        throw err;
    }
}

function displayProcessingInfo(estimatedSeconds) {
    if (!processingInfoDiv) return;
    estimatedTimeRemaining = estimatedSeconds;
    processingInfoDiv.textContent = `Processing... Estimated time remaining: ${estimatedTimeRemaining} seconds`;
    processingInfoDiv.style.color = 'var(--color-info)';

    if (countdownInterval) clearInterval(countdownInterval);

    countdownInterval = setInterval(() => {
        estimatedTimeRemaining = Math.max(0, estimatedTimeRemaining - 1);
        processingInfoDiv.textContent = `Processing... Estimated time remaining: ${estimatedTimeRemaining} seconds`;
        if (estimatedTimeRemaining <= 0) {
             processingInfoDiv.textContent = `Processing... Finishing up...`;
             clearInterval(countdownInterval);
        }
    }, 1000);
}

function clearProcessingInfo(message = "", isError = false) {
     if (countdownInterval) clearInterval(countdownInterval);
     countdownInterval = null;
     if (processingInfoDiv) {
        processingInfoDiv.textContent = message;
        processingInfoDiv.style.color = isError ? 'var(--color-error)' : 'var(--color-text-secondary)';
     }
}

function pollProcessingStatus() {
    if (processingPollInterval) clearInterval(processingPollInterval);

    processingPollInterval = setInterval(async () => {
        try {
            const statusData = await apiCall('/processing_status');

            if (statusData.status === "complete") {
                clearInterval(processingPollInterval);
                processingPollInterval = null;
                clearProcessingInfo();
                log('Backend processing complete.');
                totalSteps = statusData.steps_available || 0;
                 if (statusData.validation_metrics) {
                     log(`Validation Adaptive MAE: ${statusData.validation_metrics.adaptive_mae.toFixed(4)}`);
                     log(`Validation Monolithic MAE: ${statusData.validation_metrics.monolithic_mae.toFixed(4)}`);
                 }
                running = true;
                statusSpan.textContent = '● Running';
                statusSpan.className = 'status-running';
                log('Simulation started.');
                update(); 
                startBtn.disabled = false;
                stopBtn.disabled = false;
                resetBtn.disabled = false;

            } else if (statusData.status === "error") {
                clearInterval(processingPollInterval);
                processingPollInterval = null;
                const errorMsg = statusData.error_detail || "Unknown error during processing.";
                clearProcessingInfo(`Error: ${errorMsg}`, true);
                log(`[FATAL] Backend processing failed: ${errorMsg}`, true);
                statusSpan.textContent = '● Error';
                statusSpan.className = 'status-stopped';
                 startBtn.disabled = false;
                 stopBtn.disabled = true;
                 resetBtn.disabled = false;

            } else if (statusData.status === "running") {
                statusSpan.textContent = '● Processing...';
                statusSpan.className = 'status-running';
            }
        } catch (error) {
            clearInterval(processingPollInterval);
            processingPollInterval = null;
            clearProcessingInfo(`Error checking status: ${error.message}`, true);
            log(`[FATAL] Could not get processing status: ${error.message}`, true);
            statusSpan.textContent = '● API Error';
            statusSpan.className = 'status-stopped';
             startBtn.disabled = false;
             stopBtn.disabled = true;
             resetBtn.disabled = false;
        }
    }, POLLING_INTERVAL_MS);
}

async function startBackendProcessing() {
    const trainLimit = 20000;
    const testLimit = 500;

    log(`Requesting backend processing (trainLimit=${trainLimit}, testLimit=${testLimit})`);
    statusSpan.textContent = '● Requesting...';
    statusSpan.className = 'status-running';
    startBtn.disabled = true;
    stopBtn.disabled = true;
    resetBtn.disabled = true;
    clearProcessingInfo();

    try {
        const body = {
            train_dataset: "train",
            test_dataset: "metro",
            window_size: 10,
            val_split: 0.15,
            max_train_rows: trainLimit,
            max_test_rows: testLimit
        };
        
        const initialResponse = await apiCall('/run_dataset_processing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        if (initialResponse && initialResponse.status === "processing_started") {
            log(`Backend processing started. Estimated time: ${initialResponse.estimated_seconds}s.`);
            displayProcessingInfo(initialResponse.estimated_seconds);
            pollProcessingStatus();
        } else {
            throw new Error("Backend did not start processing correctly.");
        }

    } catch (err) {
        statusSpan.textContent = '● Error';
        statusSpan.className = 'status-stopped';
        log(`[FATAL] Failed to start backend processing: ${err.message}`, true);
        clearProcessingInfo(`Error starting: ${err.message}`, true);
         startBtn.disabled = false;
         stopBtn.disabled = true;
         resetBtn.disabled = false;
    }
}

async function fetchBackendStatus() {
    try {
        const data = await apiCall('/status');
        backendCentroids = data.centroids || [];
    } catch (err) {
        console.warn('Could not fetch cluster status:', err);
    }
}

async function update() {
    if (!running || timeStep >= totalSteps) {
        running = false;
        statusSpan.textContent = '● Finished';
        statusSpan.className = 'status-stopped';
        log('Simulation finished.');
        stopBtn.disabled = true;
        resetBtn.disabled = false;
        return;
    }

    try {
        const stepData = await apiCall(`/get_simulation_step/${timeStep}`);
        if (!stepData) throw new Error("Received empty step data.");

        const trueRegime = stepData.detected_regime;
        const detectedRegime = stepData.detected_regime;

        if (timeStep > 0 && detectedRegime !== currentRegime) transitionCount++;
        currentRegime = detectedRegime;

        dataBuffer.push({
            t: stepData.time_step, value: stepData.true_value,
            regime: detectedRegime, features: stepData.features
        });
        if (dataBuffer.length > maxBufferSize) dataBuffer.shift();

        adaptiveErrors.push(stepData.adaptive_error);
        monolithicErrors.push(stepData.monolithic_error);

        if (trueRegime >= 0 && trueRegime < numRegimes) {
            regimeErrors[trueRegime].adaptive.push(stepData.adaptive_error);
            regimeErrors[trueRegime].monolithic.push(stepData.monolithic_error);
            regimeSamples[trueRegime]++;
        } else { console.warn(`Invalid trueRegime ${trueRegime} at t=${timeStep}`); }

    } catch (err) {
        log(`[FATAL] Simulation stopped at step ${timeStep}: ${err.message}`, true);
        running = false;
        statusSpan.textContent = '● API Error';
        statusSpan.className = 'status-stopped';
         startBtn.disabled = false;
         stopBtn.disabled = true;
         resetBtn.disabled = false;
        return;
    }

    updateUI();
    if (timeStep % 10 === 0) await fetchBackendStatus();
    updateCharts();

    timeStep++;
    if (running) {
        setTimeout(update, 50);
    }
}

function updateUI() {
    document.getElementById('timeStep').textContent = timeStep;
    document.getElementById('transitions').textContent = transitionCount;
    const regimeEl = document.getElementById('currentRegime');
    regimeEl.textContent = `Regime ${currentRegime}`;
    regimeEl.className = `regime-indicator regime-${currentRegime}`;
    
    let adaptiveMAE = 0.00; let monolithicMAE = 0.00; let improvement = 0.0;
    
    if (adaptiveErrors.length > 0) {
        adaptiveMAE = adaptiveErrors.reduce((a, b) => a + b, 0) / adaptiveErrors.length;
    }
    if (monolithicErrors.length > 0) {
        monolithicMAE = monolithicErrors.reduce((a, b) => a + b, 0) / monolithicErrors.length;
    }
    if (monolithicMAE > 0.001) {
        improvement = ((monolithicMAE - adaptiveMAE) / monolithicMAE) * 100;
    }

    document.getElementById('adaptiveMAE').textContent = isNaN(adaptiveMAE) ? "N/A" : adaptiveMAE.toFixed(2);
    document.getElementById('monolithicMAE').textContent = isNaN(monolithicMAE) ? "N/A" : monolithicMAE.toFixed(2);
    
    const improvementEl = document.getElementById('improvement');
    if (isNaN(improvement)) {
         improvementEl.textContent = 'N/A';
         improvementEl.style.color = 'var(--color-text-secondary)';
    } else {
        improvementEl.textContent = `${improvement.toFixed(1)}%`;
        improvementEl.style.color = improvement >= 0 ? 'var(--color-success)' : 'var(--color-error)';
    }
    updateComparisonTable();
}

function updateComparisonTable() {
     const tbody = document.getElementById('comparisonTable');
     tbody.innerHTML = '';
     for (let i = 0; i < numRegimes; i++) {
         const adaptive = regimeErrors[i].adaptive; const monolithic = regimeErrors[i].monolithic;
         let adaptiveMAE = 0.00; let monolithicMAE = 0.00; let improvement = 0.0;
         if (adaptive.length > 0) adaptiveMAE = adaptive.reduce((a, b) => a + b, 0) / adaptive.length;
         if (monolithic.length > 0) {
             monolithicMAE = monolithic.reduce((a, b) => a + b, 0) / monolithic.length;
             if (monolithicMAE > 0.001) improvement = ((monolithicMAE - adaptiveMAE) / monolithicMAE) * 100;
         }
         const row = document.createElement('tr');
         row.innerHTML = `<td>Regime ${i}</td><td>${adaptive.length > 0 && !isNaN(adaptiveMAE) ? adaptiveMAE.toFixed(2) : 'N/A'}</td><td>${monolithic.length > 0 && !isNaN(monolithicMAE) ? monolithicMAE.toFixed(2) : 'N/A'}</td><td style="color: ${improvement >= 0 ? 'var(--color-success)' : 'var(--color-error)'}">${(adaptive.length > 0 && monolithic.length > 0 && !isNaN(improvement)) ? (improvement >= 0 ? '+' : '') + improvement.toFixed(1) + '%' : 'N/A'}</td><td>${regimeSamples[i]}</td>`;
         tbody.appendChild(row);
     }
}

function initCharts() {
    timeSeriesChart = document.getElementById('timeSeriesChart')?.getContext('2d');
    clusterChart = document.getElementById('clusterChart')?.getContext('2d');
    errorChart = document.getElementById('errorChart')?.getContext('2d');
}

function updateCharts() {
    if (timeSeriesChart) updateTimeSeriesChart();
    if (clusterChart) updateClusterChart();
    if (errorChart) updateErrorChart();
}

function updateTimeSeriesChart() {
    if (!timeSeriesChart || dataBuffer.length < 1) return;
    const canvas = timeSeriesChart.canvas;
    const ctx = timeSeriesChart;
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    ctx.clearRect(0, 0, width, height);
    if (dataBuffer.length < 2) return;

    const darkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const gridColor = darkMode ? 'rgba(119, 124, 124, 0.2)' : 'rgba(94, 82, 64, 0.2)';

    for (let i = 0; i <= 5; i++) {
        const y = (height / 5) * i;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); 
        ctx.strokeStyle = gridColor; ctx.lineWidth = 1; ctx.stroke();
    }

    const recentData = dataBuffer;
    const values = recentData.map(d => d.value).filter(Number.isFinite);
    const maxVal = values.length > 0 ? Math.max(...values) : 1;
    const minVal = values.length > 0 ? Math.min(...values) : 0;
    const range = maxVal - minVal || 1;

    const xScale = width / Math.max(recentData.length - 1, 1);
    const yScale = height * 0.8 / range;
    const yOffset = height * 0.9;

    ctx.strokeStyle = darkMode ? '#32B8C6' : '#21808D';
    ctx.lineWidth = 2;
    ctx.beginPath();
    recentData.forEach((d, i) => {
        const x = i * xScale;
        const y = yOffset - (d.value - minVal) * yScale;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    recentData.forEach((d, i) => {
        if (i > 0 && d.regime !== recentData[i - 1].regime) {
            const x = i * xScale;
            ctx.strokeStyle = darkMode ? '#E68161' : '#A84B2F';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
            ctx.setLineDash([]);
        }
    });

    const textColor = darkMode ? '#f5f5f5' : '#134252';
    ctx.fillStyle = textColor;
    ctx.font = '11px monospace';
    ctx.fillText('Actual Values (Test Set)', 10, 20);
}

function updateClusterChart() {
    if (!clusterChart || dataBuffer.length < 1) return;
    const canvas = clusterChart.canvas;
    const ctx = clusterChart;
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    ctx.clearRect(0, 0, width, height);

    const darkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const colors = [
        darkMode ? '#3B82F6' : '#2563EB',
        darkMode ? '#F59E0B' : '#D97706',
        darkMode ? '#22C55E' : '#16A34A',
        darkMode ? '#EF4444' : '#DC2626'
    ];

    const points = dataBuffer.map(d => {
        const xVal = Array.isArray(d.features) && d.features.length > 0 ? d.features[0] : 0;
        const yVal = Array.isArray(d.features) && d.features.length > 2 ? d.features[2] : 0;
        return { x: xVal, y: yVal, regime: d.regime };
    });

    const centroidPoints = backendCentroids.map(c => {
        const xVal = Array.isArray(c) && c.length > 0 ? c[0] : 0;
        const yVal = Array.isArray(c) && c.length > 2 ? c[2] : 0;
         return { x: xVal, y: yVal };
    });

    const allPoints = [...points, ...centroidPoints];
    if (allPoints.length === 0) return;

    const xValues = allPoints.map(p => p.x).filter(Number.isFinite);
    const yValues = allPoints.map(p => p.y).filter(Number.isFinite);
    const xMin = xValues.length > 0 ? Math.min(...xValues) : 0;
    const xMax = xValues.length > 0 ? Math.max(...xValues) : 1;
    const yMin = yValues.length > 0 ? Math.min(...yValues) : 0;
    const yMax = yValues.length > 0 ? Math.max(...yValues) : 1;
    const xRange = (xMax - xMin) || 1;
    const yRange = (yMax - yMin) || 1;
    const padding = 40;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;
    const scaleX = (x) => padding + ((x - xMin) / xRange) * chartWidth * 0.9 + (chartWidth * 0.05);
    const scaleY = (y) => height - padding - ((y - yMin) / yRange) * chartHeight * 0.9 - (chartHeight * 0.05);

    points.forEach(p => {
        const regimeColorIndex = (p.regime >= 0 && p.regime < colors.length) ? p.regime : 0;
        ctx.fillStyle = colors[regimeColorIndex] + '80';
        ctx.beginPath(); ctx.arc(scaleX(p.x), scaleY(p.y), 4, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = colors[regimeColorIndex];
        ctx.lineWidth = 1;
        ctx.stroke();
    });

    if (backendCentroids.length > 0) {
        backendCentroids.forEach((centroid, i) => {
            if (i >= colors.length) return;
            const xVal = Array.isArray(centroid) && centroid.length > 0 ? centroid[0] : 0;
            const yVal = Array.isArray(centroid) && centroid.length > 2 ? centroid[2] : 0;
            const x = scaleX(xVal);
            const y = scaleY(yVal);
            ctx.strokeStyle = colors[i];
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6);
            ctx.moveTo(x - 6, y + 6); ctx.lineTo(x + 6, y - 6);
            ctx.stroke();
        });
    }

    const textColor = darkMode ? '#f5f5f5' : '#134252';
    ctx.fillStyle = textColor;
    ctx.font = '11px monospace';
    ctx.fillText('Roll. Mean →', width - 80, height - 10);
    ctx.save();
    ctx.translate(10, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Trend →', 0, 0);
    ctx.restore();
}

function updateErrorChart() {
    if (!errorChart) return;
    const canvas = errorChart.canvas;
    const ctx = errorChart;
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    ctx.clearRect(0, 0, width, height);

    const darkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const textColor = darkMode ? '#f5f5f5' : '#134252';
    
    const barWidth = width / (numRegimes * 2 + 1);
    const padding = 40;
    const chartHeight = height - 2 * padding;

    const adaptiveErrorsByRegime = regimeErrors.map(r => r.adaptive.length > 0 ? (r.adaptive.reduce((a, b) => a + b, 0) / r.adaptive.length) : 0);
    const monolithicErrorsByRegime = regimeErrors.map(r => r.monolithic.length > 0 ? (r.monolithic.reduce((a, b) => a + b, 0) / r.monolithic.length) : 0);
    const validErrors = [...adaptiveErrorsByRegime, ...monolithicErrorsByRegime].filter(Number.isFinite);
    const maxError = validErrors.length > 0 ? Math.max(...validErrors, 1) : 1;

    for (let i = 0; i < numRegimes; i++) {
        const x = barWidth * (i * 2 + 0.5);
        const adaptiveHeightRaw = (adaptiveErrorsByRegime[i] / maxError) * chartHeight;
        const adaptiveHeight = Math.max(0, isFinite(adaptiveHeightRaw) ? adaptiveHeightRaw : 0);
        ctx.fillStyle = darkMode ? '#32B8C6' : '#21808D';
        ctx.fillRect(x, height - padding - adaptiveHeight, barWidth * 0.8, adaptiveHeight);

        const monolithicHeightRaw = (monolithicErrorsByRegime[i] / maxError) * chartHeight;
        const monolithicHeight = Math.max(0, isFinite(monolithicHeightRaw) ? monolithicHeightRaw : 0);
        ctx.fillStyle = darkMode ? 'rgba(119, 124, 124, 0.5)' : 'rgba(94, 82, 64, 0.5)';
        ctx.fillRect(x + barWidth, height - padding - monolithicHeight, barWidth * 0.8, monolithicHeight);
    }

    ctx.fillStyle = textColor;
    ctx.font = '11px monospace';
    for (let i = 0; i < numRegimes; i++) {
        const x = barWidth * (i * 2 + 1);
        ctx.fillText(`R${i}`, x, height - padding + 20);
    }

    ctx.fillStyle = darkMode ? '#32B8C6' : '#21808D';
    ctx.fillRect(10, 10, 15, 15);
    ctx.fillStyle = textColor;
    ctx.fillText('Adaptive', 30, 20);
    ctx.fillStyle = darkMode ? 'rgba(119, 124, 124, 0.5)' : 'rgba(94, 82, 64, 0.5)';
    ctx.fillRect(120, 10, 15, 15);
    ctx.fillStyle = textColor;
    ctx.fillText('Monolithic', 140, 20);
}

document.getElementById('startBtn').addEventListener('click', async () => {
    if (running || processingPollInterval) return; 
    resetFrontendState();
    await startBackendProcessing(); 
});

document.getElementById('stopBtn').addEventListener('click', () => {
    running = false; 
    if (processingPollInterval) {
         clearInterval(processingPollInterval); 
         processingPollInterval = null;
    }
    statusSpan.textContent = '● Stopped';
    statusSpan.className = 'status-stopped';
    log('Simulation stopped by user.');
    startBtn.disabled = false;
    stopBtn.disabled = true;
    resetBtn.disabled = false;
});

document.getElementById('resetBtn').addEventListener('click', resetSimulation);

function resetFrontendState(){
    running = false;
    timeStep = 0;
    currentRegime = 0;
    transitionCount = 0;
    dataBuffer = [];
    adaptiveErrors = [];
    monolithicErrors = [];
    regimeErrors = Array(numRegimes).fill(null).map(() => ({ adaptive: [], monolithic: [] }));
    regimeSamples = Array(numRegimes).fill(0);
    
    if (statusSpan) statusSpan.textContent = '● Stopped';
    if (statusSpan) statusSpan.className = 'status-stopped';
    
    const logContainer = document.getElementById('logContainer');
    if (logContainer) logContainer.innerHTML = '<div class="log-entry">[INFO] System reset. Ready to start.</div>';

    updateUI();
    updateCharts(); 
}

async function resetSimulation() {
    running = false;
    if (processingPollInterval) {
         clearInterval(processingPollInterval); 
         processingPollInterval = null;
    }
    if (countdownInterval) {
        clearInterval(countdownInterval); 
        countdownInterval = null;
    }
    clearProcessingInfo("System reset.");

    resetFrontendState();
    backendCentroids = []; 
    totalSteps = 0;

    await fetchBackendStatus();
    updateUI(); 
    updateCharts();
    log('System reset. Press Start to run processing on backend.');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    if (resetBtn) resetBtn.disabled = false;
}

document.addEventListener('DOMContentLoaded', () => {
    statusSpan = document.getElementById('status');
    processingInfoDiv = document.getElementById('processing-info');
    startBtn = document.getElementById('startBtn');
    stopBtn = document.getElementById('stopBtn');
    resetBtn = document.getElementById('resetBtn');

    initCharts();
    resetSimulation(); 
});
