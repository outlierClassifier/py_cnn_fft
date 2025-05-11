import json
import numpy as np
import time, datetime
from typing import List
import re

from fastapi import HTTPException, Request, FastAPI
from scipy.fft import rfft
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, ReLU,
                                     GlobalAveragePooling2D, Dense, Dropout)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import uvicorn
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import time
import datetime
import os
import psutil
import logging
from signals import (Signal as InternalSignal, Discharge as InternalDischarge, 
                     DisruptionClass, SignalType, get_signal_type, normalize, are_normalized, mean_sensor_psd)
from zscore_normalizer import apply_zscore, compute_zscore_stats

PATTERN = "DES_(\\d+)_(\\d+)"
WINDOW_SIZE = 64
FREQ_BINS   = WINDOW_SIZE // 2 + 1
OVERLAP     = 0.5
SAMPLE_PER_DISCHARGE = 120
MODEL_PATH  = "cnn_fft_model.keras" 
start_time = time.time()
last_training_time = None
model = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Pydantic models for request/response based on API schemas
class Signal(BaseModel):
    fileName: str
    values: List[float]
    times: Optional[List[float]] = None
    length: Optional[int] = None

class Discharge(BaseModel):
    id: str
    times: Optional[List[float]] = None
    length: Optional[int] = None
    anomalyTime: Optional[float] = None
    signals: List[Signal]

class PredictionRequest(BaseModel):
    discharges: List[Discharge]

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    executionTimeMs: float
    model: str
    details: Optional[Dict[str, Any]] = None

class TrainingOptions(BaseModel):
    epochs: Optional[int] = None
    batchSize: Optional[int] = None
    hyperparameters: Optional[Dict[str, Any]] = None

class TrainingRequest(BaseModel):
    discharges: List[Discharge]
    options: Optional[TrainingOptions] = None

class TrainingMetrics(BaseModel):
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    f1Score: Optional[float] = None

class TrainingResponse(BaseModel):
    status: str
    message: Optional[str] = None
    trainingId: Optional[str] = None
    metrics: Optional[TrainingMetrics] = None
    executionTimeMs: float

class MemoryInfo(BaseModel):
    total: float
    used: float

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    uptime: float
    memory: MemoryInfo
    load: float
    lastTraining: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Initialize FastAPI application
app = FastAPI(
    title="CNN Anomaly Detection API",
    description="API for anomaly detection using CNN models",
    version="1.0.0"
)

def to_internal_discharges(discharges_pyd: List[Discharge]) -> List[InternalDischarge]:
    """Convert Pydantic discharges → InternalDischarge (sean Normal o Anomaly)."""
    internal = []
    for d in discharges_pyd:
        signals_int = []
        for s in d.signals:
            sig_type = get_signal_type(get_sensor_id(s))
            signals_int.append(
                InternalSignal(
                    label=s.fileName,
                    times=s.times or d.times,
                    values=s.values,
                    signal_type=sig_type,
                    disruption_class=DisruptionClass.Unknown
                )
            )
        internal.append(
            InternalDischarge(signals=signals_int, disruption_class=DisruptionClass.Unknown)
        )
    return internal


# Load model if exists
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        logger.info("Existing model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

def get_sensor_id(signal: Signal) -> str:
    """Extract sensor ID from signal file name"""
    match = re.match(PATTERN, signal.fileName)
    if match:
        return match.group(2)
    else:
        raise ValueError(f"Invalid signal file name format: {signal.fileName}")

def window_fft(signal: np.ndarray) -> np.ndarray:
    """
    Returns the FFT of the signal, normalized and reshaped
    """
    eps = 1e-9
    spec = np.abs(rfft(signal, axis=0))  # (FREQ_BINS, sensores)
    spec = np.log(spec + eps)
    return spec[..., np.newaxis]

def build_fft_cnn_model(n_sensors: int) -> Model:
    inp = Input(shape=(FREQ_BINS, n_sensors, 1))
    x = Conv2D(32, (3, 3), padding="same")(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def is_anomaly(discharge: Discharge) -> bool:
    """Determine if a discharge has an anomaly based on anomalyTime"""
    return discharge.anomalyTime is not None

@app.post('/train', response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    start_time = time.time()
    global model, last_training_time

    # 1) Parse internal discharges
    internal: List[InternalDischarge] = []
    for d in request.discharges:
        signals = [InternalSignal(
            label=s.fileName,
            times=s.times or d.times,
            values=s.values,
            signal_type=get_signal_type(get_sensor_id(s)),
            disruption_class=(DisruptionClass.Anomaly if d.anomalyTime else DisruptionClass.Normal)
        ) for s in d.signals]
        internal.append(InternalDischarge(signals=signals,
                          disruption_class=(DisruptionClass.Anomaly if d.anomalyTime else DisruptionClass.Normal)))
    
    # Normalize signals
    if not are_normalized(internal):
        internal = normalize(internal)

    # 2) Optional augment
    if len(internal) < 10:
        aug = []
        for disc in internal:
            aug.extend(disc.generate_similar_discharges(1))
        internal += aug

    # 3) Manual sliding windows + FFT
    X, y, groups = [], [], []
    stride = int(WINDOW_SIZE * (1 - OVERLAP))
    for disc_id, disc in enumerate(internal):
        # Montar matriz sensores x tiempo
        data = np.stack([s.values for s in disc.signals])  # shape (n_sensors, T)
        T = data.shape[1]
        # Generar ventanas
        idxs = list(range(0, T - WINDOW_SIZE + 1, stride))
        if len(idxs) > SAMPLE_PER_DISCHARGE:
            idxs = np.linspace(0, len(idxs)-1, SAMPLE_PER_DISCHARGE, dtype=int).tolist()
            idxs = [idxs[i] * stride for i in range(len(idxs))]
        for start in idxs:
            window = data[:, start:start+WINDOW_SIZE].T  # (WINDOW_SIZE, n_sensors)
            fft_spec = window_fft(window)
            X.append(fft_spec)
            y.append(1 if disc.disruption_class == DisruptionClass.Anomaly else 0)
            groups.append(disc_id)

    X = np.stack(X)
    y = np.array(y)
    groups = np.array(groups)

    # 4) CV con balance de descargas
    splitter = GroupShuffleSplit(n_splits=6, test_size=0.25, random_state=42)
    for fold, (tr, val) in enumerate(splitter.split(X, y, groups), start=1):
        print(f'--- Fold {fold} ---')
        X_tr, y_tr = X[tr], y[tr]
        X_val, y_val = X[val], y[val]

        model = build_fft_cnn_model(n_sensors=X.shape[2])
        callbacks = [EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
                     ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)]
        model.fit(X_tr, y_tr, validation_data=(X_val,y_val), epochs=40, batch_size=16,
                  class_weight={0:1.0,1:1.0}, callbacks=callbacks, verbose=2)

        preds = (model.predict(X_val)[:,0] > 0.5).astype(int)
        print(classification_report(y_val, preds, target_names=['Normal','Anomaly'], zero_division=0))

    # Guardar modelo
    model.save(MODEL_PATH)
    last_training_time = datetime.datetime.now().isoformat()
    
    elapsed = int((time.time() - start_time)*1000)
    return TrainingResponse(status='success', message='FFT-CNN trained',
                             trainingId=f'train_{datetime.datetime.now():%Y%m%d_%H%M%S}',
                             metrics=TrainingMetrics(accuracy=None,loss=None,f1Score=None),
                             executionTimeMs=elapsed)

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    start_time = time.time()
    for disc in request.discharges:
        data = np.stack([s.values for s in disc.signals])  # (n_sensors, T)
    
    discharges_int = to_internal_discharges(request.discharges)
    if not are_normalized(discharges_int):
        discharges_int = normalize(discharges_int)
    
    
    mean_probs = []
    stride = int(WINDOW_SIZE * (1 - OVERLAP))
    for disc in discharges_int:
        data = np.stack([s.values for s in disc.signals])  # shape (n_sensors, T)
        windows = []
        for start in range(0, data.shape[1] - WINDOW_SIZE + 1, stride):
            win = data[:, start:start+WINDOW_SIZE].T
            windows.append(window_fft(win))

        if not windows:
            continue
        X_batch = np.stack(windows)
        probs = model.predict(X_batch)[:, 0]
        mean_probs.append(probs.mean())
    

    if not mean_probs:
        print("No valid windows generated for prediction")
        raise HTTPException(status_code=400, detail='No valid windows generated for prediction')

    overall_confidence = float(np.mean(mean_probs))
    prediction = int(overall_confidence > 0.5)
    exec_ms = int((time.time() - start_time) * 1000)
    mean_probs = np.array(mean_probs)

    return PredictionResponse(
        prediction=prediction,
        confidence=overall_confidence,
        executionTimeMs=exec_ms,
        model="fft_cnn",
        details={
            "individualPredictions": mean_probs.tolist()
        }
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    # Get memory information
    mem = psutil.virtual_memory()
    
    return HealthCheckResponse(
        status="online" if model is not None else "degraded",
        version="1.0.0",
        uptime=time.time() - start_time,
        memory=MemoryInfo(
            total=mem.total / (1024*1024),  # Convert to MB
            used=mem.used / (1024*1024)
        ),
        load=psutil.cpu_percent() / 100,
        lastTraining=last_training_time
    )

# Custom middleware to handle large request JSON payloads
@app.middleware("http")
async def increase_json_size_limit(request: Request, call_next):
    # Increase JSON size limit for this specific request
    # Default is 1MB, we're setting it to 64MB
    request.app.state.json_size_limit = 64 * 1024 * 1024  # 64MB
    response = await call_next(request)
    return response

if __name__ == "__main__":
    # Try to load the model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            logger.info("Existing model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            model = None

    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False, 
                limit_concurrency=50, 
                limit_max_requests=20000,
                timeout_keep_alive=120)

