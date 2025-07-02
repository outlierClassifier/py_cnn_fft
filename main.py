import json
import random
import numpy as np
import time, datetime
from typing import List
import re
from collections import Counter

from fastapi import HTTPException, Request, FastAPI
from scipy.fft import rfft
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, ReLU,
                                     GlobalAveragePooling2D, Dense, Dropout, Concatenate)
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

SEED = 50
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
PATTERN = "DES_(\\d+)_(\\d+)"
WINDOW_SIZE = 256
FREQ_BINS   = WINDOW_SIZE // 2 + 1
OVERLAP     = 0.5
SAMPLE_PER_DISCHARGE = 120
MODEL_PATH  = "cnn_fft_model.keras"
start_time = time.time()
last_training_time = None
model = None
training_session = None  # holds pending training discharges

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Pydantic models for request/response based on API schemas
class Signal(BaseModel):
    filename: str
    values: List[float]
    # allow legacy field name
    class Config:
        fields = {"filename": "fileName"}

class Discharge(BaseModel):
    id: str
    times: List[float]
    length: int
    anomalyTime: Optional[float] = None
    signals: List[Signal]


class PredictionRequest(BaseModel):
    discharge: Discharge

class PredictionResponse(BaseModel):
    prediction: str
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

class StartTrainingRequest(BaseModel):
    totalDischarges: int
    timeoutSeconds: int

class StartTrainingResponse(BaseModel):
    expectedDischarges: int

class DischargeAck(BaseModel):
    ordinal: int
    totalDischarges: int

class MemoryInfo(BaseModel):
    total: float
    used: float

class HealthCheckResponse(BaseModel):
    name: str
    uptime: float
    lastTraining: Optional[str] = None
    # additional optional information
    status: Optional[str] = None
    version: Optional[str] = None
    memory: Optional[MemoryInfo] = None
    load: Optional[float] = None

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
                    label=s.filename,
                    times=d.times,
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
    match = re.match(PATTERN, signal.filename)
    if match:
        return match.group(2)
    else:
        raise ValueError(f"Invalid signal file name format: {signal.filename}")

def window_fft(signal: np.ndarray) -> np.ndarray:
    """
    Returns the FFT of the signal, normalized and reshaped
    """
    eps = 1e-9
    spec = np.abs(rfft(signal, axis=0))  # (FREQ_BINS, sensores)
    spec = np.log(spec + eps)
    return spec[..., np.newaxis]

def build_fft_cnn_model(n_sensors: int) -> Model:
    # Spectral branch
    inp_spec = Input(shape=(FREQ_BINS, n_sensors, 1), name="spectral_input")
    x = Conv2D(32, (3, 3), padding="same")(inp_spec)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)

    # Energy branch
    inp_e = Input(shape=(1,), name="energy_input")
    e = Dense(32, activation="relu")(inp_e)
    e = Dense(16, activation="relu", name="dense_energy")(e)

    cat = Concatenate()([x, e])
    cat = Dense(32, activation="relu")(cat)
    cat = Dropout(0.3)(cat)
    out = Dense(1, activation="sigmoid")(cat)
    model = Model(inputs=[inp_spec, inp_e], outputs=out)
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def is_anomaly(discharge: Discharge) -> bool:
    """Determine if a discharge has an anomaly based on anomalyTime"""
    return discharge.anomalyTime is not None

def train_model(discharges: List[Discharge]) -> TrainingResponse:
    start_time = time.time()
    global model, last_training_time

    # 1) Parse internal discharges
    internal: List[InternalDischarge] = []
    for d in discharges:
        signals = [InternalSignal(
            label=s.filename,
            times=d.times,
            values=s.values,
            signal_type=get_signal_type(get_sensor_id(s)),
            disruption_class=(DisruptionClass.Anomaly if d.anomalyTime else DisruptionClass.Normal)
        ) for s in d.signals]
        internal.append(InternalDischarge(signals=signals,
                          disruption_class=(DisruptionClass.Anomaly if d.anomalyTime else DisruptionClass.Normal)))
        
    stats = compute_zscore_stats(internal)

    # Convert SignalType enum keys to strings for JSON serialization
    serializable_stats = {}
    for key, value in stats.items():
        if isinstance(key, SignalType):
            serializable_stats[key.name] = value  # Using the name attribute of the enum
        else:
            serializable_stats[str(key)] = value
            
    # save stats to a json for inference
    with open('zscore_stats.json', 'w') as f:
        json.dump(serializable_stats, f)
    
    # Normalize signals
    if not are_normalized(internal):
        internal = apply_zscore(internal, stats)


    # 2) Optional augment
    if len(internal) < 10:
        aug = []
        for disc in internal:
            aug.extend(disc.generate_similar_discharges(1))
        internal += aug

    # 3) Manual sliding windows + FFT
    X, E, y, groups = [], [], [], []
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
            E_win = np.mean(window ** 2).astype(np.float32)
            X.append(fft_spec)
            E.append(E_win)
            y.append(1 if disc.disruption_class == DisruptionClass.Anomaly else 0)
            groups.append(disc_id)

    # etiquetas a nivel de descarga
    y_disc = [1 if d.disruption_class == DisruptionClass.Anomaly else 0 for d in internal]
    y_disc = np.array(y_disc)
    groups_disc = np.arange(len(internal))

    X = np.stack(X)
    E = np.asarray(E).reshape(-1, 1)
    y = np.array(y)
    groups = np.array(groups)

    max_trials = 250
    n_splits = min(4, np.bincount(y_disc).min())
    balanced_splits = None

    for trial in range(max_trials):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED + trial)
        candidate = [] # store tr, val for this trial

        for (tr, val) in splitter.split(groups_disc, y_disc, groups_disc):
            cnt = Counter(y_disc[val])
            ratio = min(cnt.values()) / max(cnt.values())

            if len(cnt) == 1:
                break

            if ratio < 0.25:
                break
            candidate.append((tr, val))
        else:
            # If we reach here, it means we found a balanced split
            balanced_splits = candidate
            break
    
    if balanced_splits is None:
        raise RuntimeError(
            f"Unable to find a balanced split after {max_trials} trials. "
            "Consider reducing the number of splits or adjusting the distribution of discharges."
        )

    for fold, (tr, val) in enumerate(balanced_splits, start=1):
        tr_mask = np.isin(groups, tr)
        val_mask = ~tr_mask

        X_tr, E_tr, y_tr = X[tr_mask], E[tr_mask], y[tr_mask]
        X_val, E_val, y_val = X[val_mask], E[val_mask], y[val_mask]

        cnt = Counter(y_val)
        print(f'--- Fold: {fold} support: {cnt} ---')

        model = build_fft_cnn_model(n_sensors=X.shape[2])
        callbacks = [EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
                     ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)]
        class_weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
        class_weights = {i: w for i, w in enumerate(class_weights)}
        model.fit([X_tr, E_tr], y_tr, 
                  validation_data=([X_val, E_val], y_val), 
                  epochs=40, 
                  batch_size=16,
                  class_weight=class_weights, 
                  callbacks=callbacks, 
                  verbose=2)

        w_E, b_E = model.get_layer('dense_energy').get_weights()
        print(f"||W_E||: {np.linalg.norm(w_E)}")
        print(f"||b_E||: {np.linalg.norm(b_E)}")

        preds = (model.predict([X_val, E_val])[:,0] > 0.5).astype(int)
        print(classification_report(y_val, 
                                    preds,
                                    labels=[0, 1],
                                    target_names=['Normal','Anomaly'], 
                                    zero_division=0))

    # Guardar modelo
    model.save(MODEL_PATH)
    last_training_time = datetime.datetime.now().isoformat()
    
    elapsed = int((time.time() - start_time)*1000)
    return TrainingResponse(status='success', message='FFT-CNN trained',
                             trainingId=f'train_{datetime.datetime.now():%Y%m%d_%H%M%S}',
                             metrics=TrainingMetrics(accuracy=None,loss=None,f1Score=None),
                             executionTimeMs=elapsed)


# ---------------------------------------------------------------------------
# Outlier protocol training endpoints
# ---------------------------------------------------------------------------

@app.post('/train', response_model=StartTrainingResponse)
async def start_training(request: StartTrainingRequest):
    global training_session
    if training_session is not None:
        raise HTTPException(status_code=503, detail='Training already in progress')
    training_session = {
        'total': request.totalDischarges,
        'discharges': [],
    }
    return StartTrainingResponse(expectedDischarges=request.totalDischarges)


@app.post('/train/{ordinal}', response_model=DischargeAck)
async def push_discharge(ordinal: int, discharge: Discharge):
    global training_session
    if training_session is None:
        raise HTTPException(status_code=400, detail='No active training session')
    if ordinal != len(training_session['discharges']) + 1:
        raise HTTPException(status_code=400, detail='Unexpected ordinal')
    training_session['discharges'].append(discharge)
    total = training_session['total']
    if ordinal == total:
        # run training synchronously
        train_model(training_session['discharges'])
        training_session = None
    return DischargeAck(ordinal=ordinal, totalDischarges=total)

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    start_time = time.time()
    disc = request.discharge
    data = np.stack([s.values for s in disc.signals])  # (n_sensors, T)
    discharges_int = to_internal_discharges([disc])
    with open('zscore_stats.json', 'r') as f:
        stats = {SignalType[k]: v for k, v in json.load(f).items()}

    if not are_normalized(discharges_int):
        discharges_int = apply_zscore(discharges_int, stats)
    
    
    mean_probs = []
    stride = int(WINDOW_SIZE * (1 - OVERLAP))
    for disc in discharges_int:
        data = np.stack([s.values for s in disc.signals])  # shape (n_sensors, T)
        windows, energies = [], []
        for start in range(0, data.shape[1] - WINDOW_SIZE + 1, stride):
            win = data[:, start:start+WINDOW_SIZE].T
            windows.append(window_fft(win))
            energies.append(np.mean(win ** 2).astype(np.float32))

        if not windows:
            continue
        X_batch = np.stack(windows)
        E_batch = np.asarray(energies).reshape(-1, 1)
        probs = model.predict([X_batch, E_batch])[:, 0]
        mean_probs.append(probs.mean())
    

    if not mean_probs:
        print("No valid windows generated for prediction")
        raise HTTPException(status_code=400, detail='No valid windows generated for prediction')

    overall_confidence = float(np.mean(mean_probs))
    pred_label = "Anomaly" if overall_confidence > 0.5 else "Normal"
    exec_ms = int((time.time() - start_time) * 1000)
    mean_probs = np.array(mean_probs)

    return PredictionResponse(
        prediction=pred_label,
        confidence=overall_confidence if pred_label == "Anomaly" else 1 - overall_confidence,
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
        name="fft_cnn",
        uptime=time.time() - start_time,
        lastTraining=last_training_time,
        status="online" if model is not None else "degraded",
        version="1.0.0",
        memory=MemoryInfo(
            total=mem.total / (1024*1024),  # Convert to MB
            used=mem.used / (1024*1024)
        ),
        load=psutil.cpu_percent() / 100,
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

