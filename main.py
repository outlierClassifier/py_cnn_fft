import json
import random
import numpy as np
import time, datetime
from typing import List
import re
from collections import Counter

from fastapi import HTTPException, Request, FastAPI
from scipy.fft import rfft
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
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
from pydantic import BaseModel, Field, validator
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
WINDOW_SIZE = 16
FREQ_BINS   = WINDOW_SIZE // 2 + 1
OVERLAP     = 0
SAMPLE_PER_DISCHARGE = 120
MODEL_PATH  = "cnn_fft_model copy.keras"
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

    @validator('signals', pre=True)
    def _ensure_list(cls, v):
        if isinstance(v, dict):
            return [v]
        return v


class WindowProperties(BaseModel):
    featureValues: List[float] = Field(..., min_items=1)
    prediction: str = Field(..., pattern=r'^(Anomaly|Normal)$')
    justification: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    executionTimeMs: float
    model: str
    windowSize: int = WINDOW_SIZE
    windows: List[WindowProperties]

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
    spec = np.log(spec + eps).astype(np.float32)
    return spec[..., np.newaxis]

def window_label_and_weight(
    times,
    anomaly_time: float | None,
    end_idx: int,
    dt_s: float,
    H_pre: float = 0.5,
    H_post: float = 0.5,
    post_plateau_windows: int = 2,
    post_tail_floor: float = 0.3
) -> tuple[int, float]:
    if anomaly_time is None:
        return 0, 1.0  # non disruptive discharge

    t_end = times[end_idx]
    dt = anomaly_time - t_end   # >0: pre-event, <0: post-event

    # Far before the event
    if dt >= H_pre:
        return 0, 1.0

    # Pre-event with linear ramp of weights
    if 0.0 <= dt < H_pre:
        w = 1.0 - (dt / H_pre)     # 0 in -H_pre, 1 in 0
        return 1, w

    # Post-event
    # Duration of the plateau after the event
    plateau_s = max(0, post_plateau_windows) * dt_s * WINDOW_SIZE
    dt_post = -dt  # seconds elapsed since the event

    if dt_post <= plateau_s:
        # Immediately after: as important as the event
        return 1, 1.0

    if dt_post <= H_post:
        # Tail with constant weight (non-zero)
        return 1, max(0.0, float(post_tail_floor))

    # Beyond the POST horizon: if you want it to NEVER decay, we keep the floor
    return 1, max(0.0, float(post_tail_floor))

A_MINOR = 0.95  # [m] radio menor de JET (usa tu valor de referencia y documenta)

def _safe_div(a, b):
    return 0.0 if b == 0 or not np.isfinite(a) or not np.isfinite(b) else (a / b)

def inter_signal_features(win_raw: np.ndarray, prev_logs: tuple[float,float] | None):
    mean_vals = np.mean(win_raw, axis=1)  # (7,)
    Ip, LM, LI, NE, dWdt, Prad, Pin = mean_vals

    Ip_MA  = Ip * 1e-6
    ne_1e20 = NE * 1e-20

    rad_frac  = _safe_div(Prad, Pin)
    greenwald = _safe_div(ne_1e20, _safe_div(Ip_MA, (np.pi * A_MINOR**2)))
    LM_norm   = _safe_div(abs(LM), max(Ip_MA, 1e-12))
    li_norm   = _safe_div(LI, max(Ip_MA, 1e-12))
    beta_loss = _safe_div(abs(dWdt), max(Pin, 1e-12))
    cross_std = float(np.std(mean_vals))
    logE      = float(np.log1p(np.mean(win_raw.astype(np.float64) ** 2)))

    L_rad   = np.log1p(max(0.0, rad_frac))
    L_gw    = np.log1p(max(0.0, greenwald))
    L_lm    = np.log1p(LM_norm)
    L_li    = np.log1p(abs(li_norm))
    L_beta  = np.log1p(beta_loss)

    # DELTAS (diferencia en el espacio log-comprimido)
    if prev_logs is None:
        d_rad, d_beta = 0.0, 0.0
    else:
        prev_Lrad, prev_Lbeta = prev_logs
        d_rad  = L_rad  - prev_Lrad
        d_beta = L_beta - prev_Lbeta

    feat = np.array([L_rad, L_gw, L_lm, L_li, L_beta, cross_std, logE, d_rad, d_beta],
                    dtype=np.float32)

    return feat, (L_rad, L_beta)


def build_fft_cnn_model(n_sensors: int, n_aux: int = 1) -> Model:
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
    inp_e = Input(shape=(n_aux,), name="energy_input")
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
    raw_arrays = [np.stack([s.values for s in d.signals]) for d in discharges]
    times_list = [np.array(d.times) for d in discharges]
    anom_times = [d.anomalyTime for d in discharges]
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
    X, AUX, y, W, groups = [], [], [], [], []
    stride = int(WINDOW_SIZE * (1 - OVERLAP))

    for disc_id, disc in enumerate(internal):
        data_norm = np.stack([s.values for s in disc.signals])
        data_raw  = raw_arrays[disc_id]
        times     = times_list[disc_id]
        anom_t    = anom_times[disc_id]
        dt_s      = float(np.median(np.diff(times))) if len(times) > 1 else 0.001

        T = data_norm.shape[1]
        idxs = list(range(0, T - WINDOW_SIZE + 1, stride))
        # (aquí entra el recorte prioritario por peso del apartado 1)

        prev_logs = None
        for start in idxs:
            end = start + WINDOW_SIZE
            win_norm = data_norm[:, start:end].T
            win_raw  = data_raw[:,  start:end]

            fft_spec = window_fft(win_norm)
            aux_vec, prev_logs = inter_signal_features(win_raw, prev_logs)

            end_idx = min(end - 1, len(times) - 1)
            yi, wi  = window_label_and_weight(times, anom_t, end_idx, dt_s)

            X.append(fft_spec)
            AUX.append(aux_vec)
            y.append(yi)
            W.append(wi)
            groups.append(disc_id)


    y_disc = [1 if d.disruption_class == DisruptionClass.Anomaly else 0 for d in internal]
    y_disc = np.array(y_disc)

    X   = np.stack(X).astype(np.float32)
    AUX = np.stack(AUX).astype(np.float32)
    y   = np.array(y, dtype=int)
    W   = np.array(W, dtype=float)
    groups = np.array(groups)

    aux_mean = np.nanmean(AUX, axis=0)
    aux_std  = np.nanstd(AUX,  axis=0) + 1e-6
    AUX_norm = ((AUX - aux_mean) / aux_std).astype(np.float32)

    with open('aux_stats.json', 'w') as f:
        json.dump({'mean': aux_mean.tolist(), 'std': aux_std.tolist()}, f)


    gkf = GroupKFold(n_splits=2)
    folds = []
    neg_probs_all, pos_probs_all = [], []
    neg_max_by_shot, pos_max_by_shot = [], []
    n_aux = AUX_norm.shape[1]

    for tr_idx, val_idx in gkf.split(X, y, groups):
        tr_mask = np.zeros(len(y), dtype=bool); tr_mask[tr_idx] = True
        val_mask = ~tr_mask
        folds.append((tr_mask, val_mask))

    for fold, (tr_mask, val_mask) in enumerate(folds, start=1):
        X_tr, A_tr, y_tr, W_tr = X[tr_mask], AUX_norm[tr_mask], y[tr_mask], W[tr_mask]
        X_val, A_val, y_val    = X[val_mask], AUX_norm[val_mask], y[val_mask]

        cnt = Counter(y_val)
        print(f'--- Fold: {fold} support: {cnt} ---')

        cw = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=y_tr)
        cdict = {0: float(cw[0]), 1: float(cw[1])}
        sw_tr = np.asarray([cdict[yi] * wi for yi, wi in zip(y_tr, W_tr)], dtype=np.float32)
        med = np.median(sw_tr) if np.isfinite(sw_tr).any() else 1.0
        sw_tr = sw_tr / (med + 1e-8)
        sw_tr = np.clip(sw_tr, 1e-3, 100.0)

        model = build_fft_cnn_model(n_sensors=X.shape[2], n_aux=n_aux)

        callbacks = [EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
                     ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)]

        model.fit([X_tr, A_tr], y_tr, 
                  validation_data=([X_val, A_val], y_val), 
                  epochs=40, 
                  batch_size=16,
                  sample_weight=sw_tr,
                  callbacks=callbacks,
                  verbose=2)

        w_E, b_E = model.get_layer('dense_energy').get_weights()
        print(f"||W_E||: {np.linalg.norm(w_E)}")
        print(f"||b_E||: {np.linalg.norm(b_E)}")

        val_probs = model.predict([X_val, A_val], verbose=0)[:, 0]

        preds = (val_probs > 0.5).astype(int)
        print(classification_report(y_val, preds, labels=[0,1],
            target_names=['Normal','Anomaly'], zero_division=0))

        g_val = groups[val_mask]
        shots = np.unique(g_val)
        for sh in shots:
            m = (g_val == sh)
            max_p = float(np.max(val_probs[m]))
            y_sh = int(np.max(y_val[m]))
            if y_sh == 0:
                neg_max_by_shot.append(max_p)
            else:
                pos_max_by_shot.append(max_p)

    if len(neg_max_by_shot) > 0:
        q = 0.99
        tau = float(np.quantile(np.array(neg_max_by_shot), q))
        with open('threshold.json', 'w') as f:
            json.dump({'tau': tau, 'quantile': q, 'neg_shots': len(neg_max_by_shot)}, f)
        print(f'[CALIB] TAU (per-shot) = p{int(q*100)} of negatives: {tau:.4f} '
            f'(neg shots={len(neg_max_by_shot)})')
    else:
        print('[CALIB] No negative shots in validation; TAU not updated.')


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
async def predict(discharge: Discharge):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    start_time = time.time()

    discharges_int = to_internal_discharges([discharge])
    with open('zscore_stats.json', 'r') as f:
        stats = {SignalType[k]: v for k, v in json.load(f).items()}
    if not are_normalized(discharges_int):
        discharges_int = apply_zscore(discharges_int, stats)

    try:
        with open('aux_stats.json', 'r') as f:
            aux_stats = json.load(f)
        aux_mean = np.array(aux_stats['mean'], dtype=np.float32)
        aux_std  = np.array(aux_stats['std'],  dtype=np.float32)
    except Exception:
        aux_mean = aux_std = None

    data_norm = np.stack([s.values for s in discharges_int[0].signals])  # (n_sensors, T)

    data_raw = np.stack([s.values for s in discharge.signals])           # (n_sensors, T)

    window_probs, all_windows = [], []
    stride = int(WINDOW_SIZE * (1 - OVERLAP))

    prev_logs = None
    for start in range(0, data_norm.shape[1] - WINDOW_SIZE + 1, stride):
        win_norm = data_norm[:, start:start+WINDOW_SIZE].T
        win_raw  = data_raw[:,  start:start+WINDOW_SIZE]

        X_batch = np.expand_dims(window_fft(win_norm), axis=0)
        aux_vec, prev_logs = inter_signal_features(win_raw, prev_logs)
        aux_vec = aux_vec[np.newaxis, :].astype(np.float32)
        if aux_mean is not None:
            aux_vec = ((aux_vec - aux_mean) / aux_std).astype(np.float32)

        prob = float(model.predict([X_batch, aux_vec], verbose=0)[0, 0])
        window_probs.append(prob); all_windows.append(X_batch[0])

    if not window_probs:
        raise HTTPException(status_code=400, detail='No valid windows generated for prediction')

    max_prob = float(np.max(window_probs))
    TAU = 0.5
    try:
        with open('threshold.json', 'r') as f:
            TAU = float(json.load(f).get('tau', 0.5))
    except Exception:
        pass

    pred_label = "Anomaly" if max_prob >= TAU else "Normal"
    overall_confidence = max_prob if pred_label == "Anomaly" else 1.0 - max_prob

    return PredictionResponse(
        prediction=pred_label,
        confidence=overall_confidence,
        executionTimeMs=int((time.time() - start_time) * 1000),
        model="fft_cnn",
        windowSize=WINDOW_SIZE,
        windows=[
            WindowProperties(
                featureValues=list(win.flatten()),
                prediction=("Anomaly" if prob >= TAU else "Normal"),
                justification=float(prob)
            ) for win, prob in zip(all_windows, window_probs)
        ]
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

    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False)

