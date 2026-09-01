import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, sosfiltfilt, detrend, find_peaks, correlate
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import median_filter
import torch
from typing import Dict, Any

from models import model_manager

CFG = {
    "target_fs": 100.0, "win_sec_det": 2.5, "step_sec_det": 0.5,
    "win_sec_seg": 1.5, "step_sec_seg": 0.2, "acc_cutoff": 20.0, "gyr_cutoff": 15.0,
}

FS = CFG["target_fs"]

WIN_SIZE_DET = int(CFG["target_fs"] * CFG["win_sec_det"])
STEP_SIZE_DET = int(CFG["target_fs"] * CFG["step_sec_det"])
WIN_SIZE_SEG = int(CFG["target_fs"] * CFG["win_sec_seg"])
STEP_SIZE_SEG = int(CFG["target_fs"] * CFG["step_sec_seg"])

# Регистр 'X' сохранен для совпадения с моделью
FEATURE_NAMES = ['cadence', 'sample_entropy', 'freeze_index_mean', 'step_time_cv', 'X_std', 'X_rms', 'X_jerk_std']

_last_processed = {'df_proc': None}

# --- ФИЛЬТРЫ И НОРМАЛИЗАЦИЯ ---

def butter_lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data)

def butter_highpass(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    sos = butter(order, cutoff / nyq, btype='high', analog=False, output='sos')
    return sosfiltfilt(sos, data)

def butter_bandpass(data, lowcut=0.5, highcut=7.0, fs=FS, order=4):
    """Полосовой фильтр Баттерворта 4-го порядка (по умолчанию 0.5–7 Гц)."""
    try:
        data = np.array(data, dtype=float)
        if len(data) < 15:
            return data
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return filtfilt(b, a, data)
    except Exception:
        return data

def minmax_scale(data):
    """MinMax нормализация в диапазон [0, 1]."""
    data = np.array(data, dtype=float)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def preprocess_signal_for_features(data, lowcut=0.5, highcut=7.0, fs=FS, order=4):
    """Комбинированная предобработка: Баттерворт 0.5-7 Гц + MinMax нормализация."""
    filtered = butter_bandpass(data, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    return minmax_scale(filtered)

# --- ИНФРАСТРУКТУРА ДЕТЕКЦИИ И СЕГМЕНТАЦИИ ---

def calculate_imu_angles(df):
    acc_x, acc_y, acc_z = df["acc_x"].values, df["acc_y"].values, df["acc_z"].values
    pitch = np.arctan2(acc_x, np.sqrt(acc_y**2 + acc_z**2))
    roll = np.arctan2(acc_y, acc_z)
    yaw = np.cumsum(df["gyr_z"].values * 1 / CFG['target_fs']) if "gyr_z" in df.columns else np.zeros_like(pitch)
    return yaw, pitch, roll

def resample_and_prep(df):
    df = df.copy()
    for c in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]:
        if c in df.columns: df[c] = df[c].ffill().bfill().fillna(0.0)

    df["ori_yaw"], _, _ = calculate_imu_angles(df)
    t_old = df["time_sec"].values
    dt = 1.0 / CFG["target_fs"]
    
    resampled = {"time_sec": t_old}
    for c in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "ori_yaw"]:
        resampled[c] = np.interp(t_old, t_old, df[c].values)

    df_res = pd.DataFrame(resampled)
    for ax in ["x", "y", "z"]:
        df_res[f"acc_{ax}"] = butter_lowpass(df_res[f"acc_{ax}"].values, CFG["acc_cutoff"], CFG["target_fs"])
        df_res[f"gyr_{ax}"] = butter_lowpass(df_res[f"gyr_{ax}"].values, CFG["gyr_cutoff"], CFG["target_fs"])
    
    df_res["ori_yaw"] = butter_lowpass(df_res["ori_yaw"].values, CFG["gyr_cutoff"], CFG["target_fs"])
    try:
        df_res["ori_yaw"] = butter_highpass(df_res["ori_yaw"].values, cutoff=0.1, fs=CFG["target_fs"])
    except ValueError:
        df_res["ori_yaw"] = detrend(df_res["ori_yaw"].values)
    
    df_res["acc_mag"] = np.sqrt(df_res["acc_x"]**2 + df_res["acc_y"]**2 + df_res["acc_z"]**2)
    df_res["gyr_mag"] = np.sqrt(df_res["gyr_x"]**2 + df_res["gyr_y"]**2 + df_res["gyr_z"]**2)

    for c in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "ori_yaw", "acc_mag", "gyr_mag"]:
        df_res[f"{c}_d1"] = butter_lowpass(np.gradient(df_res[c].values, dt), cutoff=3.0, fs=CFG["target_fs"])

    features = ["acc_mag", "gyr_mag", "ori_yaw", "acc_mag_d1", "gyr_mag_d1"]
    return np.nan_to_num(df_res[features].values), df_res["time_sec"].values, df_res

def create_windows(data_mat, time_array, win_size, step_size):
    X, T = [], []
    for i in range(0, len(data_mat) - win_size + 1, step_size):
        X.append(data_mat[i : i + win_size])
        T.append(time_array[i + win_size // 2])
    return np.array(X), np.array(T)

def smooth_predictions(preds, k=5):
    return median_filter(preds, size=k).astype(int)

def find_true_movement_boundaries(det_pred, min_duration_sec=1.5):
    min_windows = int(min_duration_sec / CFG["step_sec_det"])
    padded = np.concatenate(([0], np.clip(det_pred, 0, 1), [0]))
    diff = np.diff(padded)
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    valid = [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_windows]
    return max(valid, key=lambda x: x[1] - x[0]) if valid else None

def convert_windows_to_points(df, T_windows, window_seg_pred, det_pred, main_segment):
    point_preds = np.zeros(len(df), dtype=int)
    if main_segment is None: return point_preds
    t_points = df["time_sec"].values
    st_t, en_t = T_windows[main_segment[0]], T_windows[main_segment[1] - 1]
    mask = (t_points >= st_t) & (t_points <= en_t)
    idx = np.clip(np.searchsorted(T_windows, t_points[mask]), 0, len(T_windows) - 1)
    prev_idx = np.clip(idx - 1, 0, len(T_windows) - 1)
    diff_curr = np.abs(T_windows[idx] - t_points[mask])
    diff_prev = np.abs(T_windows[prev_idx] - t_points[mask])
    point_preds[mask] = window_seg_pred[np.where(diff_prev < diff_curr, prev_idx, idx)]
    return point_preds

def extract_precise_timestamps(time_signals, point_seg_pred, det_pred, pred_segment, T_det):
    timestamps = {"T1_start": None, "T2_turn1_start": None, "T3_walk2_start": None, "T4_turn2_start": None, "T5_end": None}
    if pred_segment is None or np.sum(det_pred == 1) == 0: return timestamps
    t_start, t_end = T_det[pred_segment[0]], T_det[pred_segment[1]-1]
    idx_start, idx_end = int(np.argmin(np.abs(time_signals - t_start))), int(np.argmin(np.abs(time_signals - t_end)))
    if idx_end <= idx_start: return timestamps
    timestamps["T1_start"], timestamps["T5_end"] = float(time_signals[idx_start]), float(time_signals[idx_end])
    transitions = [(float(time_signals[i]), int(point_seg_pred[i])) for i in range(idx_start + 1, idx_end + 1) if point_seg_pred[i] != point_seg_pred[i-1]]
    t2 = t3 = t4 = None
    for t, cls in transitions:
        if cls == 1 and t2 is None: t2 = t
        elif cls == 0 and t2 is not None and t3 is None: t3 = t
        elif cls == 1 and t3 is not None and t4 is None: t4 = t; break
    timestamps["T2_turn1_start"], timestamps["T3_walk2_start"], timestamps["T4_turn2_start"] = t2, t3, t4
    return timestamps

# --- ДЕТЕКЦИЯ ПИКОВ И ВПАДИН (по отфильтрованному и нормализованному сигналу) ---

def estimate_step_distance(signal):
    signal = signal - np.mean(signal)
    corr = correlate(signal, signal, mode='full')
    corr = corr[len(corr) // 2:]

    min_lag = int(FS * 0.3)
    max_lag = int(FS * 1.2)
    corr_range = corr[min_lag:max_lag]

    if len(corr_range) == 0:
        return int(FS * 0.5)

    return np.argmax(corr_range) + min_lag

def detect_peaks_advanced(signal):
    """Детекция максимумов (пиков) на уже подготовленном сигнале."""
    step_dist = estimate_step_distance(signal)
    height = np.percentile(signal, 70)

    peaks, _ = find_peaks(
        signal,
        distance=int(step_dist * 0.45),
        height=height,
        prominence=np.std(signal) * 0.3
    )
    return peaks

def detect_troughs_advanced(signal):
    """Детекция минимумов (впадин) на уже подготовленном сигнале."""
    step_dist = estimate_step_distance(-signal)
    height = np.percentile(-signal, 70)

    troughs, _ = find_peaks(
        -signal,
        distance=int(step_dist * 0.45),
        height=height,
        prominence=np.std(-signal) * 0.3
    )
    return troughs

# --- РАСЧЕТ ПРИЗНАКОВ ---

def sample_entropy(signal, m=2, r=0.2):
    """Расчет sample entropy."""
    try:
        signal = np.array(signal, dtype=float)
        N = len(signal)
        if N < 100: return np.nan
        std_sig = np.std(signal)
        if std_sig == 0: return np.nan
        r_val = r * std_sig
        
        def _phi(m_val):
            x = np.array([signal[i:i+m_val] for i in range(N - m_val + 1)])
            C = np.sum([np.sum(np.max(np.abs(x - x_i), axis=1) <= r_val) - 1 for x_i in x])
            return C / (N - m_val + 1)
        
        phi_m = _phi(m)
        phi_m1 = _phi(m+1)
        if phi_m == 0 or phi_m1 == 0: return np.nan
        return float(-np.log(phi_m1 / phi_m))
    except Exception:
        return np.nan

def filter_for_freeze_index(signal, fs=FS, lowcut=0.5, highcut=15.0, order=4):
    """Полосовой фильтр Баттерворта 4-го порядка специально для Freeze Index (0.5–15 Гц)."""
    return butter_bandpass(signal, lowcut=lowcut, highcut=highcut, fs=fs, order=order)

def freeze_index_sliding(signal, fs=FS, window_sec=4.0, step_sec=0.5):
    """Расчет Freeze Index методом скользящего окна."""
    try:
        signal = np.array(signal, dtype=float)
        win_len = int(window_sec * fs)
        step_len = int(step_sec * fs)

        if len(signal) < win_len:
            win_len = len(signal)
            step_len = len(signal)

        fi_list = []
        for start in range(0, len(signal) - win_len + 1, step_len):
            window = signal[start : start + win_len]

            yf = np.abs(rfft(window))
            xf = rfftfreq(len(window), 1 / fs)
            power_spectrum = yf**2

            locomotor = (xf >= 0.5) & (xf <= 3.0)
            freeze = (xf >= 3.0) & (xf <= 8.0)

            power_loco = np.sum(power_spectrum[locomotor])
            power_freeze = np.sum(power_spectrum[freeze])

            if power_loco > 0:
                fi_list.append(power_freeze / power_loco)

        if not fi_list:
            return {"freeze_index_mean": np.nan, "freeze_index_max": np.nan}

        return {
            "freeze_index_mean": float(np.mean(fi_list)),
            "freeze_index_max": float(np.max(fi_list)),
        }
    except Exception:
        return {"freeze_index_mean": np.nan, "freeze_index_max": np.nan}

def compute_all_segment_features(segment):
    features = {}
    
    # 1. Предварительная фильтрация (0.5-7 Гц) и MinMax нормализация для общих признаков
    raw_x = segment['acc_x'].values if 'acc_x' in segment.columns else segment['acc_mag'].values
    prep_x = preprocess_signal_for_features(raw_x, lowcut=0.5, highcut=7.0, fs=FS, order=4)
    
    # Поиск пиков по отфильтрованному и нормализованному сигналу
    peaks = detect_peaks_advanced(prep_x)
    
    if len(peaks) >= 2:
        step_times = np.diff(peaks) / FS
        step_times = step_times[step_times > 0]
        if len(step_times) > 0:
            step_time_mean = float(np.mean(step_times))
            step_time_std = float(np.std(step_times))
            step_time_cv = step_time_std / step_time_mean if step_time_mean > 0 else np.nan
            cadence = 60.0 / step_time_mean if step_time_mean > 0 else np.nan
            n_steps = len(step_times)
        else:
            step_time_mean, step_time_std, step_time_cv, cadence, n_steps = np.nan, np.nan, np.nan, np.nan, 0
    else:
        step_time_mean, step_time_std, step_time_cv, cadence, n_steps = np.nan, np.nan, np.nan, np.nan, 0

    raw_mag = segment['acc_mag'].values if 'acc_mag' in segment.columns else raw_x
    prep_mag = preprocess_signal_for_features(raw_mag, lowcut=0.5, highcut=7.0, fs=FS, order=4)

    features.update({
        'step_time_mean': step_time_mean,
        'step_time_std': step_time_std,
        'step_time_cv': step_time_cv,
        'cadence': cadence,
        'n_steps': n_steps,
        'sample_entropy': sample_entropy(prep_mag)
    })

    # 2. ИЗОЛИРОВАННЫЙ РАСЧЕТ FREEZE INDEX: полосовой фильтр (0.5–15 Гц), БЕЗ MinMax
    x_filt_fi = filter_for_freeze_index(raw_x, fs=FS, lowcut=0.5, highcut=15.0, order=4)
    fi_dict = freeze_index_sliding(x_filt_fi, fs=FS)
    features.update(fi_dict)

    # 3. Вычисление RMS, STD, Jerk STD по отфильтрованным (0.5–7 Гц) и нормализованным осям
    for axis in ['acc_x', 'acc_y', 'acc_z', 'acc_mag']:
        if axis not in segment.columns:
            continue
        
        raw_signal = segment[axis].values
        prep_signal = preprocess_signal_for_features(raw_signal, lowcut=0.5, highcut=7.0, fs=FS, order=4)
        jerk = np.diff(prep_signal) * FS
        
        prefix = axis.replace('acc_', '').upper()
        features[f'{prefix}_rms'] = float(np.sqrt(np.mean(prep_signal**2)))
        features[f'{prefix}_std'] = float(np.std(prep_signal))
        features[f'{prefix}_jerk_std'] = float(np.std(jerk))

    return features

def classify_features(features):
    print("\n" + "="*70)
    print("DEBUG: Признаки ПЕРЕД обработкой NaN:")
    for k, v in features.items():
        print(f"  {k}: {v}")
    
    safe_defaults = {
        'cadence': 100.0, 'sample_entropy': 1.0, 'freeze_index_mean': 0.1,
        'step_time_cv': 0.05, 'X_std': 0.5, 'X_rms': 1.0, 'X_jerk_std': 10.0
    }
    
    feature_vec = []
    for name in FEATURE_NAMES:
        val = features.get(name)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            print(f"  ⚠️ ВНИМАНИЕ: {name} был NaN/None, заменен на {safe_defaults[name]}")
            val = safe_defaults[name]
        feature_vec.append(float(val))
        
    print("DEBUG: Итоговый вектор, который получит модель:", feature_vec)
    print("="*70 + "\n")
        
    feature_vec = np.array(feature_vec, dtype=np.float64).reshape(1, -1)
    scaled = model_manager.clf_scaler.transform(feature_vec)
    prediction = int(model_manager.clf_model.predict(scaled)[0])
    probs = model_manager.clf_model.predict_proba(scaled)[0]
    
    print(f"DEBUG: Prediction={prediction}, Probabilities={probs}\n")
    
    return {"prediction": prediction, "probability": float(probs[prediction])}

def _build_result(points: Dict[str, float]) -> dict:
    global _last_processed
    df_proc = _last_processed['df_proc']
    t_vals = df_proc['time_sec'].values
    
    i1, i2 = int(np.argmin(np.abs(t_vals - points["T1"]))), int(np.argmin(np.abs(t_vals - points["T2"])))
    i3, i4 = int(np.argmin(np.abs(t_vals - points["T3"]))), int(np.argmin(np.abs(t_vals - points["T4"])))
    
    min_len = 200
    if i2 - i1 < min_len: i2 = min(i1 + min_len, len(t_vals) - 1)
    if i4 - i3 < min_len: i4 = min(i3 + min_len, len(t_vals) - 1)

    seg1, seg2 = df_proc.iloc[i1:i2+1], df_proc.iloc[i3:i4+1]
    
    print(f"\n!!! DEBUG: Длина сегмента 1: {len(seg1)}, Длина сегмента 2: {len(seg2)} !!!")

    feat1 = compute_all_segment_features(seg1)
    feat2 = compute_all_segment_features(seg2)

    all_features = {}
    for k in feat1.keys():
        v1, v2 = feat1[k], feat2[k]
        all_features[k] = float((v1 + v2) / 2) if not (np.isnan(v1) or np.isnan(v2)) else (float(v1) if not np.isnan(v1) else (float(v2) if not np.isnan(v2) else np.nan))

    model_features = {k: all_features[k] for k in FEATURE_NAMES}
    result = classify_features(model_features)

    output = {
        "points": points,
        "segments": {
            "segment_1": {"start_time": float(seg1["time_sec"].iloc[0]), "end_time": float(seg1["time_sec"].iloc[-1]), "n_samples": int(len(seg1)), "time_data": seg1['time_sec'].values.tolist(), "signal_data": seg1['acc_mag'].values.tolist()},
            "segment_2": {"start_time": float(seg2["time_sec"].iloc[0]), "end_time": float(seg2["time_sec"].iloc[-1]), "n_samples": int(len(seg2)), "time_data": seg2['time_sec'].values.tolist(), "signal_data": seg2['acc_mag'].values.tolist()}
        },
        "features": model_features,
        "all_features": all_features,
        "prediction": result["prediction"],
        "probability": result["probability"],
        "full_signal": {"time": df_proc['time_sec'].values.tolist(), "magnitude": df_proc['acc_mag'].values.tolist()}
    }

    if model_manager.feature_bounds:
        output["feature_norm"] = check_feature_norm(model_features, model_manager.feature_bounds)
    return to_python_types(output)

def process_signal(df: pd.DataFrame) -> dict:
    global _last_processed
    data_mat, time_array, df_proc = resample_and_prep(df)
    _last_processed['df_proc'] = df_proc

    X_det, T_det = create_windows(model_manager.detector_scaler.transform(data_mat), time_array, WIN_SIZE_DET, STEP_SIZE_DET)
    with torch.no_grad():
        det_preds = smooth_predictions(torch.argmax(model_manager.detector_model(torch.tensor(X_det, dtype=torch.float32)), dim=1).cpu().numpy(), k=7)
    
    main_seg = find_true_movement_boundaries(det_preds, min_duration_sec=1.5)
    if main_seg is None:
        raise ValueError("Движение не обнаружено или сегмент слишком короткий")
    det_preds[main_seg[0]:main_seg[1]] = 1

    seg_pred_resampled = np.zeros(len(T_det), dtype=int)
    st_time, en_time = T_det[main_seg[0]], T_det[min(main_seg[1] - 1, len(T_det) - 1)] + WIN_SIZE_DET
    st_idx, en_idx = int(np.searchsorted(time_array, st_time)), min(len(time_array), int(np.searchsorted(time_array, en_time)))
    
    if en_idx > st_idx:
        X_seg, T_seg = create_windows(model_manager.segmenter_scaler.transform(data_mat[st_idx:en_idx]), time_array[st_idx:en_idx], WIN_SIZE_SEG, STEP_SIZE_SEG)
        if len(X_seg) > 0:
            with torch.no_grad():
                seg_preds = smooth_predictions(torch.argmax(model_manager.segmenter_model(torch.tensor(X_seg, dtype=torch.float32)), dim=1).cpu().numpy(), k=5)
            for i, val in enumerate(seg_preds):
                det_idx = int(np.argmin(np.abs(T_det - T_seg[i])))
                if main_seg[0] <= det_idx < main_seg[1]:
                    seg_pred_resampled[det_idx] = val

    pts = extract_precise_timestamps(df_proc["time_sec"].values, convert_windows_to_points(df_proc, T_det, seg_pred_resampled, det_preds, main_seg), det_preds, main_seg, T_det)
    points = {
        "T1": pts["T1_start"] or float(T_det[main_seg[0]]),
        "T5": pts["T5_end"] or float(T_det[main_seg[1]-1]),
        "T2": pts["T2_turn1_start"], "T3": pts["T3_walk2_start"], "T4": pts["T4_turn2_start"]
    }
    total = points["T5"] - points["T1"]
    points["T2"] = points["T2"] or (points["T1"] + total * 0.25)
    points["T3"] = points["T3"] or (points["T1"] + total * 0.50)
    points["T4"] = points["T4"] or (points["T1"] + total * 0.75)

    return _build_result(points)

def recalculate_with_points(points: Dict[str, float]) -> dict:
    if _last_processed['df_proc'] is None:
        raise ValueError("Нет данных для пересчёта. Сначала запустите полный анализ.")
    return _build_result(points)

def check_feature_norm(features: Dict[str, float], bounds: Dict[str, Dict]) -> Dict[str, Dict]:
    norm_check = {}
    for feature, value in features.items():
        if feature not in bounds:
            norm_check[feature] = {"value": value, "in_norm": None, "message": "Границы не определены"}
            continue
        b = bounds[feature]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            norm_check[feature] = {"value": None, "in_norm": None, "message": "Не рассчитано"}
        elif b["lower_bound"] <= value <= b["upper_bound"]:
            norm_check[feature] = {"value": value, "in_norm": True, "lower_bound": b["lower_bound"], "upper_bound": b["upper_bound"], "message": "В норме ✓"}
        else:
            dev = f"Ниже на {b['lower_bound'] - value:.4f}" if value < b["lower_bound"] else f"Выше на {value - b['upper_bound']:.4f}"
            norm_check[feature] = {"value": value, "in_norm": False, "lower_bound": b["lower_bound"], "upper_bound": b["upper_bound"], "message": f"Отклонение: {dev}"}
    return norm_check

def to_python_types(obj: Any) -> Any:
    if isinstance(obj, dict): return {k: to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_python_types(v) for v in obj]
    if isinstance(obj, np.ndarray): return to_python_types(obj.tolist())
    if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
    if isinstance(obj, float) and np.isnan(obj): return None
    return obj