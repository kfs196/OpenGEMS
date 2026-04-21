import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class FEVTConfig:
    fs: float
    transform: str = "NEO"          # ND, ASD, NEO, DEO4
    smoothing_sec: float = 2.0       # paper-optimal for ASD/NEO/DEO4 on mixed data
    threshold_multiplier: float = 4.0
    refractory_sec: float = 7.0
    running_half_width_sec: float = 15.0
    edge_kernel_sec: float = 1.0
    edge_kernel_points: Optional[int] = None
    min_event_amplitude_uv: Optional[float] = None
    is_prefilter: bool = True


class FEVTDetector:
    """
    Falling-Edge, Variable Threshold (FEVT) detector.

    Paper correspondence:
    - transform_signal()      -> Eq. (1), (2), (4), (6)
    - smooth_signal()         -> Eq. (7), (8)
    - edge_detector_signal()  -> Eq. (14)
    - fevt_signal()           -> Eq. (15)
    - variable_threshold()    -> Eq. (16) with F_thresh = eta * sigma_hat(t)
    - cluster_candidates()    -> Eq. (12)
    - choose_activation_times()-> Eq. (13)
    """

    def __init__(self, config: FEVTConfig):
        self.cfg = config
        self.fs = float(config.fs)
        self.transform = config.transform.upper()
        if self.transform not in {"ND", "ASD", "NEO", "DEO4"}:
            raise ValueError("transform must be one of: ND, ASD, NEO, DEO4")

    # ------------------------- preprocessing helpers -------------------------
    # @staticmethod
    # def load_csv_signal(path: str) -> Tuple[np.ndarray, np.ndarray]:
    #     arr = np.loadtxt(path, delimiter=",", ndmin=2)
    #     if arr.shape[1] < 2:
    #         raise ValueError("CSV must have at least two columns: time, signal")
    #     return arr[:, 0], arr[:, 1]
    
    @staticmethod
    def load_csv_signal(file_obj:object) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.loadtxt(file_obj, delimiter=",", ndmin=2)        
        if arr.shape[1] < 2:
            raise ValueError("CSV must have at least two columns: time, signal")
        
        return arr[:, 0], arr[:, 1]

    @staticmethod
    def bandpass_filter(x: np.ndarray, fs: float, low_hz: float = 1.0 / 60.0, high_hz: float = 1.0,
                        order: int = 2) -> np.ndarray:
        nyq = fs / 2.0
        b, a = signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass")
        return signal.filtfilt(b, a, x)
    
    @staticmethod
    def infer_fs_from_time(t: np.ndarray, fallback_fs: float) -> float:
        dt = np.diff(t)
        if len(dt) == 0:
            return fallback_fs
        return 1.0 / np.median(dt)

    # ------------------------- paper transform block -------------------------
    def derivative(self, v: np.ndarray) -> np.ndarray:
        # 3-point central difference approximation used in the paper.
        dv = np.zeros_like(v, dtype=float)
        dv[1:-1] = (v[2:] - v[:-2]) / (2.0 / self.fs)
        dv[0] = dv[1]
        dv[-1] = dv[-2]
        return dv

    def transform_signal(self, v: np.ndarray) -> np.ndarray:
        dv = self.derivative(v)

        if self.transform == "ND":
            # Eq. (1): keep only negative derivative and invert sign
            return np.where(dv <= 0.0, -dv, 0.0)

        if self.transform == "ASD":
            # Eq. (2)
            return np.abs(v) * np.abs(v * dv)

        if self.transform == "NEO":
            # Eq. (4)
            x = np.zeros_like(v, dtype=float)
            x[1:-1] = v[1:-1] ** 2 - v[:-2] * v[2:]
            x[0] = x[1]
            x[-1] = x[-2]
            return x

        # DEO4, Eq. (6)
        x = np.zeros_like(v, dtype=float)
        if len(v) >= 4:
            x[1:-2] = v[1:-2] * v[2:-1] - v[:-3] * v[3:]
            x[0] = x[1]
            x[-2] = x[-3]
            x[-1] = x[-2]
        return x

    def smooth_signal(self, x: np.ndarray) -> np.ndarray:
        sec = float(self.cfg.smoothing_sec)
        if sec <= 0:
            return x.copy()
        width = max(1, int(round(sec * self.fs)))
        kernel = np.ones(width, dtype=float) / width
        return np.convolve(x, kernel, mode="same") # Do Convolution

    # ------------------------- falling-edge block ----------------------------
    def edge_detector_kernel(self) -> np.ndarray:
        """
        Approximate Sezan's smoother*differencer kernel.
        We use a difference of two adjacent boxcars, which behaves as a
        signed edge detector: falling edges -> positive output.
        """
        if self.cfg.edge_kernel_points is not None:
            n = int(self.cfg.edge_kernel_points)
        else:
            n = int(round(self.cfg.edge_kernel_sec * self.fs))
        n = max(4, n)
        half = n // 2
        left = -np.linspace(0, 1, half, endpoint=False)
        right = np.linspace(1, 0, max(1, (n - half)))
        kernel = np.concatenate([left, right])
        kernel -= kernel.mean() # zero-center to suppress DC leakage
        return kernel

    def edge_detector_signal(self, v: np.ndarray) -> np.ndarray:
        k = self.edge_detector_kernel()
        return np.convolve(v, k, mode="same") # # flip sign so falling edge gives positive pulse. (?)

    def fevt_signal(self, s: np.ndarray, e: np.ndarray) -> np.ndarray:
        prod = s * e # Multiply corresponding positions
        return np.where(prod >= 0.0, prod, 0.0)

    # ------------------------- threshold block -------------------------------
    @staticmethod
    def robust_sigma(x: np.ndarray) -> float:
        med = np.mean(x)
        mad = np.median(np.abs(x - med))
        return mad / 0.6745 if mad > 0 else 0.0

    def variable_threshold(self, f: np.ndarray) -> np.ndarray:
        half = max(1, int(round(self.cfg.running_half_width_sec * self.fs)))
        thr = np.zeros_like(f, dtype=float)
        for i in range(len(f)):
            lo = max(0, i - half)
            hi = min(len(f), i + half + 1)
            sigma = self.robust_sigma(f[lo:hi])
            thr[i] = self.cfg.threshold_multiplier * sigma
        return thr

    # ------------------------- event selection -------------------------------
    def cluster_candidates(self, candidate_idx: np.ndarray) -> List[np.ndarray]:
        if len(candidate_idx) == 0:
            return []
        refractory_samples = max(1, int(round(self.cfg.refractory_sec * self.fs)))
        clusters: List[List[int]] = [[int(candidate_idx[0])]]
        for idx in candidate_idx[1:]:
            if int(idx) - clusters[-1][-1] <= refractory_samples:
                clusters[-1].append(int(idx))
            else:
                clusters.append([int(idx)])
        return [np.asarray(c, dtype=int) for c in clusters]

    def choose_activation_times(self, v: np.ndarray, clusters: List[np.ndarray]) -> np.ndarray:
        dv = self.derivative(v)
        ats: List[int] = []
        for cl in clusters:
            idx = cl[np.argmin(dv[cl])]  # most negative derivative, Eq. (13)
            if self.cfg.min_event_amplitude_uv is not None:
                lo = max(0, idx - int(self.fs))
                hi = min(len(v), idx + int(self.fs))
                local_span = np.max(v[lo:hi]) - np.min(v[lo:hi])
                if local_span < self.cfg.min_event_amplitude_uv:
                    continue
            ats.append(int(idx))
        return np.asarray(ats, dtype=int)

    # ------------------------- end-to-end pipeline ---------------------------
    def detect(self, v: np.ndarray, t: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        v = np.asarray(v, dtype=float).ravel() # np.ravel(): return view
        if t is None:
            t = np.arange(len(v)) / self.fs
        else:
            t = np.asarray(t, dtype=float).ravel()

        x = self.transform_signal(v)
        s = self.smooth_signal(x)
        e = self.edge_detector_signal(v)
        f = self.fevt_signal(s, e)
        thr = self.variable_threshold(f)
        candidate_idx = np.flatnonzero(f >= thr)
        clusters = self.cluster_candidates(candidate_idx)
        at_idx = self.choose_activation_times(v, clusters)

        return {
            "time": t,
            "signal": v,
            "transform_signal": x,
            "smoothed_signal": s,
            "edge_signal": e,
            "fevt_signal": f,
            "threshold": thr,
            "candidate_idx": candidate_idx,
            "activation_idx": at_idx,
            "activation_times": t[at_idx] if len(at_idx) else np.array([]),
            "config": np.array([json.dumps(asdict(self.cfg), ensure_ascii=False)]),
        }


def save_activation_times_csv(path: str, activation_times: np.ndarray) -> None:
    data = np.column_stack([np.arange(1, len(activation_times) + 1), activation_times])
    header = "event_id,activation_time_sec"
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt=["%d", "%.6f"])


def make_demo_signal(fs: float = 30.0, duration_sec: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic single-channel intestinal/gastric-like slow-wave demo."""
    t = np.arange(0.0, duration_sec, 1.0 / fs)
    rng = np.random.default_rng(42)

    # slow baseline + mild respiratory artifact + noise
    x = 20 * np.sin(2 * np.pi * 0.03 * t) + 15 * np.sin(2 * np.pi * 0.22 * t)
    x += 6 * rng.standard_normal(len(t))

    # add several events with small upstroke, sharp negative transient, plateau
    event_times = np.array([5, 18, 31, 44, 56])
    for et in event_times:
        up = 18 * np.exp(-0.5 * ((t - (et - 0.35)) / 0.12) ** 2)
        down = -95 * np.exp(-0.5 * ((t - et) / 0.10) ** 2)
        plateau = -20 * np.exp(-np.maximum(0, t - et) / 2.0) * (t >= et)
        x += up + down + plateau
    return t, x

