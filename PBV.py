import time
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import cv2
from PIL import Image, ImageTk
from scipy.signal import butter, filtfilt, detrend, welch


# Camera config 

CAM_SRC = 0
CAM_BACKEND = cv2.CAP_DSHOW      
CAM_FOURCC = "MJPG"              

CAP_WIDTH = 640
CAP_HEIGHT = 480
CAP_FPS = 30

CAP_WARMUP_READS = 20
CAP_OPEN_RETRY_DELAY = 0.7
CAP_READ_FAIL_TOL = 15
CAP_FRAME_TIMEOUT = 2.0


# Processing / UI

UI_TICK_MS = 10
PROC_TARGET_FPS = 30.0


# rPPG windows 

WINDOW_FAST_SECONDS = 5.0
WINDOW_STABLE_SECONDS = 8.0
MAX_HISTORY_SECONDS = 12.0

TARGET_RESAMPLE_FPS_MAX = 60.0

BPM_MIN = 45
BPM_MAX = 210

BPM_EMA_ALPHA = 0.18
MEDIAN_K = 7

CONF_MIN_STABLE = 1.20
CONF_MIN_FAST = 0.95
FAST_BLEND_CONF = 1.35

MAX_BPM_STEP_PER_SEC_NORMAL = 4.0
MAX_BPM_STEP_PER_SEC_FAST = 22.0
FAST_JUMP_SECONDS_AFTER_WARMUP = 10.0
FAST_JUMP_CONF_THRESHOLD = 1.25

LOWCUT = 0.7
HIGHCUT = 4.0  # 240 bpm

WARMUP_SECONDS = 5.0


# Gesicht / ROI

FACE_DETECT_EVERY_N_FRAMES = 2
ROI_SMOOTH_ALPHA = 0.18
FACE_SCALE = 1.08

# Stirn ROI 
FOREHEAD_X = 0.18
FOREHEAD_Y = 0.10
FOREHEAD_W = 0.64
FOREHEAD_H = 0.22

# Wangen ROIs 
CHEEK_Y = 0.52
CHEEK_H = 0.22
CHEEK_W = 0.26
CHEEK_INSET_X = 0.08  

# Pixel masking
USE_BRIGHTNESS_REJECT = True
V_MIN = 35
V_MAX = 235

# Haut mask 
USE_SKIN_MASK = True
CR_MIN, CR_MAX = 135, 180
CB_MIN, CB_MAX = 85, 135

MIN_GOOD_PIXEL_RATIO = 0.08  # nicht genug gut Pixles = nicht annehmen


# Motion (um Bewegungs Rauschen zu vermeiden)
MOTION_FREEZE_THRESHOLD = 9.0
MOTION_SMOOTH_ALPHA = 0.25



# Signal utils (Bandpass filter)
def butter_bandpass(sig, fs, lowcut=LOWCUT, highcut=HIGHCUT, order=4):
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < 20 or fs <= 0:
        return sig
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999)
    if not (0 < low < high < 1):
        return sig
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, sig)


def estimate_fs_from_timestamps(ts):
    ts = np.asarray(ts, dtype=np.float64)
    if len(ts) < 10:
        return None
    d = np.diff(ts)
    d = d[(d > 1e-4) & (d < 0.5)]
    if len(d) < 5:
        return None
    med = float(np.median(d))
    if med <= 0:
        return None
    return 1.0 / med


def resample_uniform(ts, x, fs_target):
    ts = np.asarray(ts, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if len(ts) < 5:
        return None
    t0, t1 = ts[0], ts[-1]
    if t1 <= t0:
        return None
    t_u = np.arange(t0, t1, 1.0 / fs_target)
    if len(t_u) < 25:
        return None
    return np.interp(t_u, ts, x)


def estimate_hr_with_conf_harmonic(sig, fs, prev_bpm=None, lowcut=LOWCUT, highcut=HIGHCUT):
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < int(2.5 * fs) or fs <= 0:
        return None, 0.0

    nperseg = min(256, len(sig))
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg)
    band = (f >= lowcut) & (f <= highcut)
    if not np.any(band):
        return None, 0.0

    fb = f[band]
    Pb = Pxx[band]
    if len(Pb) < 5:
        return None, 0.0

    i1 = int(np.argmax(Pb))
    peak = float(Pb[i1])

    Pb2 = Pb.copy()
    guard = 2
    lo = max(0, i1 - guard)
    hi = min(len(Pb2), i1 + guard + 1)
    Pb2[lo:hi] = 0
    second = float(np.max(Pb2)) if len(Pb2) else 0.0
    conf = peak / (second + 1e-12)

    f0 = float(fb[i1])
    p0 = peak
    best_f = f0

    prev_f = (float(prev_bpm) / 60.0) if prev_bpm is not None else None

    f2 = 2.0 * f0
    if lowcut <= f2 <= highcut:
        j = int(np.argmin(np.abs(fb - f2)))
        p2 = float(Pb[j])

        if prev_f is None:
            if p2 > 0.72 * p0:
                best_f = float(fb[j])
        else:
            sigma = 0.33
            s0 = np.log(p0 + 1e-12) - 0.5 * ((f0 - prev_f) ** 2) / (sigma ** 2)
            s2 = np.log(p2 + 1e-12) - 0.5 * ((float(fb[j]) - prev_f) ** 2) / (sigma ** 2)
            s2 -= 0.04
            if s2 > s0:
                best_f = float(fb[j])

    return float(best_f * 60.0), float(conf)



# rPPG algorithms
#POS 
def pos_algorithm(rgb):
    X = rgb.astype(np.float64)
    m = np.mean(X, axis=0)
    Xn = X / (m + 1e-9) - 1.0
    S1 = Xn[:, 1] - Xn[:, 2]
    S2 = Xn[:, 1] + Xn[:, 2] - 2 * Xn[:, 0]
    return (S1 / (np.std(S1) + 1e-9)) + (S2 / (np.std(S2) + 1e-9))

#PBV
def pbv_algorithm(rgb):
    Pbv = np.array([0.33, 0.77, 0.53], dtype=np.float64)
    Pbv = Pbv / np.linalg.norm(Pbv)

    X = rgb.astype(np.float64)
    m = np.mean(X, axis=0)
    Cn = X / (m + 1e-9) - 1.0
    Cn = Cn.T

    R = np.cov(Cn)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        return pos_algorithm(rgb)

    W = np.dot(R_inv, Pbv)
    denom = np.sqrt(np.dot(W.T, np.dot(R, W)))
    if denom > 1e-9:
        W = W / denom

    S = np.dot(W.T, Cn)
    return S

#Green Channel
def green_algorithm(rgb):
    X = rgb.astype(np.float64)
    g = X[:, 1]
    mg = np.mean(g)
    return (g / (mg + 1e-9)) - 1.0



# Masked mean RGB + motion
def mean_rgb_masked(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return None, 0.0, 0

    H, W = roi_bgr.shape[:2]
    total = H * W

    mask = np.ones((H, W), dtype=np.uint8) * 255

    if USE_BRIGHTNESS_REJECT:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        m_v = ((v > V_MIN) & (v < V_MAX)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, m_v)

    if USE_SKIN_MASK:
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        m_skin = ((cr >= CR_MIN) & (cr <= CR_MAX) & (cb >= CB_MIN) & (cb <= CB_MAX)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, m_skin)

    good = int(np.count_nonzero(mask))
    ratio = good / max(1, total)

    if ratio < MIN_GOOD_PIXEL_RATIO:
        return None, float(ratio), good

    b, g, r, _ = cv2.mean(roi_bgr, mask=mask)
    return (float(r), float(g), float(b)), float(ratio), good


def roi_motion_score(prev_gray_small, gray_small):
    if prev_gray_small is None or gray_small is None:
        return 0.0
    d = cv2.absdiff(prev_gray_small, gray_small)
    return float(np.mean(d))


# Video 
class VideoThread(threading.Thread):
    def __init__(self, src=CAM_SRC):
        super().__init__(daemon=True)
        self.src = src
        self.running = True

        self.lock = threading.Lock()
        self.latest = None

        self._last_capture_t = None
        self._fps_ema = None
        self.cam_fps = 0.0

        self._cap = None
        self._consecutive_fail = 0

        self.actual_w = 0
        self.actual_h = 0
        self.actual_fps_prop = 0.0

    def _open_camera(self):
        cap = cv2.VideoCapture(self.src, CAM_BACKEND)
        if not cap.isOpened():
            cap.release()
            return None

        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAM_FOURCC))
        except Exception:
            pass

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(CAP_WIDTH))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CAP_HEIGHT))
        cap.set(cv2.CAP_PROP_FPS, float(CAP_FPS))

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        ok_any = False
        for _ in range(CAP_WARMUP_READS):
            ok, frame = cap.read()
            if ok and frame is not None:
                ok_any = True
                break
            time.sleep(0.02)

        if not ok_any:
            cap.release()
            return None

        self.actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps_prop = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        return cap

    def _close_camera(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    def run(self):
        while self.running:
            if self._cap is None or not self._cap.isOpened():
                self._close_camera()
                time.sleep(CAP_OPEN_RETRY_DELAY)
                self._cap = self._open_camera()
                self._last_capture_t = None
                self._fps_ema = None
                self.cam_fps = 0.0
                self._consecutive_fail = 0
                continue

            try:
                ret, frame = self._cap.read()
            except Exception:
                ret, frame = False, None

            now = time.time()

            if not ret or frame is None:
                self._consecutive_fail += 1
                if self._consecutive_fail >= CAP_READ_FAIL_TOL:
                    self._close_camera()
                time.sleep(0.01)
                continue

            self._consecutive_fail = 0

            if self._last_capture_t is not None:
                dt = now - self._last_capture_t
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps_ema = inst if self._fps_ema is None else 0.9 * self._fps_ema + 0.1 * inst
                    self.cam_fps = float(self._fps_ema)
            self._last_capture_t = now

            with self.lock:
                self.latest = (now, frame)

        self._close_camera()

    def read_latest(self):
        with self.lock:
            return self.latest

    def last_frame_age(self):
        with self.lock:
            if self.latest is None:
                return 1e9
            t, _ = self.latest
        return time.time() - t

    def stop(self):
        self.running = False
        self._close_camera()



# State
class AlgoState:
    def __init__(self):
        self.bpm_display = None
        self.bpm_candidates = []
        self.last_bpm_time = None

    def reset(self):
        self.bpm_display = None
        self.bpm_candidates.clear()
        self.last_bpm_time = None



# Main App

class RPPGApp:
    def __init__(self, master):
        self.master = master
        self.master.title("rPPG – Forehead+Cheeks (Skin Mask Fusion) – POS vs PBV vs GREEN")
        self.master.geometry("1020x680")
        self.master.resizable(False, False)

        self.PW, self.PH = 560, 420

        self.frame_count = 0

        # Smoothed face rect (x,y,w,h) in frame coords
        self.face_rect = None

        # Histories of fused RGB
        self.rgb_hist = []
        self.t_hist = []

        self.start_time = None
        self.pos_state = AlgoState()
        self.pbv_state = AlgoState()
        self.green_state = AlgoState()

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.running = False
        self._next_proc_time = 0.0

        self.canvas_img_id = None
        self._tkimg_ref = None

        # motion gating
        self._prev_face_gray_small = None
        self.motion_ema = 0.0

        # ROI qualität
        self.skin_var = tk.StringVar(value="Skin%: --")

        self._build_ui()

    def _build_ui(self):
        ttk.Button(self.master, text="Start", command=self.start).place(x=20, y=20)
        ttk.Button(self.master, text="Stop", command=self.stop).place(x=110, y=20)

        self.canvas = tk.Canvas(self.master, width=self.PW, height=self.PH, bg="black")
        self.canvas.place(x=20, y=70)

        self.pos_bpm_var = tk.StringVar(value="POS: -- BPM")
        self.pbv_bpm_var = tk.StringVar(value="PBV: -- BPM")
        self.green_bpm_var = tk.StringVar(value="GREEN: -- BPM")

        tk.Label(self.master, textvariable=self.pos_bpm_var, font=("Arial", 26)).place(x=610, y=70)
        tk.Label(self.master, textvariable=self.pbv_bpm_var, font=("Arial", 26)).place(x=610, y=120)
        tk.Label(self.master, textvariable=self.green_bpm_var, font=("Arial", 26)).place(x=610, y=170)

        self.pos_qual_var = tk.StringVar(value="POS Quality: --")
        self.pbv_qual_var = tk.StringVar(value="PBV Quality: --")
        self.green_qual_var = tk.StringVar(value="GREEN Quality: --")

        tk.Label(self.master, textvariable=self.pos_qual_var, font=("Arial", 12)).place(x=610, y=230)
        tk.Label(self.master, textvariable=self.pbv_qual_var, font=("Arial", 12)).place(x=610, y=255)
        tk.Label(self.master, textvariable=self.green_qual_var, font=("Arial", 12)).place(x=610, y=280)

        self.fs_var = tk.StringVar(value="Signal FS: -- Hz")
        tk.Label(self.master, textvariable=self.fs_var, font=("Arial", 12)).place(x=610, y=310)

        self.caminfo_var = tk.StringVar(value="Cam: --")
        tk.Label(self.master, textvariable=self.caminfo_var, font=("Arial", 12)).place(x=610, y=335)

        self.motion_var = tk.StringVar(value="Motion: --")
        tk.Label(self.master, textvariable=self.motion_var, font=("Arial", 12)).place(x=610, y=360)

        tk.Label(self.master, textvariable=self.skin_var, font=("Arial", 12)).place(x=610, y=385)

        self.warmup_var = tk.StringVar(value="")
        tk.Label(self.master, textvariable=self.warmup_var, font=("Arial", 12), fg="blue").place(x=20, y=505)

        self.status_var = tk.StringVar(value="")
        tk.Label(self.master, textvariable=self.status_var, font=("Arial", 12), fg="red").place(x=20, y=535)

    def start(self):
        self.vt = VideoThread(CAM_SRC)
        self.vt.start()

        self.running = True
        self.frame_count = 0
        self.face_rect = None

        self.rgb_hist.clear()
        self.t_hist.clear()

        self.start_time = time.time()
        self.pos_state.reset()
        self.pbv_state.reset()
        self.green_state.reset()

        self._prev_face_gray_small = None
        self.motion_ema = 0.0

        self.pos_bpm_var.set("POS: -- BPM")
        self.pbv_bpm_var.set("PBV: -- BPM")
        self.green_bpm_var.set("GREEN: -- BPM")

        self.pos_qual_var.set("POS Qualität: --")
        self.pbv_qual_var.set("PBV Qualität: --")
        self.green_qual_var.set("GREEN Qualität: --")

        self.fs_var.set("Signal FS: -- Hz")
        self.skin_var.set("Haut%: --")
        self.warmup_var.set("Vorbereitung...")

        self._next_proc_time = 0.0
        self.master.after(UI_TICK_MS, self.update)

    def stop(self):
        self.running = False
        if hasattr(self, "vt"):
            self.vt.stop()

    def update(self):
        if not self.running:
            return

        age = self.vt.last_frame_age()
        if age > CAP_FRAME_TIMEOUT:
            self.status_var.set("Camera wird geöffnet")
        else:
            self.status_var.set("")

        item = self.vt.read_latest()
        if item is not None:
            t, frame = item

            self.caminfo_var.set(
                f"Cam FPS: {self.vt.cam_fps:.1f} | "
            )

            elapsed = time.time() - (self.start_time or time.time())
            if elapsed < WARMUP_SECONDS:
                self.warmup_var.set(f"Vorbereitung... {WARMUP_SECONDS - elapsed:.1f}s")
            else:
                self.warmup_var.set("")

            self.frame_count += 1
            if self.face_rect is None or (self.frame_count % FACE_DETECT_EVERY_N_FRAMES) == 0:
                self._update_face_rect(frame)

            self.show_frame(frame)

            if t >= self._next_proc_time:
                self._next_proc_time = t + (1.0 / PROC_TARGET_FPS)
                self.process_frame(frame, t, elapsed)

        self.master.after(UI_TICK_MS, self.update)

    def _update_face_rect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(110, 110))
        if len(faces) == 0:
            return

        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])

        cx, cy = fx + fw / 2, fy + fh / 2
        fw2, fh2 = fw * FACE_SCALE, fh * FACE_SCALE
        fx2 = int(cx - fw2 / 2)
        fy2 = int(cy - fh2 / 2)
        fw2, fh2 = int(fw2), int(fh2)

        H, W = frame.shape[:2]
        fx2 = max(0, min(fx2, W - 1))
        fy2 = max(0, min(fy2, H - 1))
        fw2 = max(1, min(fw2, W - fx2))
        fh2 = max(1, min(fh2, H - fy2))

        new = np.array([fx2, fy2, fw2, fh2], dtype=np.float64)
        if self.face_rect is None:
            sm = new
        else:
            old = np.array(self.face_rect, dtype=np.float64)
            sm = (1 - ROI_SMOOTH_ALPHA) * old + ROI_SMOOTH_ALPHA * new
        self.face_rect = [int(sm[0]), int(sm[1]), int(sm[2]), int(sm[3])]

    def _rect_clip(self, x, y, w, h, W, H):
        x = max(0, min(int(x), W - 1))
        y = max(0, min(int(y), H - 1))
        w = max(1, min(int(w), W - x))
        h = max(1, min(int(h), H - y))
        return x, y, w, h

    def _get_rois(self, frame):
        if self.face_rect is None:
            return None

        fx, fy, fw, fh = self.face_rect
        H, W = frame.shape[:2]
        #Stirn Parameters
        rx_f = fx + fw * FOREHEAD_X
        ry_f = fy + fh * FOREHEAD_Y
        rw_f = fw * FOREHEAD_W
        rh_f = fh * FOREHEAD_H
        forehead = self._rect_clip(rx_f, ry_f, rw_f, rh_f, W, H)

        ry_c = fy + fh * CHEEK_Y
        rh_c = fh * CHEEK_H
        rw_c = fh * 0 + fw * CHEEK_W  # Wangen Parameters

        lx = fx + fw * CHEEK_INSET_X
        left_cheek = self._rect_clip(lx, ry_c, rw_c, rh_c, W, H)

        rx = fx + fw - fw * CHEEK_INSET_X - rw_c
        right_cheek = self._rect_clip(rx, ry_c, rw_c, rh_c, W, H)

        return forehead, left_cheek, right_cheek

    def _compute_bpm_window(self, rgb, ts, algo_name, prev_bpm, fs_target):
        if algo_name == "PBV":
            raw = pbv_algorithm(rgb)
        elif algo_name == "POS":
            raw = pos_algorithm(rgb)
        else:
            raw = green_algorithm(rgb)

        raw_u = resample_uniform(ts, raw, fs_target=fs_target)
        if raw_u is None:
            return None, 0.0

        sig = detrend(raw_u)
        sig = butter_bandpass(sig, fs=fs_target)

        bpm, conf = estimate_hr_with_conf_harmonic(sig, fs=fs_target, prev_bpm=prev_bpm)
        if bpm is None or not np.isfinite(bpm):
            return None, float(conf)
        if not (BPM_MIN <= bpm <= BPM_MAX):
            return None, float(conf)
        return float(bpm), float(conf)

    def _update_state(self, state: AlgoState, bpm, conf, elapsed, t_now):
        seconds_after_warmup = max(0.0, elapsed - WARMUP_SECONDS)
        max_step_rate = MAX_BPM_STEP_PER_SEC_NORMAL
        if seconds_after_warmup < FAST_JUMP_SECONDS_AFTER_WARMUP and conf >= FAST_JUMP_CONF_THRESHOLD:
            max_step_rate = MAX_BPM_STEP_PER_SEC_FAST

        if state.bpm_display is None:
            bpm_limited = float(bpm)
            state.last_bpm_time = t_now
        else:
            dt = max(1e-3, t_now - (state.last_bpm_time or t_now))
            max_step = max_step_rate * dt
            bpm_limited = float(np.clip(bpm, state.bpm_display - max_step, state.bpm_display + max_step))
            state.last_bpm_time = t_now

        state.bpm_candidates.append(bpm_limited)
        if len(state.bpm_candidates) > MEDIAN_K:
            state.bpm_candidates.pop(0)
        bpm_med = float(np.median(state.bpm_candidates))

        if state.bpm_display is None:
            state.bpm_display = bpm_med
        else:
            state.bpm_display = (1 - BPM_EMA_ALPHA) * state.bpm_display + BPM_EMA_ALPHA * bpm_med

        return float(state.bpm_display)

    def _compute_algo(self, rgb_fast, ts_fast, rgb_stable, ts_stable,
                      algo_name, state: AlgoState, fs_target, elapsed,
                      motion_freeze, t_now):
        prev = state.bpm_display

        bpm_s, conf_s = self._compute_bpm_window(rgb_stable, ts_stable, algo_name, prev, fs_target)
        bpm_f, conf_f = self._compute_bpm_window(rgb_fast, ts_fast, algo_name, prev, fs_target)

        conf_out = max(conf_s, conf_f)

        if motion_freeze:
            return None, conf_out

        ok_s = (bpm_s is not None and conf_s >= CONF_MIN_STABLE)
        ok_f = (bpm_f is not None and conf_f >= CONF_MIN_FAST)

        if ok_s and ok_f:
            w_fast = 0.35
            if conf_f >= FAST_BLEND_CONF:
                w_fast = 0.60
            bpm_out = (1 - w_fast) * bpm_s + w_fast * bpm_f
        elif ok_f:
            bpm_out = bpm_f
        elif ok_s:
            bpm_out = bpm_s
        else:
            return None, conf_out

        bpm_disp = self._update_state(state, bpm_out, conf_out, elapsed, t_now)
        return bpm_disp, conf_out

    def process_frame(self, frame, tstamp, elapsed):
        rois = self._get_rois(frame)
        if rois is None:
            return

        forehead, left_cheek, right_cheek = rois

        #Tracking der gesicht
        fx, fy, fw, fh = self.face_rect
        face_crop = frame[fy:fy+fh, fx:fx+fw]
        if face_crop.size > 0:
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_gray_small = cv2.resize(face_gray, (96, 96), interpolation=cv2.INTER_AREA)
            mot = roi_motion_score(self._prev_face_gray_small, face_gray_small)
            self._prev_face_gray_small = face_gray_small
            self.motion_ema = (1 - MOTION_SMOOTH_ALPHA) * self.motion_ema + MOTION_SMOOTH_ALPHA * mot
        else:
            self.motion_ema *= (1 - MOTION_SMOOTH_ALPHA)

        motion_freeze = (self.motion_ema >= MOTION_FREEZE_THRESHOLD)
        self.motion_var.set(f"Motion: {self.motion_ema:.1f} (freeze>{MOTION_FREEZE_THRESHOLD})")

        # ROI
        fused_sum = np.zeros(3, dtype=np.float64)
        fused_w = 0.0
        skin_ratios = []

        for (x, y, w, h) in [forehead, left_cheek, right_cheek]:
            roi = frame[y:y+h, x:x+w]
            rgb, ratio, good = mean_rgb_masked(roi)
            skin_ratios.append(ratio)
            if rgb is None:
                continue
            fused_sum += np.array(rgb, dtype=np.float64) * float(good)
            fused_w += float(good)

        if fused_w <= 0:
            self.skin_var.set("Skin%: sehr gering")
            return

        fused_rgb = fused_sum / fused_w
        skin_avg = float(np.mean(skin_ratios)) if skin_ratios else 0.0
        self.skin_var.set(f"Skin% (avg): {100*skin_avg:.1f}%")

     
        self.rgb_hist.append(fused_rgb.tolist())
        self.t_hist.append(tstamp)

        cutoff = tstamp - MAX_HISTORY_SECONDS
        while self.t_hist and self.t_hist[0] < cutoff:
            self.t_hist.pop(0)
            self.rgb_hist.pop(0)

        if len(self.t_hist) < 60:
            return

        fs_in = estimate_fs_from_timestamps(self.t_hist)
        if fs_in is None:
            return
        fs_target = float(min(TARGET_RESAMPLE_FPS_MAX, max(12.0, fs_in)))
        self.fs_var.set(f"Signal FS: {fs_target:.1f} Hz")

        if (self.t_hist[-1] - self.t_hist[0]) < WINDOW_STABLE_SECONDS:
            return

        t_end = self.t_hist[-1]
        t0_stable = t_end - WINDOW_STABLE_SECONDS
        t0_fast = t_end - WINDOW_FAST_SECONDS

        rgb_arr = np.array(self.rgb_hist, dtype=np.float64)
        ts_arr = np.array(self.t_hist, dtype=np.float64)

        idx_stable = np.searchsorted(ts_arr, t0_stable, side="left")
        idx_fast = np.searchsorted(ts_arr, t0_fast, side="left")

        rgb_stable = rgb_arr[idx_stable:]
        ts_stable = ts_arr[idx_stable:]
        rgb_fast = rgb_arr[idx_fast:]
        ts_fast = ts_arr[idx_fast:]

        pos_bpm, pos_conf = self._compute_algo(
            rgb_fast, ts_fast, rgb_stable, ts_stable,
            "POS", self.pos_state, fs_target, elapsed, motion_freeze, t_end
        )
        pbv_bpm, pbv_conf = self._compute_algo(
            rgb_fast, ts_fast, rgb_stable, ts_stable,
            "PBV", self.pbv_state, fs_target, elapsed, motion_freeze, t_end
        )
        green_bpm, green_conf = self._compute_algo(
            rgb_fast, ts_fast, rgb_stable, ts_stable,
            "GREEN", self.green_state, fs_target, elapsed, motion_freeze, t_end
        )

        self.pos_qual_var.set(f"POS Qualität: {pos_conf:.2f}")
        self.pbv_qual_var.set(f"PBV Qualität: {pbv_conf:.2f}")
        self.green_qual_var.set(f"GREEN Qualität: {green_conf:.2f}")

        if elapsed < WARMUP_SECONDS:
            return

        if pos_bpm is not None:
            self.pos_bpm_var.set(f"POS: {int(round(pos_bpm))} BPM")
        if pbv_bpm is not None:
            self.pbv_bpm_var.set(f"PBV: {int(round(pbv_bpm))} BPM")
        if green_bpm is not None:
            self.green_bpm_var.set(f"GREEN: {int(round(green_bpm))} BPM")

    def show_frame(self, frame):
        disp = frame.copy()
        rois = self._get_rois(frame)

        if rois is not None:
            for (x, y, w, h) in rois:
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 255), 2)

        img = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.PW, self.PH), interpolation=cv2.INTER_AREA)

        self._tkimg_ref = ImageTk.PhotoImage(Image.fromarray(img))
        if self.canvas_img_id is None:
            self.canvas_img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._tkimg_ref)
        else:
            self.canvas.itemconfig(self.canvas_img_id, image=self._tkimg_ref)


# Run
if __name__ == "__main__":
    root = tk.Tk()
    app = RPPGApp(root)
    root.mainloop()
