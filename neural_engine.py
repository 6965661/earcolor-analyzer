import hashlib
import os
import threading
import urllib.request
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

MODEL_URL = "https://huggingface.co/musetric/chordmini-onnx/resolve/main/chordnet.onnx?download=true"
MODEL_SHA256 = "9a6570bf611cdc3f2c36286307af46fb94927fe7f6a2bc22a87c0ebf5f6c082e"
MODEL_PATH = Path(os.getenv("EARCOLOR_CHORD_MODEL", "/tmp/earcolor-chordnet.onnx"))
SR = 22050
HOP = 2048
SEQ_LEN = 108
N_BINS = 144
BINS_PER_OCTAVE = 24
SMOOTHING_KERNEL = 9
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = [
    "min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7",
    "maj7", "7", "dim7", "hdim7", "sus2", "sus4",
]
CHORD_VOCAB = [
    (f"{root}:{quality}" if quality != "maj" else root)
    for root in NOTE_NAMES
    for quality in QUALITIES
] + ["X", "N"]

_SESSION = None
_SESSION_LOCK = threading.Lock()


def _progress(cb, value, stage):
    if cb:
        cb(value, stage)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_model():
    if MODEL_PATH.exists() and _sha256(MODEL_PATH) == MODEL_SHA256:
        return str(MODEL_PATH)
    tmp = MODEL_PATH.with_suffix(".download")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    urllib.request.urlretrieve(MODEL_URL, tmp)
    digest = _sha256(tmp)
    if digest != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Chord model checksum mismatch: {digest}")
    tmp.replace(MODEL_PATH)
    return str(MODEL_PATH)


def _get_session():
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        import onnxruntime as ort
        model_path = _ensure_model()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = True
        _SESSION = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        return _SESSION


def _load_audio(path):
    # Load channels separately and average explicitly. ChordMini was trained with
    # arithmetic-mean stereo downmix, not ffmpeg's energy-preserving rematrix.
    y, sr = librosa.load(path, sr=SR, mono=False, dtype=np.float32)
    if y.ndim == 2:
        y = np.mean(y, axis=0, dtype=np.float32)
    else:
        y = np.asarray(y, dtype=np.float32)
    return y, sr


def _features(y):
    cqt = librosa.cqt(
        y=y,
        sr=SR,
        hop_length=HOP,
        fmin=32.70319566257483,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        norm=1,
        sparsity=0.01,
        window="hann",
        scale=True,
        pad_mode="constant",
    )
    feat = np.log(np.abs(cqt).T.astype(np.float32, copy=False) + 1e-6)
    return feat


def _run_model(feat):
    frame_count = feat.shape[0]
    window_count = max(1, int(np.ceil(frame_count / SEQ_LEN)))
    padded = np.zeros((window_count * SEQ_LEN, N_BINS), dtype=np.float32)
    padded[:frame_count] = feat
    windows = padded.reshape(window_count, SEQ_LEN, N_BINS)
    session = _get_session()
    logits = session.run(["logits"], {"features": windows})[0]
    logits = logits.reshape(-1, logits.shape[-1])[:frame_count]
    # Same nine-frame uniform smoothing described by the model's host runtime.
    smoothed = uniform_filter1d(logits, size=SMOOTHING_KERNEL, axis=0, mode="nearest")
    indices = np.argmax(smoothed, axis=1).astype(np.int16)
    # Confidence is only a display heuristic; argmax is the actual decision.
    centered = smoothed - np.max(smoothed, axis=1, keepdims=True)
    probs = np.exp(np.clip(centered, -30.0, 0.0))
    probs /= np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-8)
    confidence = np.max(probs, axis=1).astype(np.float32)
    return indices, confidence


def _beat_times(y, duration):
    try:
        onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=512)
        _, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset,
            sr=SR,
            hop_length=512,
            trim=False,
        )
        beats = librosa.frames_to_time(beat_frames, sr=SR, hop_length=512)
        beats = np.asarray(beats, dtype=np.float32)
        beats = beats[(beats > 0.08) & (beats < duration - 0.08)]
        if beats.size < 2:
            return np.array([0.0, duration], dtype=np.float32)
        return np.concatenate(([0.0], beats, [duration])).astype(np.float32)
    except Exception:
        return np.array([0.0, duration], dtype=np.float32)


def _dominant_label(indices, confidence, start, end):
    times = np.arange(len(indices), dtype=np.float32) * (HOP / SR)
    mask = (times >= start) & (times < end)
    ids = indices[mask]
    conf = confidence[mask]
    if ids.size == 0:
        center = int(np.clip(round(((start + end) * 0.5) * SR / HOP), 0, len(indices) - 1))
        return int(indices[center]), float(confidence[center])
    scores = np.zeros(len(CHORD_VOCAB), dtype=np.float32)
    for idx, w in zip(ids, conf):
        scores[int(idx)] += max(float(w), 0.02)
    winner = int(np.argmax(scores))
    winner_conf = float(np.mean(conf[ids == winner])) if np.any(ids == winner) else float(np.mean(conf))
    return winner, winner_conf


def _repair_unknowns(beat_labels):
    # Keep genuine silence at the edges, but prevent tiny X/N predictions from
    # making the displayed chord flicker in the middle of a song.
    out = list(beat_labels)
    for i, item in enumerate(out):
        if item["label"] not in ("X", "N"):
            continue
        prev_valid = i > 0 and out[i - 1]["label"] not in ("X", "N")
        next_valid = i + 1 < len(out) and out[i + 1]["label"] not in ("X", "N")
        span = item["end"] - item["start"]
        if span <= 1.2 and (prev_valid or next_valid):
            src = out[i - 1] if prev_valid else out[i + 1]
            item["label"] = src["label"]
    return out


def _merge_beats(beat_labels):
    segments = []
    for item in beat_labels:
        label = item["label"]
        if segments and segments[-1]["chord"] == label:
            segments[-1]["end"] = item["end"]
            segments[-1]["confidence_values"].append(item["confidence"])
        else:
            segments.append({
                "chord": label,
                "start": item["start"],
                "end": item["end"],
                "confidence_values": [item["confidence"]],
            })
    return segments


def _parse_root_quality(chord):
    if chord in ("X", "N"):
        return None, None
    root = chord.split(":", 1)[0]
    quality = chord.split(":", 1)[1] if ":" in chord else "maj"
    return NOTE_NAMES.index(root), quality


def _infer_key(segments):
    major_scale = {0, 2, 4, 5, 7, 9, 11}
    minor_scale = {0, 2, 3, 5, 7, 8, 10}
    scores = []
    for tonic in range(12):
        smaj = 0.0
        smin = 0.0
        for seg in segments:
            root, quality = _parse_root_quality(seg["chord"])
            if root is None:
                continue
            dur = max(0.05, seg["end"] - seg["start"])
            rel = (root - tonic) % 12
            if rel in major_scale:
                smaj += dur
            if rel in minor_scale:
                smin += dur
            if root == tonic:
                smaj += 0.32 * dur
                smin += 0.32 * dur
            is_minor = quality.startswith("min") or quality in ("dim", "dim7", "hdim7")
            if not is_minor and rel in (0, 5, 7):
                smaj += 0.14 * dur
            if is_minor and rel in (0, 5, 7):
                smin += 0.14 * dur
        scores.append((smaj, tonic, "major"))
        scores.append((smin, tonic, "minor"))
    if not scores:
        return "C major"
    _, tonic, mode = max(scores)
    return f"{NOTE_NAMES[tonic]} {mode}"


def _pretty(label):
    if label in ("X", "N"):
        return label
    root, quality = (label.split(":", 1) + ["maj"])[:2] if ":" in label else (label, "maj")
    suffix = {
        "maj": "",
        "min": "m",
        "dim": "dim",
        "aug": "aug",
        "min6": "m6",
        "maj6": "6",
        "min7": "m7",
        "minmaj7": "m(maj7)",
        "maj7": "maj7",
        "7": "7",
        "dim7": "dim7",
        "hdim7": "ø7",
        "sus2": "sus2",
        "sus4": "sus4",
    }.get(quality, quality)
    return root + suffix


def analyze_neural(path, progress_cb=None):
    _progress(progress_cb, 8, "Loading audio...")
    y, _ = _load_audio(path)
    if y.size < SR // 2:
        raise ValueError("Audio is too short to analyze")
    duration = float(len(y) / SR)

    _progress(progress_cb, 22, "Computing neural CQT features...")
    feat = _features(y)

    _progress(progress_cb, 48, "Running ChordMini neural model...")
    indices, confidence = _run_model(feat)

    _progress(progress_cb, 68, "Detecting beats and stabilizing chords...")
    beats = _beat_times(y, duration)
    beat_labels = []
    for i in range(len(beats) - 1):
        start = float(beats[i])
        end = float(beats[i + 1])
        idx, conf = _dominant_label(indices, confidence, start, end)
        beat_labels.append({
            "label": CHORD_VOCAB[idx],
            "start": start,
            "end": end,
            "confidence": conf,
        })
    beat_labels = _repair_unknowns(beat_labels)
    raw_segments = _merge_beats(beat_labels)

    _progress(progress_cb, 82, "Inferring key from the chord sequence...")
    key = _infer_key(raw_segments)

    final_segments = []
    progression = []
    for seg in raw_segments:
        label = _pretty(seg["chord"])
        if label in ("X", "N"):
            # Preserve silence/no-chord only when it is meaningful; the EarColor
            # display should otherwise remain on the last stable harmony.
            if seg["end"] - seg["start"] < 1.0:
                continue
        if not progression or progression[-1] != label:
            progression.append(label)
        conf = float(np.mean(seg["confidence_values"]))
        final_segments.append({
            "chord": label,
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "duration": round(float(seg["end"] - seg["start"]), 3),
            "confidence": round(conf, 3),
        })

    # Make boundaries continuous if short N/X segments were omitted.
    for i in range(1, len(final_segments)):
        if final_segments[i]["start"] > final_segments[i - 1]["end"]:
            midpoint = round((final_segments[i]["start"] + final_segments[i - 1]["end"]) / 2.0, 3)
            final_segments[i - 1]["end"] = midpoint
            final_segments[i]["start"] = midpoint
    if final_segments:
        final_segments[0]["start"] = 0.0
        final_segments[-1]["end"] = round(duration, 3)
        for seg in final_segments:
            seg["duration"] = round(seg["end"] - seg["start"], 3)

    _progress(progress_cb, 96, "Finishing neural analysis...")
    return {
        "duration_seconds": round(duration, 2),
        "key": key,
        "progression": progression,
        "chord_count": len(final_segments),
        "segments": final_segments,
        "raw_chords": None,
        "engine": "chordmini-neural-beat-v3",
    }
