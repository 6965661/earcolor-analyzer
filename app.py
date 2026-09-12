import os
import tempfile
import threading
import uuid
from pathlib import Path

import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="EarColor Analyzer", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS = {}
LOCK = threading.Lock()
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def set_job(job_id, **values):
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values)


def chord_templates():
    states = []
    templates = []
    for root in range(12):
        for quality, intervals in (("maj", (0, 4, 7)), ("min", (0, 3, 7))):
            t = np.full(12, -0.16, dtype=np.float32)
            for i, weight in zip(intervals, (1.0, 0.82, 0.70)):
                t[(root + i) % 12] = weight
            t /= np.linalg.norm(t) + 1e-8
            states.append((root, quality))
            templates.append(t)
    return states, np.stack(templates)


STATES, TEMPLATES = chord_templates()


def decode_sequence(emissions):
    n_t, n_s = emissions.shape
    dp = np.empty((n_t, n_s), dtype=np.float32)
    back = np.zeros((n_t, n_s), dtype=np.int16)
    dp[0] = emissions[0]

    trans = np.full((n_s, n_s), -0.13, dtype=np.float32)
    for i, (r1, q1) in enumerate(STATES):
        trans[i, i] = 0.32
        for j, (r2, q2) in enumerate(STATES):
            dist = (r2 - r1) % 12
            if dist in (5, 7):
                trans[i, j] = max(trans[i, j], -0.02)
            if (q1, q2, dist) in (("maj", "min", 9), ("min", "maj", 3)):
                trans[i, j] = max(trans[i, j], 0.01)

    for t in range(1, n_t):
        scores = dp[t - 1][:, None] + trans
        back[t] = np.argmax(scores, axis=0)
        dp[t] = emissions[t] + scores[back[t], np.arange(n_s)]

    seq = np.zeros(n_t, dtype=np.int16)
    seq[-1] = int(np.argmax(dp[-1]))
    for t in range(n_t - 2, -1, -1):
        seq[t] = back[t + 1, seq[t + 1]]
    return seq


def infer_key(segments):
    if not segments:
        return "C major"
    major_scale = {0, 2, 4, 5, 7, 9, 11}
    minor_scale = {0, 2, 3, 5, 7, 8, 10}
    scores = []
    for tonic in range(12):
        smaj = 0.0
        smin = 0.0
        for seg in segments:
            root = seg["root"]
            dur = max(0.05, seg["end"] - seg["start"])
            rel = (root - tonic) % 12
            if rel in major_scale:
                smaj += dur
            if rel in minor_scale:
                smin += dur
            if root == tonic:
                smaj += 0.30 * dur
                smin += 0.30 * dur
            if seg["quality"] == "maj" and rel in (0, 5, 7):
                smaj += 0.12 * dur
            if seg["quality"] == "min" and rel in (0, 5, 7):
                smin += 0.12 * dur
        scores.append((smaj, tonic, "major"))
        scores.append((smin, tonic, "minor"))
    _, tonic, mode = max(scores)
    return f"{NOTE_NAMES[tonic]} {mode}"


def merge_short_segments(segments, min_duration=0.55):
    if len(segments) < 2:
        return segments
    out = [dict(s) for s in segments]
    changed = True
    while changed and len(out) > 1:
        changed = False
        for i, seg in enumerate(out):
            if seg["end"] - seg["start"] >= min_duration:
                continue
            if i == 0:
                out[1]["start"] = seg["start"]
                out.pop(0)
            elif i == len(out) - 1:
                out[i - 1]["end"] = seg["end"]
                out.pop(i)
            else:
                left = out[i - 1]
                right = out[i + 1]
                target = left if (left["end"] - left["start"]) >= (right["end"] - right["start"]) else right
                if target is left:
                    left["end"] = seg["end"]
                else:
                    right["start"] = seg["start"]
                out.pop(i)
            changed = True
            break
    return out


def analyze_audio(path, job_id):
    set_job(job_id, progress=8, stage="Loading audio...")
    y, sr = librosa.load(path, sr=22050, mono=True, dtype=np.float32)
    if y.size < sr // 2:
        raise ValueError("Audio is too short to analyze")
    duration = float(len(y) / sr)

    set_job(job_id, progress=22, stage="Extracting harmonic spectrum...")
    n_fft = 2048
    hop = 1024
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    mag = np.abs(stft).astype(np.float32, copy=False)
    del stft
    power = mag * mag
    chroma = librosa.feature.chroma_stft(S=power, sr=sr, n_chroma=12, tuning=0.0, norm=2)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bass_mask = (freqs >= 55.0) & (freqs <= 330.0)
    bass = np.zeros((12, mag.shape[1]), dtype=np.float32)
    bass_freqs = freqs[bass_mask]
    bass_mag = mag[bass_mask]
    midi = np.rint(librosa.hz_to_midi(np.maximum(bass_freqs, 1.0))).astype(int)
    pcs = np.mod(midi, 12)
    for pc in range(12):
        rows = pcs == pc
        if np.any(rows):
            bass[pc] = np.sum(bass_mag[rows], axis=0)
    bass /= np.maximum(np.linalg.norm(bass, axis=0, keepdims=True), 1e-8)
    del mag, power, bass_mag

    set_job(job_id, progress=48, stage="Tracking stable chord roots...")
    group = 4
    n_frames = chroma.shape[1]
    n_bins = int(np.ceil(n_frames / group))
    emissions = np.zeros((n_bins, len(STATES)), dtype=np.float32)
    times = np.zeros(n_bins, dtype=np.float32)

    for b in range(n_bins):
        a = b * group
        z = min(n_frames, a + group)
        c = np.mean(chroma[:, a:z], axis=1)
        bs = np.mean(bass[:, a:z], axis=1)
        c /= np.linalg.norm(c) + 1e-8
        bs /= np.linalg.norm(bs) + 1e-8
        scores = TEMPLATES @ c
        for s, (root, _) in enumerate(STATES):
            scores[s] += 0.42 * bs[root]
            scores[s] += 0.10 * c[root]
        emissions[b] = scores
        frame_center = (a + z - 1) / 2.0
        times[b] = frame_center * hop / sr

    seq = decode_sequence(emissions)
    set_job(job_id, progress=70, stage="Building chord timeline...")

    raw_segments = []
    start_bin = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start_bin]:
            state = int(seq[start_bin])
            root, quality = STATES[state]
            start = 0.0 if start_bin == 0 else float((times[start_bin - 1] + times[start_bin]) / 2.0)
            end = duration if i == len(seq) else float((times[i - 1] + times[i]) / 2.0)
            confidence = float(np.mean(np.max(emissions[start_bin:i], axis=1)))
            raw_segments.append({
                "root": root,
                "quality": quality,
                "start": max(0.0, start),
                "end": min(duration, end),
                "confidence": confidence,
            })
            start_bin = i

    segments = merge_short_segments(raw_segments)
    key = infer_key(segments)

    final_segments = []
    progression = []
    for seg in segments:
        name = NOTE_NAMES[seg["root"]] + ("m" if seg["quality"] == "min" else "")
        if not progression or progression[-1] != name:
            progression.append(name)
        final_segments.append({
            "chord": name,
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "duration": round(seg["end"] - seg["start"], 3),
            "confidence": round(seg["confidence"], 3),
        })

    set_job(job_id, progress=96, stage="Finishing...")
    return {
        "job_id": job_id,
        "status": "completed",
        "duration_seconds": round(duration, 2),
        "key": key,
        "progression": progression,
        "chord_count": len(final_segments),
        "segments": final_segments,
        "raw_chords": None,
    }


def worker(job_id, path):
    try:
        result = analyze_audio(path, job_id)
        set_job(job_id, status="completed", progress=100, stage="Done", result=result)
    except Exception as exc:
        set_job(job_id, status="failed", progress=100, stage="Failed", error=str(exc))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.get("/health")
def health():
    return {"ok": True, "service": "earcolor-analyzer", "engine": "lightweight-harmonic-v2"}


@app.get("/api/v1/health/ping")
def ping():
    return {"ping": "pong"}


@app.get("/api/v1/health")
def api_health():
    return {"status": "ok", "version": "2.0.0", "model_loaded": True, "engine": "lightweight-harmonic-v2"}


@app.post("/api/v1/jobs")
async def create_job(
    audio: UploadFile = File(...),
    smooth_method: str = Query("hmm"),
    include_raw_chords: bool = Query(False),
):
    if not audio.filename:
        raise HTTPException(400, "Missing audio file")
    suffix = Path(audio.filename).suffix or ".audio"
    fd, path = tempfile.mkstemp(prefix="earcolor_", suffix=suffix)
    total = 0
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > 60 * 1024 * 1024:
                    raise HTTPException(413, "Audio file is too large")
                f.write(chunk)
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    if total == 0:
        os.remove(path)
        raise HTTPException(400, "Empty audio file")

    job_id = uuid.uuid4().hex
    with LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "progress": 1,
            "stage": "Queued",
            "error": None,
            "result": None,
        }
    threading.Thread(target=worker, args=(job_id, path), daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/v1/jobs/{job_id}/status")
def job_status(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "stage": job["stage"],
            "error": job.get("error"),
        }


@app.get("/api/v1/jobs/{job_id}/result")
def job_result(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] == "failed":
            raise HTTPException(500, job.get("error") or "Analysis failed")
        if job["status"] != "completed":
            raise HTTPException(409, "Analysis is still running")
        return job["result"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
