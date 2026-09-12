"""Memory-conscious chord-engine pipeline for EarColor on Railway.

Keeps trained BTC inference and key/segment post-processing while fitting
inside the 1 GB Railway instance. This version uses the original compact BTC
model (25 chord classes: major/minor + no-chord) instead of the heavier
ChordMini 170-class model, because the latter is being killed by the host
memory limit during real-song analysis.
"""

import asyncio
import os
import time
from datetime import datetime, timezone

from engine.loader import load_audio
from engine.model import load_detector
from engine.postprocess import (
    smooth_chords,
    merge_segments,
    infer_key,
    to_roman_numerals,
    extract_progression,
)
from engine.output import build_output
from api.services.job_store import job_store
from config import MODEL

_detector = None


def get_detector(device: str):
    global _detector
    if _detector is None:
        # Original BTC trained model: substantially lighter in RAM than
        # ChordMini/170-class while preserving the core requirement for
        # EarColor: stable trained-model root + major/minor chord recognition.
        _detector = load_detector(device, large_voca=False, use_chordmini=False)
    return _detector


def _run_analysis_stages(
    job_id: str,
    audio_path: str,
    audio_filename: str,
    device: str,
    smooth_method: str,
    include_raw: bool,
    progress_offset: int = 0,
) -> dict:
    job_store.update_progress(job_id, 8 + progress_offset, "Loading audio...")
    audio_dict = load_audio(audio_path)
    y = audio_dict["y"]
    sr = audio_dict["sr"]

    # Avoid a duplicate full-song chroma/HPSS/beat pass. BTC extracts the
    # model CQT itself; doing both substantially increases peak RAM.
    beat_times = []

    job_store.update_progress(job_id, 25 + progress_offset, "Loading BTC model...")
    detector = get_detector(device)

    job_store.update_progress(job_id, 45 + progress_offset, "Running trained chord model...")
    raw_chords = detector.predict(y, sr)

    job_store.update_progress(job_id, 72 + progress_offset, "Stabilizing chord sequence...")
    smoothed = smooth_chords(raw_chords, method=smooth_method)

    job_store.update_progress(job_id, 82 + progress_offset, "Building chord segments...")
    hop_length_btc = MODEL["hop_length"]
    frame_times = [i * hop_length_btc / sr for i in range(len(smoothed))]
    import numpy as np
    segments = merge_segments(smoothed, np.array(frame_times))

    job_store.update_progress(job_id, 90 + progress_offset, "Inferring key...")
    key = infer_key(segments)
    segments = to_roman_numerals(segments, key)
    progression = extract_progression(segments)

    job_store.update_progress(job_id, 96 + progress_offset, "Formatting output...")
    output = build_output(
        segments=segments,
        key=key,
        progression=progression,
        audio_dict=audio_dict,
        raw_chords=list(raw_chords),
        beats=beat_times,
    )

    chord_segments = []
    for seg in output["segments"]:
        chord_segments.append({
            "chord": seg["chord"],
            "roman": seg.get("roman", "?"),
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "duration": round(seg["duration"], 3),
            "confidence": round(seg.get("confidence", 0.0), 3),
        })

    return {
        "job_id": job_id,
        "status": "completed",
        "audio_filename": audio_filename,
        "duration_seconds": round(output["duration_seconds"], 2),
        "tempo_bpm": round(output.get("tempo_bpm", 0.0), 1),
        "key": output["key"],
        "progression": output["progression"],
        "chord_count": output["chord_count"],
        "segments": chord_segments,
        "raw_chords": list(output["raw_chords"]) if include_raw else None,
        "processing_time_seconds": 0.0,
        "created_at": "",
    }


def _run_stages(job_id: str) -> dict:
    record = job_store.get(job_id)
    if record is None:
        raise ValueError(f"Job not found: {job_id}")
    return _run_analysis_stages(
        job_id=job_id,
        audio_path=record.audio_path,
        audio_filename=record.audio_filename,
        device=record.params.get("device", "cpu"),
        smooth_method=record.params.get("smooth_method", "hmm"),
        include_raw=record.params.get("include_raw_chords", False),
        progress_offset=0,
    )


async def run_pipeline(job_id: str) -> None:
    await _run_async_wrapper(job_id, _run_stages)


async def _run_async_wrapper(job_id: str, stages_fn) -> None:
    loop = asyncio.get_event_loop()
    start_time = time.monotonic()
    audio_path = None
    record = job_store.get(job_id)
    if record is not None:
        audio_path = record.audio_path
    try:
        result = await loop.run_in_executor(None, stages_fn, job_id)
        result["processing_time_seconds"] = round(time.monotonic() - start_time, 2)
        now = datetime.now(timezone.utc)
        result["created_at"] = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        job_store.set_completed(job_id, result)
    except Exception as exc:
        job_store.set_failed(job_id, str(exc))
    finally:
        if audio_path is not None:
            try:
                os.remove(audio_path)
            except (FileNotFoundError, PermissionError, OSError):
                pass
