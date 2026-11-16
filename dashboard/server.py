"""
Inertia — Driver Safety Scoring Server
FastAPI backend that serves the dashboard and processes video uploads.
"""

import os
import sys
import json
import time
import uuid
import asyncio
import tempfile
import shutil
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (
    discover_models,
    load_model,
    StreamingCrashDetector,
    MODEL_REDIRECT,
    DEVICE,
)

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    yield
    # Cleanup on shutdown
    if os.path.isdir(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


app = FastAPI(title="Inertia — Driver Safety Scoring", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(DASHBOARD_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")


@app.get("/")
async def serve_dashboard():
    return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))


@app.get("/api/models")
async def list_models():
    """Return available models with metadata."""
    available = discover_models()
    result = {}
    for key, info in available.items():
        result[key] = {
            "name": info["name"],
            "num_folds": info["num_folds"],
            "has_checkpoint": True,
        }
    return result


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file in chunks to handle large files reliably."""
    session_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".mp4"
    save_path = os.path.join(UPLOAD_DIR, f"{session_id}{ext}")

    total_bytes = 0
    CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB chunks
    with open(save_path, "wb") as f:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            total_bytes += len(chunk)

    file_size_mb = total_bytes / (1024 * 1024)
    print(f"Upload complete: {file.filename} ({file_size_mb:.1f} MB) -> {save_path}")

    return {
        "session_id": session_id,
        "filename": file.filename,
        "path": save_path,
        "size_mb": round(file_size_mb, 1),
    }


@app.websocket("/ws/analyze/{session_id}")
async def websocket_analyze(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time video analysis.
    Client sends: { "model": "cnn_lstm", "video_path": "..." }
    Server streams back per-frame results.
    """
    await websocket.accept()

    try:
        # Wait for start message
        init_msg = await websocket.receive_json()
        requested_model = init_msg.get("model", "cnn_lstm")
        model_key = MODEL_REDIRECT.get(requested_model, requested_model)
        video_path = init_msg.get("video_path", "")

        if not os.path.exists(video_path):
            await websocket.send_json({"error": f"Video not found: {video_path}"})
            await websocket.close()
            return

        # Load model
        await websocket.send_json(
            {
                "status": "loading_model",
                "model": requested_model,
                "backend_model": model_key,
            }
        )
        model, config, meta = load_model(model_key)
        await websocket.send_json(
            {
                "status": "model_loaded",
                "model": requested_model,
                "backend_model": model_key,
                "meta": {
                    "best_val_auc": meta.get("best_val_auc"),
                    "epoch": meta.get("epoch"),
                    "fold": meta.get("fold"),
                },
            }
        )

        # ── Step 1: Read video metadata only (no frame decoding yet) ─────
        await websocket.send_json({"status": "decoding_video"})
        loop = asyncio.get_event_loop()

        def _get_video_meta(path, cfg):
            """Return (video_fps, total_raw_frames, sampled_raw_indices, use_decord).
            Only reads metadata — no pixel data loaded."""
            sample_fps = cfg["sample_fps"]
            use_decord = False
            try:
                from decord import VideoReader, cpu as dcpu
                vr = VideoReader(path, ctx=dcpu(0))
                video_fps = float(vr.get_avg_fps() or 30.0)
                total_raw = len(vr)
                del vr
                use_decord = True
            except Exception:
                import cv2
                cap = cv2.VideoCapture(path)
                video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            step = max(1, round(video_fps / sample_fps))
            sampled_raw_indices = list(range(0, total_raw, step))
            return video_fps, total_raw, sampled_raw_indices, use_decord

        video_fps, total_raw_frames, sampled_raw_indices, use_decord = \
            await loop.run_in_executor(None, _get_video_meta, video_path, config)

        n_sampled    = len(sampled_raw_indices)
        duration_sec = total_raw_frames / video_fps
        sample_fps   = config["sample_fps"]
        seq_len      = config["sequence_length"]
        stride       = config.get("inference_stride", 4)

        await websocket.send_json({
            "status": "video_decoded",
            "total_frames": int(total_raw_frames),
            "sampled_frames": n_sampled,
            "video_fps": round(video_fps, 1),
            "sample_fps": sample_fps,
            "duration_sec": round(duration_sec, 1),
        })

        # ── Step 2: Build sliding window list over sampled indices ─────
        await websocket.send_json({"status": "analyzing"})

        windows = []
        for start in range(0, max(1, n_sampled - seq_len + 1), stride):
            end = start + seq_len
            if end > n_sampled:
                end = n_sampled
                start = max(0, end - seq_len)
            windows.append((start, end))
        if not windows:
            windows.append((0, min(n_sampled, seq_len)))
        seen = set()
        unique_windows = []
        for w in windows:
            if w not in seen:
                seen.add(w)
                unique_windows.append(w)
        windows = unique_windows
        n_windows = len(windows)
        print(f"Sliding window: {n_sampled} sampled frames -> {n_windows} windows "
              f"(seq_len={seq_len}, stride={stride}, duration={duration_sec:.1f}s)")

        from engine import SlidingWindowAnalyzer
        analyzer = SlidingWindowAnalyzer(model, config)

        # ── Step 3: On-demand frame loading — only seq_len frames in RAM ──
        # Open VideoReader once; load only the 16 raw frame indices per window.
        # This keeps peak memory at ~seq_len frames regardless of video length.
        WIN_BATCH = 4

        if use_decord:
            # decord path: random-access batch reads, VideoReader stays open
            def _open_decord(path):
                from decord import VideoReader, cpu as dcpu
                return VideoReader(path, ctx=dcpu(0))

            vr = await loop.run_in_executor(None, _open_decord, video_path)

            for wb_start in range(0, n_windows, WIN_BATCH):
                wb_end = min(wb_start + WIN_BATCH, n_windows)
                batch_windows = windows[wb_start:wb_end]

                def _process_decord(wins, raw_indices, vr_ref, start_off, n_win, s_fps, s_len):
                    results = []
                    for wi, (ws, we) in enumerate(wins):
                        raw_idx = raw_indices[ws:we]
                        batch   = vr_ref.get_batch(raw_idx).asnumpy()
                        frames  = [batch[j] for j in range(len(batch))]
                        while len(frames) < s_len and frames:
                            frames.append(frames[-1])
                        r = analyzer.analyze_window(frames)
                        if r is not None:
                            g = start_off + wi
                            r["progress"]  = round((g + 1) / n_win * 100, 1)
                            r["time_sec"]  = round((ws + we) / 2 / s_fps, 2)
                            r["status"]    = "frame_result"
                            results.append(r)
                    return results

                batch_results = await loop.run_in_executor(
                    None, _process_decord,
                    batch_windows, sampled_raw_indices, vr,
                    wb_start, n_windows, sample_fps, seq_len
                )
                for result in batch_results:
                    await websocket.send_json(result)
                await asyncio.sleep(0)

            del vr

        else:
            # OpenCV fallback: sequential ring-buffer scan — one pass, O(seq_len) memory
            def _process_opencv_streaming(path, raw_indices, wins, n_win, s_fps, s_len):
                import cv2
                from collections import deque

                cap = cv2.VideoCapture(path)
                results   = []
                buf       = {}          # raw_frame_idx -> np array
                raw_iter  = 0           # current raw frame position
                win_idx   = 0

                for wi, (ws, we) in enumerate(wins):
                    needed_raw = raw_indices[ws:we]

                    # Read forward until we have the last needed raw frame
                    while raw_iter <= needed_raw[-1]:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if raw_iter in set(needed_raw):
                            buf[raw_iter] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        raw_iter += 1

                    frames = [buf[i] for i in needed_raw if i in buf]
                    while len(frames) < s_len and frames:
                        frames.append(frames[-1])

                    r = analyzer.analyze_window(frames)
                    if r is not None:
                        r["progress"] = round((wi + 1) / n_win * 100, 1)
                        r["time_sec"] = round((ws + we) / 2 / s_fps, 2)
                        r["status"]   = "frame_result"
                        results.append(r)

                    # Evict frames before the next window's start to bound memory
                    if wi + 1 < len(wins):
                        next_ws = wins[wi + 1][0]
                        evict_before = raw_indices[next_ws] if next_ws < len(raw_indices) else 0
                        buf = {k: v for k, v in buf.items() if k >= evict_before}

                cap.release()
                return results

            all_results = await loop.run_in_executor(
                None, _process_opencv_streaming,
                video_path, sampled_raw_indices, windows,
                n_windows, sample_fps, seq_len
            )
            for result in all_results:
                await websocket.send_json(result)
                await asyncio.sleep(0)

        # ── Final summary ─────────────────────────────────────────────
        summary = analyzer.get_summary()
        summary["status"] = "complete"
        summary["duration_sec"] = round(duration_sec, 1)
        summary["total_windows"] = n_windows
        summary["history"] = analyzer.history
        await websocket.send_json(summary)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass



if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Available models: {list(discover_models().keys())}")
    print("Starting Inertia Dashboard at http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765)
