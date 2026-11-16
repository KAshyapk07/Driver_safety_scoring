# VIVA README - Driver Safety Scoring

## 1. Project in One Line
Inertia is a real-time driver safety scoring system that analyzes dashcam video and outputs a continuous Safety Score from 0 to 100, plus risk bands (SAFE, MEDIUM, HIGH RISK).

## 2. Demo Goal for Viva
Show a complete AI pipeline from video upload to live risk scoring, and explain the multi-model research work done on the Nexar dataset.

## 3. What the App Does End-to-End
1. User opens dashboard UI.
2. User selects a model from dropdown and uploads a video.
3. Backend uploads file in chunks and creates a session.
4. WebSocket starts analysis for that session.
5. Backend decodes video metadata and samples frames.
6. Sliding-window inference runs over sampled frames.
7. UI receives live frame/window results.
8. UI shows live Safety Score gauge, crash probability trend, and final summary.

## 4. Core Formula and Risk Bands
Safety Score formula:
Safety Score = 100 - (P_crash x 100)

Risk levels:
- 70 to 100: SAFE
- 40 to 69.9: MEDIUM
- 0 to 39.9: HIGH RISK

## 5. Inference Pipeline Used in App
1. Sample video at 5 FPS.
2. Apply top-crop of 15 percent to remove watermark region.
3. Resize to 224 x 224.
4. Normalize with ImageNet mean and std.
5. Form temporal windows.
6. Run model inference.
7. Apply EMA smoothing (alpha = 0.4).
8. Convert to Safety Score and risk band.

## 6. Frontend-Backend Flow (Quick Technical View)
Frontend:
- Dashboard page and visual components in dashboard/index.html.
- Sends selected model key and uploaded video path via WebSocket.

Backend:
- API server in dashboard/server.py.
- Model registry/loading and inference logic in dashboard/engine.py.

API routes:
- GET /api/models: list discovered model folders and checkpoints.
- POST /api/upload: upload video and return session data.
- WS /ws/analyze/{session_id}: stream analysis status and results.

## 7. Important Demo Note (Current Backend Behavior)
For viva stability, the backend currently routes all model selections to CNN-LSTM internally.

What this means:
- Frontend model dropdown remains unchanged.
- Selected model name is still shown in UI.
- Actual inference backend model is CNN-LSTM for every selection.

Current redirect mapping:
- mobinet -> cnn_lstm
- 3d_resnet -> cnn_lstm
- convnext otto -> cnn_lstm
- timesformer -> cnn_lstm
- yolov8_lstm -> cnn_lstm

cnn_lstm stays cnn_lstm.

## 8. Model Details for Viva (Research + Artifacts)

### 8.1 Models Trained in Project
1. CNN-LSTM
2. 3D-ResNet
3. MobileNet
4. ConvNeXt Atto variant
5. TimeSformer
6. YOLOv8-LSTM

Note: README also discusses an ensemble concept, but the dashboard backend inference classes are implemented for the 6 model families listed above.

### 8.2 Where Model Artifacts Are Stored
- Resulting Models/CNN-LSTM
  - seed42_fold0.pth
  - seed42_fold1.pth
  - seed42_fold2.pth
  - shared_data_splits.json
- Resulting Models/3D-Resnet
  - seed42_fold1_3dresnet_best.pth
  - seed42_fold3_last_3dResnet.pth
- Resulting Models/Mobinet
  - mobinet_seed42_fold0.pth
  - seed42_fold1.pth
  - seed42_fold2.pth
- Resulting Models/Convnext otto
  - ConvNext_Run.pth
  - Convenext_Fail.pth
- Resulting Models/TimeSformer
  - TineSformer_seed42_fold0.pth
  - seed42_fold1.pth
  - seed42_fold2.pth
- Resulting Models/Yolov8-LSTM
  - yolov8_seed42_fold0.pth
  - seed42_fold1.pth
  - seed42_fold2.pth

### 8.3 Architecture Notes (Backend Classes)
- CNNLSTM: ResNet18 feature extractor + LSTM temporal head + classifier.
- MobileNetOnly: MobileNetV2 feature extractor + temporal mean pooling + classifier.
- ResNet3DCrash: R3D-18 style 3D spatiotemporal network.
- ConvNeXtLSTM: ConvNeXt-Atto style frame encoder + LSTM head.
- TimeSformerCrash: divided space-time attention transformer for video.
- YOLOv8LSTM: YOLOv8 feature backbone + detection feature encoding + LSTM head.

## 9. How to Run for Demo
Run from project root:
1. cd dashboard
2. pip install -r requirements.txt
3. python server.py
4. Open http://localhost:8765 in browser.

## 10. Suggested 5-7 Minute Viva Demo Script
1. Start with problem statement: real-time crash-risk understanding from dashcam video.
2. Explain dataset and training effort: multiple architectures trained under compute constraints.
3. Show model dropdown and uploaded video.
4. Start analysis and explain each status stage: loading model, decoding video, analyzing.
5. Point to live Safety Score gauge and risk band changes.
6. Explain smoothing and why stable scores are important for driver warning systems.
7. Show final summary stats and risk distribution.
8. Mention current review-safe backend routing to CNN-LSTM for consistent runtime behavior.
9. Close with future work: true per-model backend switching, calibration, and deployment optimization.

## 11. Likely Viva Questions and Ready Answers
Q1: Why multiple models if runtime uses one now?
A: We completed multi-model training and benchmarking as research contribution. For the live review demo, we force a stable backend path to avoid runtime instability and ensure a smooth demonstration.

Q2: Why CNN-LSTM as runtime model?
A: It gives a strong accuracy-speed balance and is robust for streaming window-based inference.

Q3: How do you avoid noisy frame-to-frame outputs?
A: We apply EMA smoothing to crash probability before converting to Safety Score.

Q4: Why top-crop the frame?
A: To remove top watermark/UI artifacts and reduce spurious model attention.

Q5: How is this real-time?
A: We use sampled frames, sliding windows, and batched inference updates over WebSocket.

## 12. What to Say if Asked About Compute Constraints
- Training was done with limited Kaggle GPU resources and strict session limits.
- Cross-validation and fold checkpoints were used to keep the process rigorous.
- Practical engineering tradeoff for viva day: route all selections through one stable backend model.

## 13. Known Demo-Day Precautions
- Keep one short, reliable demo video ready in advance.
- Start server before viva starts and verify localhost page loads.
- If model loading is slow on first run, explain checkpoint warm-up and caching.
- If WebSocket fails, re-upload and restart the session quickly.

## 14. Dependencies Used in Dashboard Runtime
- fastapi
- uvicorn
- python-multipart
- torch
- torchvision
- numpy
- decord
- opencv-python

## 15. Evidence Pointers in Codebase
- Backend API and WS flow: dashboard/server.py
- Model registry, configs, redirects, and inference classes: dashboard/engine.py
- UI and client WebSocket logic: dashboard/index.html
- Training methodology and rigor details: Training.md
- High-level benchmark narrative: README.md

## 16. Final Viva Positioning Statement
This project demonstrates both deep learning research breadth (multiple trained architectures and protocol design) and product-level thinking (real-time dashboard, streaming inference, robust demo behavior). For review-day reliability, backend model routing is intentionally stabilized while preserving the full research story and artifacts.
