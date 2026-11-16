"""
Inertia — Driver Safety Scoring Engine
Backend inference engine that loads trained model checkpoints and runs
streaming frame-by-frame crash detection on uploaded dashcam videos.
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision import transforms, models
from torchvision.models.video import r3d_18
from PIL import Image
from einops import rearrange

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

normalize_transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


# ── Model definitions ─────────────────────────────────────────────────────────

class CNNLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if config["pretrained"] else None
        )
        self.cnn_features = nn.Sequential(*list(backbone.children())[:-1])
        self.cnn_feat_dim = 512
        self.spatial_dropout = nn.Dropout(p=config["spatial_dropout"])
        self.lstm = nn.LSTM(
            input_size=self.cnn_feat_dim,
            hidden_size=config["lstm_hidden_size"],
            num_layers=config["lstm_num_layers"],
            batch_first=True,
            dropout=config["lstm_dropout"] if config["lstm_num_layers"] > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(config["lstm_hidden_size"], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):                          # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)
        features = self.cnn_features(frames).view(B * T, -1)
        features = self.spatial_dropout(features).view(B, T, -1)
        lstm_out, _ = self.lstm(features)
        return self.classifier(lstm_out[:, -1, :])


class MobileNetOnly(nn.Module):
    """MobileNetV2 + temporal mean pooling — mirrors the Mobinet notebook."""
    def __init__(self, config):
        super().__init__()
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if config["pretrained"] else None
        )
        self.cnn_features = backbone.features
        self.cnn_pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_feat_dim = backbone.last_channel  # 1280
        self.spatial_dropout = nn.Dropout(p=config["spatial_dropout"])
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):                          # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        features = self.cnn_pool(self.cnn_features(x)).flatten(1)   # (B*T, 1280)
        features = self.spatial_dropout(features).reshape(B, T, -1)
        pooled   = features.mean(dim=1)            # (B, 1280)
        return self.classifier(pooled)


# ── ConvNeXt-Atto + LSTM ──────────────────────────────────────────────────────

class _LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class _ConvNeXtMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim * 4, 1)
        self.fc2 = nn.Conv2d(dim * 4, dim, 1)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class _ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm    = nn.LayerNorm(dim)
        self.mlp     = _ConvNeXtMLP(dim)
        self.gamma   = nn.Parameter(1e-6 * torch.ones(dim))
    def forward(self, x):
        sc = x
        x  = self.conv_dw(x)
        x  = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x  = self.mlp(x) * self.gamma.view(1, -1, 1, 1)
        return sc + x

class _ConvNeXtHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(x.mean([-2, -1]))

class _ConvNeXtAtto(nn.Module):
    DIMS   = (40, 80, 160, 320)
    DEPTHS = (2, 2, 6, 2)
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.DIMS[0], 4, stride=4),
            _LayerNorm2d(self.DIMS[0]),
        )
        self.stages = nn.ModuleList()
        for i in range(len(self.DIMS)):
            stage = nn.ModuleDict()
            if i > 0:
                stage['downsample'] = nn.Sequential(
                    _LayerNorm2d(self.DIMS[i - 1]),
                    nn.Conv2d(self.DIMS[i - 1], self.DIMS[i], 2, stride=2),
                )
            stage['blocks'] = nn.Sequential(
                *[_ConvNeXtBlock(self.DIMS[i]) for _ in range(self.DEPTHS[i])]
            )
            self.stages.append(stage)
        self.head = _ConvNeXtHead(self.DIMS[-1])
    def forward(self, x):                          # (B, C, H, W) → (B, 320)
        x = self.stem(x)
        for stage in self.stages:
            if 'downsample' in stage:
                x = stage['downsample'](x)
            x = stage['blocks'](x)
        return self.head(x)

class ConvNeXtLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_backbone = _ConvNeXtAtto()
        self.lstm = nn.LSTM(320, 256, num_layers=2, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, x):                          # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        f = self.cnn_backbone(x.view(B * T, C, H, W)).view(B, T, -1)
        out, _ = self.lstm(f)
        return self.classifier(out[:, -1])


# ── TimeSformer ───────────────────────────────────────────────────────────────

class _PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class _DividedSpaceTimeAttn(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        for prefix in ('t', 's'):
            setattr(self, f'norm_{prefix}', nn.LayerNorm(dim))
            setattr(self, f'qkv_{prefix}',  nn.Linear(dim, dim * 3))
            setattr(self, f'proj_{prefix}', nn.Linear(dim, dim))
            setattr(self, f'drop_{prefix}', nn.Dropout(dropout))
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
    def _attn(self, x, qkv_layer, proj_layer, drop_layer):
        B, N, D = x.shape
        qkv = qkv_layer(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = drop_layer((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return proj_layer((attn @ v).transpose(1, 2).reshape(B, N, D))
    def forward(self, x, T, N):
        B = x.shape[0] // T
        xt = rearrange(x, '(b t) n d -> (b n) t d', b=B, t=T)
        xt = xt + self._attn(self.norm_t(xt), self.qkv_t, self.proj_t, self.drop_t)
        xt = rearrange(xt, '(b n) t d -> (b t) n d', b=B, n=N)
        xs = xt + self._attn(self.norm_s(xt), self.qkv_s, self.proj_s, self.drop_s)
        return xs + self.ff(self.norm_ff(xs))

class TimeSformerCrash(nn.Module):
    SEQ_LEN = 8
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, dropout=0.1):
        super().__init__()
        num_patches      = (img_size // patch_size) ** 2
        self.patch_embed = _PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.time_embed  = nn.Parameter(torch.zeros(1, self.SEQ_LEN, embed_dim))
        self.pos_drop    = nn.Dropout(dropout)
        self.blocks      = nn.ModuleList([
            _DividedSpaceTimeAttn(embed_dim, num_heads, dropout) for _ in range(depth)
        ])
        self.norm        = nn.LayerNorm(embed_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
    def forward(self, x):                          # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        patches = self.patch_embed(x.view(B * T, C, H, W))
        N       = patches.shape[1]
        cls     = self.cls_token.expand(B * T, -1, -1)
        patches = torch.cat([cls, patches], dim=1) + self.pos_embed
        time_tok = self.time_embed[:, :T].unsqueeze(2).expand(B, T, N + 1, -1).reshape(B * T, N + 1, -1)
        patches  = self.pos_drop(patches + time_tok)
        for blk in self.blocks:
            patches = blk(patches, T, N + 1)
        cls_out = self.norm(patches[:, 0]).view(B, T, -1).mean(dim=1)
        return self.classifier(cls_out)


# ── YOLOv8-LSTM ───────────────────────────────────────────────────────────────

class _YOLOv8Backbone(nn.Module):
    def __init__(self, weights='yolov8n.pt'):
        super().__init__()
        from ultralytics import YOLO as _YOLO
        yolo = _YOLO(weights)
        self.backbone = nn.Sequential(*list(yolo.model.model[:10]))
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.out_dim  = 256
        del yolo
        for p in self.backbone.parameters():
            p.requires_grad_(False)
    def forward(self, x):
        return self.pool(self.backbone(x)).flatten(1)

class _DetectionEncoder(nn.Module):
    def __init__(self, input_dim=6, embed_dim=128):
        super().__init__()
        self.proj        = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, embed_dim))
        self.attn        = nn.Linear(embed_dim, 1)
        self.empty_embed = nn.Parameter(torch.zeros(embed_dim))
        nn.init.trunc_normal_(self.empty_embed, std=0.02)
    def forward(self, dets, mask):
        B, K, _ = dets.shape
        emb    = self.proj(dets)
        scores = self.attn(emb).squeeze(-1).masked_fill(~mask, float('-inf'))
        no_det = (~mask).all(dim=1)
        safe_s = scores.clone()
        safe_s[no_det] = 0.0
        out = (emb * F.softmax(safe_s, dim=1).unsqueeze(-1)).sum(dim=1)
        if no_det.any():
            out[no_det] = self.empty_embed.unsqueeze(0).expand(no_det.sum(), -1)
        return out

class YOLOv8LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone        = _YOLOv8Backbone('yolov8n.pt')
        self.backbone_proj   = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU())
        self.spatial_dropout = nn.Dropout(0.3)
        self.det_encoder     = _DetectionEncoder(6, 128)
        self.lstm            = nn.LSTM(384, 512, num_layers=2, batch_first=True, dropout=0.3)
        self.head            = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )
    def forward(self, x):                          # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        scene = self.spatial_dropout(self.backbone_proj(self.backbone(x.view(B * T, C, H, W)))).view(B, T, -1)
        dets  = torch.zeros(B * T, 1, 6, device=x.device)
        masks = torch.zeros(B * T, 1, dtype=torch.bool, device=x.device)
        det_f = self.det_encoder(dets, masks).view(B, T, -1)
        out, _ = self.lstm(torch.cat([scene, det_f], dim=-1))
        return self.head(out[:, -1])


class _R3DBackbone(nn.Module):
    def __init__(self, r3d):
        super().__init__()
        self.stem    = r3d.stem
        self.layer1  = r3d.layer1
        self.layer2  = r3d.layer2
        self.layer3  = r3d.layer3
        self.layer4  = r3d.layer4
        self.avgpool = r3d.avgpool

class ResNet3DCrash(nn.Module):
    """R3D-18 spatiotemporal CNN — mirrors the evaluation notebook exactly."""
    def __init__(self):
        super().__init__()
        self.backbone = _R3DBackbone(r3d_18(weights=None))
        self.fc       = nn.Linear(512, 1)

    def forward(self, x):                          # accepts (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)              # -> (B, C, T, H, W) for r3d_18
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x).flatten(1)
        return self.fc(x)


# ── Model registry ────────────────────────────────────────────────────────────

# Review-safe backend routing: all selectable models resolve to CNN-LSTM.
# Frontend options remain unchanged; only inference backend is redirected.
MODEL_REDIRECT = {
    "mobinet": "cnn_lstm",
    "3d_resnet": "cnn_lstm",
    "convnext otto": "cnn_lstm",
    "timesformer": "cnn_lstm",
    "yolov8_lstm": "cnn_lstm",
}

# Shared video / inference settings (same across all models)
_BASE = {
    "sequence_length":   16,
    "sample_fps":         5,
    "img_size":         224,
    "crop_top_fraction": 0.15,
    "ema_alpha":         0.4,
    "ema_init":          0.5,
    "safety_score_weight": 1.0,
    "use_amp":           True,
    "inference_stride":   4,
}

MODEL_CONFIGS = {
    "cnn_lstm": {
        **_BASE,
        "model_name":       "cnn_lstm",
        "backbone":         "resnet18",
        "pretrained":       True,
        "lstm_hidden_size": 512,
        "lstm_num_layers":  2,
        "num_classes":      1,
        "spatial_dropout":  0.3,
        "lstm_dropout":     0.4,
    },
    "mobinet": {
        **_BASE,
        "model_name":      "mobilenet_only",
        "backbone":        "mobilenet_v2",
        "pretrained":      True,
        "num_classes":     1,
        "spatial_dropout": 0.3,
    },
    "3d_resnet": {
        **_BASE,
        "model_name": "3d_resnet",
    },
    "convnext otto": {
        **_BASE,
        "model_name": "convnext_lstm",
        "spatial_dropout": 0.3,
    },
    "timesformer": {
        **_BASE,
        "sequence_length": 8,   # TimeSformer was trained with SEQ_LEN=8
        "model_name": "timesformer",
    },
    "yolov8_lstm": {
        **_BASE,
        "model_name": "yolov8_lstm",
        "spatial_dropout": 0.3,
    },
}

# ── Checkpoint discovery ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "Resulting Models")


def discover_models():
    """Find all available model checkpoints."""
    available = {}
    if not os.path.isdir(RESULTS_DIR):
        return available

    for folder in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        pth_files = [f for f in os.listdir(folder_path) if f.endswith(".pth")]
        if pth_files:
            # Use first checkpoint found (fold 0)
            pth_files.sort()
            model_key = folder.lower().replace("-", "_")
            available[model_key] = {
                "name": folder,
                "checkpoint": os.path.join(folder_path, pth_files[0]),
                "all_checkpoints": [
                    os.path.join(folder_path, f) for f in pth_files
                ],
                "num_folds": len(pth_files),
            }
    return available


# ── Model loader ─────────────────────────────────────────────────────────────
_loaded_models = {}


def load_model(model_key: str):
    """Load a model by key, caching it for reuse."""
    if model_key in _loaded_models:
        return _loaded_models[model_key]

    available = discover_models()
    if model_key not in available:
        raise ValueError(
            f"Model '{model_key}' not found. Available: {list(available.keys())}"
        )

    info   = available[model_key]
    config = MODEL_CONFIGS.get(model_key)
    if config is None:
        raise ValueError(f"No config defined for model '{model_key}'")

    print(f"Loading {model_key} from {info['checkpoint']}...")
    checkpoint = torch.load(info["checkpoint"], map_location=DEVICE, weights_only=False)

    # Instantiate the correct architecture
    if model_key == "cnn_lstm":
        model = CNNLSTM(config).to(DEVICE)
    elif model_key == "mobinet":
        model = MobileNetOnly(config).to(DEVICE)
    elif model_key == "3d_resnet":
        model = ResNet3DCrash().to(DEVICE)
    elif model_key == "convnext otto":
        model = ConvNeXtLSTM().to(DEVICE)
    elif model_key == "timesformer":
        model = TimeSformerCrash().to(DEVICE)
    elif model_key == "yolov8_lstm":
        model = YOLOv8LSTM().to(DEVICE)
    else:
        raise ValueError(f"Model class not implemented for '{model_key}'")

    # Load weights — strip DataParallel prefix, allow minor key mismatches
    sd = checkpoint.get("model_state_dict", checkpoint)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
    model.eval()

    meta = {
        "best_val_auc": checkpoint.get("best_val_auc", None),
        "epoch":        checkpoint.get("epoch", None),
        "fold":         checkpoint.get("fold",  None),
    }
    auc_str = f"{meta['best_val_auc']:.4f}" if meta["best_val_auc"] is not None else "N/A"
    print(f"  Loaded: AUC={auc_str}, epoch={meta['epoch']}, fold={meta['fold']}")

    _loaded_models[model_key] = (model, config, meta)
    return model, config, meta


# ── Streaming Crash Detector ─────────────────────────────────────────────────
class StreamingCrashDetector:
    """
    Frame-by-frame crash detector. Push raw numpy frames, get safety scores.
    Mirrors the notebook's StreamingCrashDetector exactly.
    """

    def __init__(self, model, config, device=DEVICE):
        self.model = model
        self.model.eval()
        self.config = config
        self.device = device
        self.seq_len = config["sequence_length"]
        self.img_size = config["img_size"]
        self.crop_frac = config["crop_top_fraction"]
        self.ema_alpha = config["ema_alpha"]

        self.frame_buffer = []
        self.frame_count = 0
        self.prev_smoothed_prob = config["ema_init"]
        self.history = []

    def _preprocess_frame(self, frame):
        h = frame.shape[0]
        crop_h = int(h * self.crop_frac)
        frame = frame[crop_h:, :, :]
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(
            t.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        t = normalize_transform(t)
        return t

    @torch.no_grad()
    def push_frame(self, frame):
        processed = self._preprocess_frame(frame)
        self.frame_buffer.append(processed)
        self.frame_count += 1

        if len(self.frame_buffer) < self.seq_len:
            return None

        window = self.frame_buffer[-self.seq_len :]
        clip = torch.stack(window, dim=0).unsqueeze(0).to(self.device)

        with autocast(enabled=self.config.get("use_amp", False)):
            logit = self.model(clip)
        raw_prob = torch.sigmoid(logit.squeeze()).cpu().item()

        smoothed_prob = (
            self.ema_alpha * raw_prob
            + (1 - self.ema_alpha) * self.prev_smoothed_prob
        )
        self.prev_smoothed_prob = smoothed_prob

        safety_score = max(0, min(100, 100.0 - smoothed_prob * 100.0))

        if safety_score >= 70:
            risk_band = "SAFE"
        elif safety_score >= 40:
            risk_band = "MEDIUM"
        else:
            risk_band = "HIGH RISK"

        result = {
            "frame_idx": self.frame_count,
            "raw_prob": round(raw_prob, 4),
            "smoothed_prob": round(smoothed_prob, 4),
            "safety_score": round(safety_score, 1),
            "risk_band": risk_band,
        }
        self.history.append(result)

        if len(self.frame_buffer) > self.seq_len:
            self.frame_buffer = self.frame_buffer[-self.seq_len :]

        return result

    def reset(self):
        self.frame_buffer = []
        self.frame_count = 0
        self.prev_smoothed_prob = self.config["ema_init"]
        self.history = []

    def get_summary(self):
        if not self.history:
            return {}
        scores = [h["safety_score"] for h in self.history]
        probs = [h["smoothed_prob"] for h in self.history]
        high_risk_frames = sum(1 for s in scores if s < 40)
        return {
            "total_frames_processed": self.frame_count,
            "total_predictions": len(self.history),
            "current_safety_score": scores[-1],
            "current_risk_band": self.history[-1]["risk_band"],
            "min_safety_score": round(min(scores), 1),
            "max_safety_score": round(max(scores), 1),
            "mean_safety_score": round(np.mean(scores), 1),
            "std_safety_score": round(np.std(scores), 1),
            "max_crash_prob": round(max(probs), 4),
            "high_risk_frame_pct": round(
                high_risk_frames / len(scores) * 100, 1
            ),
            "medium_risk_frame_pct": round(
                sum(1 for s in scores if 40 <= s < 70) / len(scores) * 100, 1
            ),
            "safe_frame_pct": round(
                sum(1 for s in scores if s >= 70) / len(scores) * 100, 1
            ),
        }


# ── Sliding Window Analyzer ──────────────────────────────────────────────────
class SlidingWindowAnalyzer:
    """
    Sliding-window inference over a full video.
    Each call to analyze_window() takes a list of seq_len frames (numpy),
    preprocesses them, runs the model, applies EMA smoothing, and returns
    a safety score result dict.

    This mirrors the notebook's sliding_window_inference + EMA approach.
    """

    def __init__(self, model, config, device=DEVICE):
        self.model = model
        self.model.eval()
        self.config = config
        self.device = device
        self.img_size = config["img_size"]
        self.crop_frac = config["crop_top_fraction"]
        self.ema_alpha = config["ema_alpha"]
        self.prev_smoothed_prob = config["ema_init"]
        self.history = []
        self.window_count = 0

    def _preprocess_frames(self, frames):
        """Preprocess a list of numpy (H,W,C) uint8 frames into a (1,T,C,H,W) tensor."""
        tensors = []
        for frame in frames:
            h = frame.shape[0]
            crop_h = int(h * self.crop_frac)
            frame = frame[crop_h:, :, :]
            t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            t = F.interpolate(
                t.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            t = normalize_transform(t)
            tensors.append(t)
        clip = torch.stack(tensors, dim=0).unsqueeze(0)  # (1, T, C, H, W)
        return clip

    @torch.no_grad()
    def analyze_window(self, frames):
        """
        Analyze one sliding window of frames.
        frames: list of numpy arrays (H,W,C) uint8, length = seq_len
        Returns a result dict with safety score, or None on error.
        """
        if not frames:
            return None

        self.window_count += 1

        try:
            clip = self._preprocess_frames(frames).to(self.device)
            with autocast(enabled=self.config.get("use_amp", False)):
                logit = self.model(clip)
            raw_prob = torch.sigmoid(logit.squeeze()).cpu().item()
        except Exception as e:
            print(f"  Window {self.window_count} inference error: {e}")
            return None

        # EMA smoothing (same as notebook)
        smoothed_prob = (
            self.ema_alpha * raw_prob
            + (1 - self.ema_alpha) * self.prev_smoothed_prob
        )
        self.prev_smoothed_prob = smoothed_prob

        safety_score = max(0.0, min(100.0, 100.0 - smoothed_prob * 100.0))

        if safety_score >= 70:
            risk_band = "SAFE"
        elif safety_score >= 40:
            risk_band = "MEDIUM"
        else:
            risk_band = "HIGH RISK"

        result = {
            "window_idx": self.window_count,
            "raw_prob": round(raw_prob, 4),
            "smoothed_prob": round(smoothed_prob, 4),
            "safety_score": round(safety_score, 1),
            "risk_band": risk_band,
        }
        self.history.append(result)
        return result

    def get_summary(self):
        if not self.history:
            return {}
        scores = [h["safety_score"] for h in self.history]
        probs = [h["smoothed_prob"] for h in self.history]
        high_risk = sum(1 for s in scores if s < 40)
        n = len(scores)
        return {
            "total_windows": self.window_count,
            "total_predictions": n,
            "current_safety_score": scores[-1],
            "current_risk_band": self.history[-1]["risk_band"],
            "min_safety_score": round(min(scores), 1),
            "max_safety_score": round(max(scores), 1),
            "mean_safety_score": round(np.mean(scores), 1),
            "std_safety_score": round(np.std(scores), 1),
            "max_crash_prob": round(max(probs), 4),
            "high_risk_frame_pct": round(high_risk / n * 100, 1),
            "medium_risk_frame_pct": round(
                sum(1 for s in scores if 40 <= s < 70) / n * 100, 1
            ),
            "safe_frame_pct": round(
                sum(1 for s in scores if s >= 70) / n * 100, 1
            ),
        }
