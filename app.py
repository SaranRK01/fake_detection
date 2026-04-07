import os
import tempfile
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "df_lstm_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, static_folder=str(ROOT), static_url_path="")


class VideoLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention_net(lstm_out)
        return self.classifier(attn_out)


transform = Compose(
    [
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_model = None
_feature_extractor = None


def load_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    arch = checkpoint["architecture"]
    loaded_model = VideoLSTM(
        input_dim=arch["input_dim"],
        hidden_dim=arch["hidden_dim"],
        num_layers=arch["num_layers"],
        num_classes=arch["num_classes"],
        dropout=arch["dropout"],
    )
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model = loaded_model.to(DEVICE)
    loaded_model.eval()
    return loaded_model


def get_feature_extractor():
    try:
        weights = models.ResNeXt101_32X8D_Weights.DEFAULT
        backbone = models.resnext101_32x8d(weights=weights)
    except Exception:
        # Fallback avoids hard failure in offline environments.
        backbone = models.resnext101_32x8d(weights=None)
    extractor = nn.Sequential(*list(backbone.children())[:-1])
    extractor.eval()
    extractor.to(DEVICE)
    return extractor


def ensure_runtime_loaded():
    global _model, _feature_extractor
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
    if _feature_extractor is None:
        _feature_extractor = get_feature_extractor()


def extract_frames(video_path: str, num_frames: int = 8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // num_frames, 1)
    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(transform(frame))

    cap.release()

    if not frames:
        raise ValueError("Could not extract frames from video.")

    while len(frames) < num_frames:
        frames.append(torch.zeros_like(frames[0]))

    return torch.stack(frames)


def extract_video_features(frames: torch.Tensor):
    with torch.no_grad():
        features = []
        for frame in frames:
            frame = frame.unsqueeze(0).to(DEVICE)
            feature = _feature_extractor(frame).squeeze()
            features.append(feature.cpu())
        features_tensor = torch.stack(features).unsqueeze(0)
    return features_tensor


def predict_video(video_path: str):
    frames = extract_frames(video_path)
    features = extract_video_features(frames)

    with torch.no_grad():
        outputs = _model(features.to(DEVICE))
        probabilities = F.softmax(outputs, dim=1)
        fake_prob = probabilities[0, 1].item()
        predicted_class = 1 if fake_prob > 0.5 else 0

    return {
        "video": os.path.basename(video_path),
        "fake_probability": fake_prob,
        "prediction": "FAKE" if predicted_class == 1 else "REAL",
    }


@app.get("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.post("/api/predict")
def api_predict():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file uploaded."}), 400

        video_file = request.files["video"]
        if video_file.filename == "":
            return jsonify({"error": "Please choose a video file."}), 400

        ensure_runtime_loaded()

        suffix = Path(video_file.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            video_file.save(temp_path)

        try:
            result = predict_video(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return jsonify(result)
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as err:
        return jsonify({"error": f"Prediction failed: {err}"}), 500


if __name__ == "__main__":
    # On Python 3.13 + Windows, Werkzeug watchdog reloader can crash.
    app.run(debug=True, use_reloader=False)
