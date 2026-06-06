# =========================================================
# Air Pollution Detection Flask App
# (Predicts Clean/Moderate/Polluted + PM2.5 + PM10 + AQI)
# =========================================================
import os
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import cv2
import numpy as np
import shutil

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_model.pth"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "replace_with_a_random_secret_key")

# Store recent history in memory
history = []

# =========================================================
# MODEL LOADING
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 3)  # clean, moderate, polluted
model.to(device)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Train and save it first.")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

CLASS_NAMES = ['clean', 'moderate', 'polluted']

# =========================================================
# IMAGE TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT

def predict_image(image_path):
    """Predict pollution class and probability distribution."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(out.argmax(1).item())
    return probs, pred_idx

# --- Grad-CAM Generator ---
def generate_gradcam(image_path, target_layer="layer4"):
    img_pil = Image.open(image_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)
    acts, grads = {}, {}

    def fwd_hook(m, i, o): acts['val'] = o.detach()
    def bwd_hook(m, gi, go): grads['val'] = go[0].detach()

    # Register hooks
    for name, module in model.named_modules():
        if name == target_layer:
            module.register_forward_hook(fwd_hook)
            module.register_backward_hook(bwd_hook)
            break

    out = model(x)
    pred = int(out.argmax(1).item())
    model.zero_grad()
    out[0, pred].backward()

    if 'val' not in acts or 'val' not in grads:
        raise RuntimeError("Grad-CAM hooks failed to capture gradients/activations.")

    act = acts['val']
    grad = grads['val']
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    # Overlay heatmap
    img_cv = np.array(img_pil)[:, :, ::-1]
    heatmap = cv2.resize(cam, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    hm_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, hm_color, 0.4, 0)
    cv2.putText(overlay, f"Pred: {CLASS_NAMES[pred]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    out_file = OUTPUT_DIR / f"gradcam_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(out_file), overlay)
    return out_file.name

# --- PM2.5, PM10, AQI Estimator ---
def probs_to_pm_values(probs):
    """Estimate PM2.5, PM10, and AQI from probabilities."""
    p_clean, p_mod, p_poll = float(probs[0]), float(probs[1]), float(probs[2])

    # Representative PM2.5 levels for each class
    centroids_pm25 = np.array([15.0, 100.0, 225.0])
    est_pm25 = float(np.dot(probs, centroids_pm25))

    # Estimate PM10 ≈ 1.5 × PM2.5
    est_pm10 = est_pm25 * 1.5

    # Convert PM2.5 → AQI (approximation)
    pm = est_pm25
    if pm <= 12:
        aqi = (pm / 12) * 50
    elif pm <= 35.4:
        aqi = 50 + ((pm - 12) / (35.4 - 12)) * 50
    elif pm <= 55.4:
        aqi = 100 + ((pm - 35.4) / (55.4 - 35.4)) * 50
    elif pm <= 150.4:
        aqi = 150 + ((pm - 55.4) / (150.4 - 55.4)) * 50
    elif pm <= 250.4:
        aqi = 200 + ((pm - 150.4) / (250.4 - 150.4)) * 100
    else:
        aqi = 300 + (pm - 250.4)

    return round(est_pm25, 1), round(est_pm10, 1), int(round(aqi))

# =========================================================
# FLASK ROUTES
# =========================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files['image']
        if file.filename == "":
            flash("No file selected")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            fn = secure_filename(file.filename)
            save_path = UPLOAD_DIR / fn
            file.save(save_path)

            # Predict and estimate AQI/PM values
            probs, pred_idx = predict_image(save_path)
            pred_name = CLASS_NAMES[pred_idx]
            pm25, pm10, aqi = probs_to_pm_values(probs)

            # Grad-CAM visualization
            try:
                gradfile = generate_gradcam(str(save_path))
            except Exception as e:
                gradfile = None
                print("Grad-CAM error:", e)

            # Store in history
            entry = {
                "image": fn,
                "prediction": pred_name,
                "probs": [round(float(x), 3) for x in probs.tolist()],
                "pm25": pm25,
                "pm10": pm10,
                "aqi": aqi,
                "gradcam": gradfile
            }
            history.insert(0, entry)
            if len(history) > 20:
                history.pop()

            return render_template("result.html", entry=entry, history=history)

        else:
            flash("Invalid file type.")
            return redirect(request.url)

    return render_template("index.html", history=history)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    # Optional: clear old uploads on startup
    # shutil.rmtree(UPLOAD_DIR); os.makedirs(UPLOAD_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
