from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image
import torch

app = Flask(__name__)

# Load model YOLO HANYA SEKALI saat server start
# Tambahkan fuse=False untuk menghindari error AttributeError: bn
print("Loading YOLO model...")
model = YOLO("ModelV7.pt", task='detect')
model.fuse = False  # Disable fusing untuk menghindari error
print("Model loaded successfully!")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # Baca gambar dari file upload
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Inference dengan YOLO + threshold confidence 0.2
        # Tambahkan verbose=False untuk mengurangi output
        results = model.predict(img, conf=0.3, verbose=False)

        # Gambar bounding box di image
        annotated_img = results[0].plot()  # YOLO otomatis gambar box
        
        # Konversi BGR ke RGB (karena YOLO pakai BGR)
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Konversi ke bytes untuk dikirim
        pil_img = Image.fromarray(annotated_img_rgb)
        img_io = io.BytesIO()
        pil_img.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')
    
    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_json", methods=["POST"])
def predict_json():
    """Endpoint terpisah untuk dapetin JSON predictions aja"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Inference dengan YOLO
        results = model.predict(img, conf=0.2, verbose=False)

        detections = []
        for r in results[0].boxes:
            detections.append({
                "class": int(r.cls[0].item()),
                "confidence": float(r.conf[0].item()),
                "box": r.xyxy[0].tolist()
            })

        return jsonify({"predictions": detections})
    
    except Exception as e:
        print(f"Error in /predict_json: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Endpoint untuk cek apakah server hidup"""
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == "__main__":
    # Gunakan threaded=True untuk handle multiple requests
    app.run(host="0.0.0.0", port=2000, debug=True, threaded=True)