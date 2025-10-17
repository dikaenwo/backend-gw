from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Kelas untuk mengatasi masalah kompatibilitas
class PatchedDepthwiseConv2D(DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        if 'groups' in config:
            del config['groups']
        return super().from_config(config)

# Load model saat startup
inference_model = None
model_path = "model_bglow.h5"
IMG_SIZE = (224, 224)
class_names = ['Berminyak', 'Kering', 'Kombinasi', 'Normal']

# Deskripsi untuk setiap jenis kulit
skin_descriptions = {
    'Berminyak': 'Kulit berminyak adalah kondisi kulit yang memproduksi sebum (minyak alami) secara berlebihan, sehingga tampak mengkilap, pori-pori besar, dan rentan berjerawat.',
    'Kering': 'Kulit kering adalah kondisi kulit yang mengalami kekurangan kelembapan, minyak alami (sebum), atau keduanya, sehingga terasa kasar, kaku, bersisik, gatal, dan kurang elastis.',
    'Kombinasi': 'Kulit kombinasi adalah jenis kulit yang memiliki karakteristik campuran antara kulit berminyak dan kering pada area wajah yang berbeda.',
    'Normal': 'Kulit normal adalah jenis kulit yang seimbang, tidak terlalu kering maupun berminyak, dengan tekstur halus, pori-pori kecil, dan jarang mengalami masalah kulit.'
}

# Permasalahan kulit untuk setiap jenis
skin_problems = {
    'Berminyak': ['Jerawat', 'Komedo', 'Pori-pori Besar'],
    'Kering': ['Kulit Kusam', 'Garis Halus', 'Kulit Bersisik'],
    'Kombinasi': ['Jerawat di T-Zone', 'Kulit Kering di Pipi', 'Pori-pori Tidak Merata'],
    'Normal': ['Kulit Kusam', 'Dehidrasi Ringan']
}

try:
    print("Memuat model...")
    if os.path.exists(model_path):
        inference_model = tf.keras.models.load_model(
            model_path,
            custom_objects={'DepthwiseConv2D': PatchedDepthwiseConv2D}
        )
        print(f"✅ Model '{model_path}' berhasil dimuat.")
    else:
        print(f"❌ Model '{model_path}' tidak ditemukan.")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")

def predict_image(img_bytes):
    """Fungsi untuk memprediksi jenis kulit dari bytes gambar"""
    try:
        # Konversi bytes ke PIL Image
        img = Image.open(io.BytesIO(img_bytes))
        
        # Resize dan preprocess
        img = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # Prediksi
        predictions = inference_model.predict(img_array_expanded, verbose=0)
        
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions[0]) * 100)
        
        return predicted_class, confidence
    except Exception as e:
        raise Exception(f"Error saat prediksi: {str(e)}")

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'BGlow Skin Analysis API',
        'endpoints': {
            '/predict': 'POST - Upload image for skin type prediction',
            '/health': 'GET - Check API health status'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Endpoint untuk check status API dan model"""
    model_status = "loaded" if inference_model is not None else "not loaded"
    return jsonify({
        'status': 'success',
        'model_status': model_status,
        'available_classes': class_names
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi jenis kulit"""
    try:
        # Validasi model sudah dimuat
        if inference_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model belum dimuat'
            }), 500
        
        # Validasi ada file yang diupload
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Tidak ada file gambar yang diupload'
            }), 400
        
        file = request.files['image']
        
        # Validasi file tidak kosong
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Nama file kosong'
            }), 400
        
        # Baca file sebagai bytes
        img_bytes = file.read()
        
        # Prediksi
        predicted_class, confidence = predict_image(img_bytes)
        
        # Siapkan response
        response = {
            'status': 'success',
            'data': {
                'skin_type': predicted_class,
                'confidence': round(confidence, 2),
                'description': skin_descriptions.get(predicted_class, ''),
                'problems': skin_problems.get(predicted_class, [])
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Jalankan server
    # Untuk production, gunakan gunicorn atau waitress
    app.run(host='0.0.0.0', port=5000, debug=True)