from flask import Flask, request, jsonify
import os

# Force TensorFlow pakai CPU saja (sebelum import tensorflow)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Kurangi log TensorFlow

import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

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

def load_model_safe():
    """Load model dengan berbagai metode fallback"""
    global inference_model
    
    if not os.path.exists(model_path):
        print(f"❌ File model '{model_path}' tidak ditemukan!")
        return False
    
    # Metode 1: Load dengan compile=False
    print("🔄 Mencoba metode 1: load_model dengan compile=False...")
    try:
        inference_model = tf.keras.models.load_model(
            model_path,
            custom_objects={'DepthwiseConv2D': PatchedDepthwiseConv2D},
            compile=False
        )
        print(f"✅ Model berhasil dimuat (metode 1)")
        return True
    except Exception as e:
        print(f"❌ Metode 1 gagal: {str(e)[:100]}")
    
    # Metode 2: Load dengan safe_mode=False (TF 2.16+)
    print("🔄 Mencoba metode 2: safe_mode=False...")
    try:
        inference_model = tf.keras.models.load_model(
            model_path,
            custom_objects={'DepthwiseConv2D': PatchedDepthwiseConv2D},
            compile=False,
            safe_mode=False
        )
        print(f"✅ Model berhasil dimuat (metode 2)")
        return True
    except Exception as e:
        print(f"❌ Metode 2 gagal: {str(e)[:100]}")
    
    # Metode 3: Load model architecture dan weights terpisah
    print("🔄 Mencoba metode 3: load weights saja...")
    try:
        # Coba buat model dari scratch (perlu tahu arsitektur aslinya)
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
        
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(4, activation='softmax')(x)
        inference_model = Model(inputs=base_model.input, outputs=predictions)
        
        # Load weights
        inference_model.load_weights(model_path)
        print(f"✅ Model berhasil dimuat (metode 3 - weights only)")
        return True
    except Exception as e:
        print(f"❌ Metode 3 gagal: {str(e)[:100]}")
    
    print("\n" + "="*60)
    print("❌ SEMUA METODE GAGAL!")
    print("="*60)
    print("\n📋 SOLUSI:")
    print("1. Hubungi tim yang training model")
    print("2. Minta file model dalam format SavedModel (folder, bukan .h5)")
    print("3. Atau minta file .weights.h5 + script arsitektur model")
    print("4. Atau re-save model dengan: model.save('model_bglow', save_format='tf')")
    print("\n")
    
    return False

# Load model saat startup
print("\n" + "="*60)
print("🚀 Memulai BGlow API Server")
print("="*60)
print(f"📦 TensorFlow Version: {tf.__version__}")
print(f"🖥️  Device: CPU Only (VPS Mode)")
print(f"📁 Model Path: {model_path}")
print("="*60 + "\n")

if load_model_safe():
    print(f"   Input shape: {inference_model.input_shape}")
    print(f"   Output shape: {inference_model.output_shape}")
    print(f"   Classes: {class_names}\n")
else:
    print("⚠️  Server akan tetap jalan, tapi endpoint /predict tidak akan berfungsi\n")

def predict_image(img_bytes):
    """Fungsi untuk memprediksi jenis kulit dari bytes gambar"""
    try:
        # Validasi ukuran file
        max_size = 10 * 1024 * 1024  # 10MB
        if len(img_bytes) > max_size:
            raise Exception(f"Ukuran gambar terlalu besar. Maksimal: {max_size/1024/1024:.0f}MB")
        
        # Konversi bytes ke PIL Image
        img = Image.open(io.BytesIO(img_bytes))
        
        # Validasi dan konversi ke RGB jika perlu
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print(f"   ℹ️ Gambar dikonversi dari {img.mode} ke RGB")
        
        # Resize dan preprocess
        img = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # Normalisasi (uncomment jika model ditraining dengan normalisasi)
        # img_array_expanded = img_array_expanded / 255.0
        
        # Prediksi
        predictions = inference_model.predict(img_array_expanded, verbose=0)
        
        # Ambil hasil prediksi
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions[0]) * 100)
        
        # Log prediksi
        print(f"   ✅ Prediksi: {predicted_class} ({confidence:.2f}%)")
        
        return predicted_class, confidence
    except Exception as e:
        raise Exception(f"Error saat prediksi: {str(e)}")

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'BGlow Skin Analysis API',
        'version': '1.1',
        'model_loaded': inference_model is not None,
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Check API health status',
            '/predict': 'POST - Upload image for skin type prediction'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Endpoint untuk check status API dan model"""
    model_status = "loaded" if inference_model is not None else "not loaded"
    
    model_info = {}
    if inference_model is not None:
        model_info = {
            'input_shape': str(inference_model.input_shape),
            'output_shape': str(inference_model.output_shape)
        }
    
    return jsonify({
        'status': 'success',
        'model_status': model_status,
        'available_classes': class_names,
        'model_info': model_info,
        'image_size': IMG_SIZE,
        'tensorflow_version': tf.__version__
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi jenis kulit"""
    try:
        # Validasi model sudah dimuat
        if inference_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model gagal dimuat saat startup. Cek log server untuk detail.',
                'solution': 'Hubungi developer untuk mendapatkan model yang kompatibel'
            }), 500
        
        # Validasi ada file yang diupload
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Tidak ada file gambar yang diupload. Gunakan key "image" untuk upload.'
            }), 400
        
        file = request.files['image']
        
        # Validasi file tidak kosong
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Nama file kosong'
            }), 400
        
        # Validasi ekstensi file
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({
                'status': 'error',
                'message': f'Format file tidak didukung. Gunakan: {", ".join(allowed_extensions)}'
            }), 400
        
        # Baca file sebagai bytes
        img_bytes = file.read()
        
        print(f"📸 Menerima gambar: {file.filename} ({len(img_bytes)/1024:.2f} KB)")
        
        # Prediksi
        predicted_class, confidence = predict_image(img_bytes)
        
        # Siapkan response
        response = {
            'status': 'success',
            'data': {
                'skin_type': predicted_class,
                'confidence': round(confidence, 2),
                'description': skin_descriptions.get(predicted_class, ''),
                'problems': skin_problems.get(predicted_class, []),
                'filename': file.filename
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Jalankan server
    app.run(host='0.0.0.0', port=5000, debug=True)
