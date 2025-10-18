from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# =======================
# 1. Load semua rule file
# =======================
try:
    jenis_cocok = pd.read_csv("jenis-kulit-cocok.csv")
    jenis_tidak = pd.read_csv("jenis-kulit-tidak-cocok.csv")
    masalah_cocok = pd.read_csv("masalah-kulit-cocok.csv")
    masalah_tidak = pd.read_csv("masalah-kulit-tidak-cocok.csv")
    produk_df = pd.read_csv("product.csv")
    print("✅ Semua file CSV berhasil dimuat")
    
    # Debug: Print kolom yang tersedia
    print("\n📋 Kolom di Jenis Kulit Cocok:", jenis_cocok.columns.tolist())
    print("📋 Kolom di Jenis Kulit Tidak Cocok:", jenis_tidak.columns.tolist())
    print("📋 Kolom di Masalah Kulit Cocok:", masalah_cocok.columns.tolist())
    print("📋 Kolom di Masalah Kulit Tidak Cocok:", masalah_tidak.columns.tolist())
    
except Exception as e:
    print(f"❌ Error loading CSV files: {e}")
    produk_df = pd.DataFrame()


# Mapping kategori CSV ke display name
CATEGORY_MAPPING = {
    "Facial Wash": "Cleanser",
    "Moisturizer": "Pelembab",
    "Serum": "Serum",
    "Sunscreen": "Sunscreen",
    "Exfo": "Eksfoliasi"
}

# =======================
# 2. Fungsi Bobot & Analisis
# =======================
def bobot_posisi(index, total):
    """Memberikan bobot berdasarkan posisi ingredient dalam list"""
    persen = index / total
    if persen <= 0.2:
        return 1.0
    elif persen <= 0.5:
        return 0.5
    else:
        return 0.2

def analisis_produk(produk, rules_cocok, rules_tidak_cocok):
    """Menganalisis produk berdasarkan ingredient dan rules"""
    ingredients_list = [i.strip() for i in str(produk["Ingridients"]).split(",")]
    total = len(ingredients_list)
    cocok_found, tidak_cocok_found = [], []
    score = 0

    for idx, ingr in enumerate(ingredients_list):
        weight = bobot_posisi(idx, total)

        if ingr in rules_cocok["Ingredient"].values:
            cocok_found.append(f"{ingr} (+{weight:.1f})")
            score += 1 * weight

        if ingr in rules_tidak_cocok["Ingredient"].values:
            tidak_cocok_found.append(f"{ingr} (-{2*weight:.1f})")
            score -= 2 * weight

    # Ambil URL gambar dari kolom "Gambar"
    image_url = str(produk.get("Gambar", "")).strip()
    if not image_url or image_url.lower() == "nan":
        image_url = "https://via.placeholder.com/300x300.png?text=No+Image"
    
    # Pastikan harga adalah integer
    try:
        harga = int(float(produk["Harga"]))
    except (ValueError, TypeError):
        harga = 0
    
    # Get kategori dan display name
    kategori = str(produk["Kategori"]).strip()
    kategori_display = CATEGORY_MAPPING.get(kategori, kategori)
    
    return {
        "nama_produk": produk["Nama Produk"],
        "kategori": kategori,
        "kategori_display": kategori_display,
        "harga": harga,
        "cocok": cocok_found,
        "tidak_cocok": tidak_cocok_found,
        "skor": round(score, 2),
        "rekomendasi": "Direkomendasikan" if score > 0 else "Tidak Direkomendasikan",
        "image_url": image_url
    }

def get_ingredient_benefit(ingredient, rules_cocok):
    """
    Get manfaat dari ingredient yang cocok
    Cari di rules_cocok yang punya kolom 'Manfaat'
    """
    matched = rules_cocok[rules_cocok["Ingredient"] == ingredient]
    if not matched.empty:
        # Ambil manfaat dari kolom 'Manfaat'
        manfaat = str(matched.iloc[0].get("Manfaat", "")).strip()
        if manfaat and manfaat.lower() != "nan":
            return manfaat
    return "Baik untuk kulit"

def get_ingredient_side_effect(ingredient, rules_tidak_cocok):
    """
    Get efek samping dari ingredient yang tidak cocok
    Cari di rules_tidak_cocok yang punya kolom 'Efek Samping'
    """
    matched = rules_tidak_cocok[rules_tidak_cocok["Ingredient"] == ingredient]
    if not matched.empty:
        # Ambil efek samping dari kolom 'Efek Samping' atau 'Efek_Samping'
        efek_samping = str(matched.iloc[0].get("Efek Samping", 
                          matched.iloc[0].get("Efek_Samping", ""))).strip()
        if efek_samping and efek_samping.lower() != "nan":
            return efek_samping
    return "Dapat menyebabkan iritasi"

# =======================
# 3. API Endpoints
# =======================
@app.route('/', methods=['GET'])
def home():
    """Endpoint home untuk testing"""
    return jsonify({
        "message": "API Rekomendasi Produk Skincare",
        "version": "2.2",
        "endpoints": {
            "/api/recommend": "POST - Get product recommendations with filters",
            "/api/product-detail": "POST - Get detailed product information",
            "/api/skin-types": "GET - Get available skin types",
            "/api/skin-problems": "GET - Get available skin problems"
        }
    })

@app.route('/api/recommend', methods=['POST'])
def recommend_products():
    """
    Endpoint utama untuk mendapatkan rekomendasi produk
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        jenis_kulit = data.get('jenis_kulit', '')
        masalah_kulit_list = data.get('masalah_kulit', [])
        filter_kategori = data.get('kategori')
        filter_min_harga = data.get('min_harga')
        filter_max_harga = data.get('max_harga')
        
        if not jenis_kulit:
            return jsonify({"error": "jenis_kulit is required"}), 400
        
        if not masalah_kulit_list or not isinstance(masalah_kulit_list, list):
            return jsonify({"error": "masalah_kulit must be a non-empty array"}), 400
        
        rules_cocok_list = []
        rules_tidak_cocok_list = []
        
        rules_cocok_list.append(jenis_cocok[jenis_cocok["Jenis_Kulit"] == jenis_kulit])
        rules_tidak_cocok_list.append(jenis_tidak[jenis_tidak["Jenis_Kulit"] == jenis_kulit])
        
        for masalah in masalah_kulit_list:
            rules_cocok_list.append(masalah_cocok[masalah_cocok["Masalah_Kulit"] == masalah])
            rules_tidak_cocok_list.append(masalah_tidak[masalah_tidak["Masalah_Kulit"] == masalah])
        
        rules_cocok = pd.concat(rules_cocok_list, ignore_index=True)
        rules_tidak_cocok = pd.concat(rules_tidak_cocok_list, ignore_index=True)
        
        filtered_produk = produk_df.copy()
        if filter_kategori:
            filtered_produk = filtered_produk[filtered_produk["Kategori"] == filter_kategori]
        
        if filter_min_harga is not None:
            filtered_produk = filtered_produk[filtered_produk["Harga"] >= filter_min_harga]
        
        if filter_max_harga is not None:
            filtered_produk = filtered_produk[filtered_produk["Harga"] <= filter_max_harga]
        
        hasil_semua = []
        for _, row in filtered_produk.iterrows():
            hasil = analisis_produk(row, rules_cocok, rules_tidak_cocok)
            hasil_semua.append(hasil)
        
        hasil_sorted = sorted(hasil_semua, key=lambda x: x["skor"], reverse=True)
        produk_direkomendasikan = [p for p in hasil_sorted if p["rekomendasi"] == "Direkomendasikan"]
        
        return jsonify({
            "success": True,
            "input": {
                "jenis_kulit": jenis_kulit,
                "masalah_kulit": masalah_kulit_list,
                "filter_kategori": filter_kategori,
                "filter_harga": {
                    "min": filter_min_harga,
                    "max": filter_max_harga
                }
            },
            "total_produk_dianalisis": len(hasil_semua),
            "total_produk_direkomendasikan": len(produk_direkomendasikan),
            "rekomendasi": produk_direkomendasikan[:20]
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/product-detail', methods=['POST'])
def get_product_detail():
    """
    Endpoint untuk mendapatkan detail lengkap produk
    
    Expected JSON body:
    {
        "nama_produk": "Skintific 5X Ceramide...",
        "jenis_kulit": "Kombinasi",
        "masalah_kulit": ["Hiperpigmentasi", "Jerawat"]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        nama_produk = data.get('nama_produk', '')
        jenis_kulit = data.get('jenis_kulit', '')
        masalah_kulit_list = data.get('masalah_kulit', [])
        
        if not nama_produk:
            return jsonify({"error": "nama_produk is required"}), 400
        
        # Cari produk di database
        produk = produk_df[produk_df["Nama Produk"] == nama_produk]
        
        if produk.empty:
            return jsonify({
                "success": False,
                "error": "Product not found"
            }), 404
        
        produk_row = produk.iloc[0]
        
        # Get rules untuk analisis
        rules_cocok_list = []
        rules_tidak_cocok_list = []
        
        if jenis_kulit:
            rules_cocok_list.append(jenis_cocok[jenis_cocok["Jenis_Kulit"] == jenis_kulit])
            rules_tidak_cocok_list.append(jenis_tidak[jenis_tidak["Jenis_Kulit"] == jenis_kulit])
        
        if masalah_kulit_list:
            for masalah in masalah_kulit_list:
                rules_cocok_list.append(masalah_cocok[masalah_cocok["Masalah_Kulit"] == masalah])
                rules_tidak_cocok_list.append(masalah_tidak[masalah_tidak["Masalah_Kulit"] == masalah])
        
        rules_cocok = pd.concat(rules_cocok_list, ignore_index=True) if rules_cocok_list else pd.DataFrame()
        rules_tidak_cocok = pd.concat(rules_tidak_cocok_list, ignore_index=True) if rules_tidak_cocok_list else pd.DataFrame()
        
        # Parse ingredients
        ingredients_list = [i.strip() for i in str(produk_row["Ingridients"]).split(",")]
        total = len(ingredients_list)
        
        # Analisis ingredients
        ingredients_cocok = []
        ingredients_tidak_cocok = []
        
        for idx, ingr in enumerate(ingredients_list):
            weight = bobot_posisi(idx, total)
            
            # Cek ingredient yang COCOK
            if not rules_cocok.empty and ingr in rules_cocok["Ingredient"].values:
                manfaat = get_ingredient_benefit(ingr, rules_cocok)
                ingredients_cocok.append({
                    "kandungan": ingr,
                    "manfaat": manfaat,  # Menggunakan kolom Manfaat
                    "skor": round(1 * weight, 1)
                })
            
            # Cek ingredient yang TIDAK COCOK
            if not rules_tidak_cocok.empty and ingr in rules_tidak_cocok["Ingredient"].values:
                efek_samping = get_ingredient_side_effect(ingr, rules_tidak_cocok)
                ingredients_tidak_cocok.append({
                    "kandungan": ingr,
                    "manfaat": efek_samping,  # Menggunakan kolom Efek Samping (tapi key tetap 'manfaat' untuk konsistensi frontend)
                    "skor": round(2 * weight, 1)
                })
        
        # Sort by score
        ingredients_cocok = sorted(ingredients_cocok, key=lambda x: x["skor"], reverse=True)
        ingredients_tidak_cocok = sorted(ingredients_tidak_cocok, key=lambda x: x["skor"], reverse=True)
        
        # Calculate overall score
        total_score = sum([i["skor"] for i in ingredients_cocok]) - sum([i["skor"] for i in ingredients_tidak_cocok])
        
        # Get other product info
        try:
            harga = int(float(produk_row["Harga"]))
        except (ValueError, TypeError):
            harga = 0
        
        image_url = str(produk_row.get("Gambar", "")).strip()
        if not image_url or image_url.lower() == "nan":
            image_url = "https://via.placeholder.com/300x300.png?text=No+Image"
        
        link_produk = str(produk_row.get("Link_Produk", "")).strip()
        if not link_produk or link_produk.lower() == "nan":
            link_produk = ""
        
        tekstur = str(produk_row.get("Tekstur", "")).strip()
        if tekstur.lower() == "nan":
            tekstur = "Tidak ada informasi tekstur"
        
        kategori = str(produk_row["Kategori"]).strip()
        kategori_display = CATEGORY_MAPPING.get(kategori, kategori)
        
        return jsonify({
            "success": True,
            "product": {
                "nama_produk": produk_row["Nama Produk"],
                "kategori": kategori,
                "kategori_display": kategori_display,
                "harga": harga,
                "image_url": image_url,
                "link_produk": link_produk,
                "tekstur": tekstur,
                "ingredients_overview": ", ".join(ingredients_list),
                "ingredients_cocok": ingredients_cocok,
                "ingredients_tidak_cocok": ingredients_tidak_cocok,
                "skor_total": round(total_score, 2),
                "rekomendasi": "Direkomendasikan" if total_score > 0 else "Tidak Direkomendasikan"
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/skin-types', methods=['GET'])
def get_skin_types():
    """Endpoint untuk mendapatkan daftar jenis kulit yang tersedia"""
    try:
        skin_types = jenis_cocok["Jenis_Kulit"].unique().tolist()
        return jsonify({
            "success": True,
            "skin_types": skin_types
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/skin-problems', methods=['GET'])
def get_skin_problems():
    """Endpoint untuk mendapatkan daftar masalah kulit yang tersedia"""
    try:
        skin_problems = masalah_cocok["Masalah_Kulit"].unique().tolist()
        return jsonify({
            "success": True,
            "skin_problems": skin_problems
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =======================
# 4. Run Server
# =======================
if __name__ == '__main__':
    print("🚀 Starting Skincare Recommendation API v2.2...")
    print("📝 Make sure all CSV files are in the same directory")
    print("✨ New features:")
    print("   - Using 'Manfaat' column for beneficial ingredients")
    print("   - Using 'Efek Samping' column for harmful ingredients")
    print("   - Improved ingredient analysis")
    app.run(host='0.0.0.0', port=9000, debug=True)