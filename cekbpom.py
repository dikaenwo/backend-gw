from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import logging

app = Flask(__name__)

# Setup logging untuk debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/cekbpom", methods=["GET"])
def cekbpom():
    na_number = request.args.get("na")
    if not na_number:
        return jsonify({"error": "Parameter 'na' wajib diisi"}), 400

    url = f"https://cekbpom.pom.go.id/all-produk?query={na_number}"
    logger.info(f"Checking URL: {url}")

    try:
        # Setup Chrome dengan opsi tambahan
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)
        
        # Tunggu sampai tabel muncul atau timeout
        wait = WebDriverWait(driver, 60)  # Timeout 60 detik
        
        try:
            # Tunggu sampai tabel dengan data muncul
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
            logger.info("Table found, waiting additional time for full load...")
            time.sleep(10)  # Tunggu tambahan untuk memastikan semua data loaded
            
        except Exception as e:
            logger.warning(f"Table not found with wait, trying longer sleep: {e}")
            time.sleep(45)  # Fallback ke sleep yang lebih lama
        
        # Ambil HTML setelah JavaScript selesai
        html = driver.page_source
        
        # Debug: Log bagian dari HTML untuk melihat struktur
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table:
            logger.info("Table found in HTML")
        else:
            logger.warning("No table found in HTML")
            
        # Cek apakah ada pesan "tidak ada data"
        no_data_indicators = [
            "tidak ditemukan",
            "no data available",
            "data tidak ada",
            "kosong"
        ]
        
        page_text = soup.get_text().lower()
        for indicator in no_data_indicators:
            if indicator in page_text:
                logger.info(f"No data indicator found: {indicator}")
        
        driver.quit()

        # Parsing data dengan metode yang lebih robust
        rows = soup.select("table tbody tr")
        logger.info(f"Found {len(rows)} rows")

        if not rows:
            # Coba selector alternatif
            alternative_selectors = [
                "table tr",
                ".table tbody tr",
                "[role='table'] tr",
                ".data-table tr"
            ]
            
            for selector in alternative_selectors:
                rows = soup.select(selector)
                if rows:
                    logger.info(f"Found {len(rows)} rows using alternative selector: {selector}")
                    break
            
            if not rows:
                return jsonify({
                    "query": na_number,
                    "results": [],
                    "debug_info": {
                        "html_length": len(html),
                        "has_table": bool(soup.find("table")),
                        "page_title": soup.title.string if soup.title else "No title"
                    }
                })

        results = []
        for i, row in enumerate(rows):
            cols = row.find_all(["td", "th"])
            col_texts = [c.get_text(strip=True) for c in cols]
            
            # Skip header rows
            if not col_texts or len(col_texts) < 4:
                continue
                
            # Skip jika row berisi header text
            if any(header in col_texts[0].lower() for header in ["tipe", "type", "jenis"]):
                continue
            
            logger.info(f"Processing row {i}: {col_texts}")
            
            try:
                # Parsing yang lebih flexible
                if len(col_texts) >= 4:
                    # Pisahkan nomor registrasi & tanggal terbit
                    reg_raw = col_texts[1]
                    nomor_registrasi, tanggal_terbit = None, None
                    
                    if "Terbit:" in reg_raw:
                        parts = reg_raw.split("Terbit:")
                        nomor_registrasi = parts[0].strip()
                        tanggal_terbit = parts[1].strip()
                    else:
                        nomor_registrasi = reg_raw.strip()

                    # Pisahkan nama produk, merek, kemasan
                    produk_raw = col_texts[2] if len(col_texts) > 2 else ""
                    nama_produk, merek, kemasan = produk_raw, None, None
                    
                    if "Merk:" in produk_raw and "Kemasan:" in produk_raw:
                        parts = produk_raw.split("Merk:")
                        nama_produk = parts[0].strip()
                        if len(parts) > 1:
                            sisa = parts[1].split("Kemasan:")
                            merek = sisa[0].strip()
                            if len(sisa) > 1:
                                kemasan = sisa[1].strip()

                    result = {
                        "tipe": col_texts[0],
                        "nomor_registrasi": nomor_registrasi,
                        "tanggal_terbit": tanggal_terbit,
                        "nama_produk": nama_produk,
                        "merek": merek,
                        "kemasan": kemasan,
                        "pendaftar": col_texts[3] if len(col_texts) > 3 else None
                    }
                    
                    # Filter hasil berdasarkan nomor NA yang dicari
                    if na_number.lower() in nomor_registrasi.lower() if nomor_registrasi else False:
                        results.append(result)
                        
            except Exception as e:
                logger.error(f"Error parsing row {i}: {e}")
                continue

        return jsonify({
            "query": na_number,
            "results": results,
            "total_found": len(results),
            "debug_info": {
                "total_rows_processed": len(rows),
                "html_contains_na": na_number in html
            }
        })

    except Exception as e:
        logger.error(f"Error in cekbpom: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/debug/<na_number>", methods=["GET"])
def debug_bpom(na_number):
    """Endpoint untuk debugging - mengembalikan raw HTML"""
    url = f"https://cekbpom.pom.go.id/all-produk?query={na_number}"
    
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)
        time.sleep(45)

        html = driver.page_source
        driver.quit()

        return jsonify({
            "url": url,
            "html_length": len(html),
            "contains_na": na_number in html,
            "html_preview": html[:2000] + "..." if len(html) > 2000 else html
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=True)