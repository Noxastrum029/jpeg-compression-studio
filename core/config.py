"""
config.py — Merkezi Konfigürasyon

Tüm magic number'ları, sabitleri ve varsayılan parametreleri bu dosyada sakla.
Projenin herhangi yerinden import edile ve global ayarlar hemen erişilebilir.
"""

from pathlib import Path

# ============================================================================
# 📁 PROJE YOLLARI
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
STAGES_DIR = PROJECT_ROOT / "stages"
CORE_DIR = PROJECT_ROOT / "core"
UI_DIR = PROJECT_ROOT / "ui"
TESTS_DIR = PROJECT_ROOT / "tests"
OUTPUT_DIR = PROJECT_ROOT / "output"
DOCS_DIR = PROJECT_ROOT / "docs"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
LOGS_DIR = PROJECT_ROOT / "logs"

# Logs klasörünü oluştur (yoksa)
LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# 🖼️ GÖRÜNTÜ İŞLEME SABITLERI
# ============================================================================

# NOT: Orijinal görüntü boyutu korunur — yeniden boyutlandırma yapılmaz
# (Sıkıştırma boyutu değişimi değil, dosya boyutu azalması için)

# Desteklenen dosya formatları
SUPPORTED_FORMATS = ['.png', '.bmp', '.tiff', '.tif', '.jfif', '.jpg', '.jpeg', '.webp']

# Renk uzayı seçenekleri
FORCE_GRAYSCALE_DEFAULT = False  # Renkli görüntülerde YCbCr → Y kanalını kullan

# ============================================================================
# 🌊 DWT (DISCRETE WAVELET TRANSFORM) SABITLERI
# ============================================================================

# Desteklenen dalgacık türleri ve önerileri
WAVELETS = {
    "haar": {
        "name": "Haar",
        "description": "En hızlı, ama kalite düşük",
        "recommended_for": ["quick_demo", "low_quality"],
    },
    "db2": {
        "name": "Daubechies 2",
        "description": "Düşük kalite",
        "recommended_for": ["low_quality"],
    },
    "db4": {
        "name": "Daubechies 4",
        "description": "Dengeli (RECOMMENDED)",
        "recommended_for": ["general", "medical", "balanced"],
    },
    "db6": {
        "name": "Daubechies 6",
        "description": "Yüksek kalite",
        "recommended_for": ["high_quality"],
    },
    "db8": {
        "name": "Daubechies 8",
        "description": "Çok yüksek kalite",
        "recommended_for": ["lossless", "archival"],
    },
    "db32": {
        "name": "Daubechies 32",
        "description": "Aşırı yüksek kalite, çok yavaş",
        "recommended_for": ["archival", "maximum_decomposition"],
    },
    "bior4.4": {
        "name": "Biorthogonal 4.4",
        "description": "JPEG2000'e yakın",
        "recommended_for": ["jpeg2000_compatible", "symmetric"],
    },
}

# DWT boundary mode (JPEG2000 uyumu)
DWT_MODE = "periodization"  # Alternatifler: "symmetric", "smooth", "periodic"

# Ayrıştırma seviyeleri aralığı
MIN_DECOMPOSITION_LEVEL = 1
MAX_DECOMPOSITION_LEVEL = 3
DEFAULT_DECOMPOSITION_LEVEL = 2

# ============================================================================
# 📊 KUANTİZASYON SABITLERI
# ============================================================================

# Kuantalama yöntemi
QUANTIZATION_METHODS = ["uniform", "scalar", "adaptive"]
DEFAULT_QUANTIZATION_METHOD = "uniform"

# Kalite ön ayarları (0-100 ölçeğinde)
QUALITY_PRESETS = {
    "lowest": 0,      # Yoğun sıkıştırma, düşük kalite
    "low": 25,
    "medium": 50,     # Dengeli
    "high": 75,
    "highest": 90,    # Hafif sıkıştırma, yüksek kalite
    "lossless": 100,  # Kayıpsız (çok az sıkıştırma)
}

DEFAULT_QUALITY = 50

# Kalite → Step Size dönüşüm parametreleri
# step = MAX_STEP * (100 - quality) / 100
MAX_STEP_SIZE = 100.0
MIN_STEP_SIZE = 0.1

# Skalar kuantalama: banda göre farklı step faktörleri
SCALAR_QUANTIZATION_FACTORS = {
    "LL": 1.0,      # Yaklaşım (önemli) - küçük step
    "HL": 2.0,      # Yatay detay - orta step
    "LH": 2.0,      # Dikey detay - orta step
    "HH": 4.0,      # Diyagonal detay - büyük step
}

# ============================================================================
# 🔤 HUFFMAN KODLAMA SABITLERI
# ============================================================================

# Huffman bit derinliği
MAX_CODE_LENGTH = 32

# ============================================================================
# 🗂️ BİTSTREAM SABITLERI
# ============================================================================

# .jxc dosya magic byte'ları
MAGIC_BYTES = b"JXC"  # JPEG eXtended Compression (made-up)
FILE_VERSION = 1

# Header boyutu (sabit kısım)
HEADER_FIXED_SIZE = 16  # bytes

# ============================================================================
# 📈 PERFORMANCE METRIKLERI
# ============================================================================

# PSNR hesaplamada maksimum değer
PSNR_MAX_VALUE = 255.0

# SSIM (Structural Similarity Index) windowsize
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5

# ============================================================================
# 🧪 TEST SABITLERI
# ============================================================================

# Test resim boyutu (hızlı test için)
TEST_IMAGE_SIZE = (256, 256)

# Benchmark image listesi
BENCHMARK_IMAGES = [
    "test_images_generated/test_natural_landscape.jpg",   # Natural (simulated)
    "test_images_generated/test_synthetic_graphic.jpg",   # Synthetic
    "test_images_generated/test_hybrid_mixed.jpg",        # Hybrid
]

# ============================================================================
# 🎨 GUI SABITLERI
# ============================================================================

# CustomTkinter tema
GUI_APPEARANCE = "light"
GUI_COLOR_THEME = "blue"

# Başlangıç pencere boyutu
DEFAULT_WINDOW_WIDTH = 1320
DEFAULT_WINDOW_HEIGHT = 820
MIN_WINDOW_WIDTH = 1120
MIN_WINDOW_HEIGHT = 720

# Panel genişlikleri (sol=20%, orta=50%, sağ=30%)
LEFT_PANEL_WIDTH = 308
CENTER_PANEL_WEIGHT = 1
RIGHT_PANEL_WIDTH = 380

# Renkler
COLOR_BG_MAIN = "#E8EEF2"
COLOR_PANEL_BG = "#F4F7FA"
COLOR_ACCENT = "#1E88E5"
COLOR_SUCCESS = "#4CAF50"
COLOR_WARNING = "#FF9800"
COLOR_ERROR = "#F44336"
COLOR_SEPARATOR = "#CFD8DC"

# ============================================================================
# 📝 LOGGING SABITLERI
# ============================================================================

# Log level
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Log dosya formatı
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# 🔧 CLI SABITLERI
# ============================================================================

# CLI program adı
CLI_PROGRAM_NAME = "jpeg2000-compressor"

# ============================================================================
# 📊 DEFAULT KONFIGÜRASYON DİKTİONERİ
# ============================================================================

DEFAULT_CONFIG = {
    "image": {
        "target_size": None,  # Orijinal boyut korunur
        "supported_formats": SUPPORTED_FORMATS,
        "force_grayscale": FORCE_GRAYSCALE_DEFAULT,
    },
    "dwt": {
        "wavelet": "db4",
        "level": DEFAULT_DECOMPOSITION_LEVEL,
        "mode": DWT_MODE,
    },
    "quantization": {
        "method": DEFAULT_QUANTIZATION_METHOD,
        "quality": DEFAULT_QUALITY,
        "step_size": None,  # Kaliteden otomatik hesaplanır
    },
    "output": {
        "format": "jxc",
        "save_visualization": True,
        "output_dir": str(OUTPUT_DIR),
    },
    "logging": {
        "level": LOG_LEVEL,
        "file": str(LOGS_DIR / "compression.log"),
    },
}

# ============================================================================
# 🚀 DİĞER SABİTLER
# ============================================================================

# Bellek tahmini için çarpan
MEMORY_MULTIPLIER = 1.5  # DWT processing için ~1.5x bellek gerekir

# ============================================================================
# ✅ CONFIG VALIDATION
# ============================================================================

def validate_config():
    """Konfigürasyonun tutarlı olduğunu kontrol et"""
    assert MIN_STEP_SIZE < MAX_STEP_SIZE, "MIN_STEP_SIZE < MAX_STEP_SIZE olmalı"
    assert DEFAULT_QUALITY in range(0, 101), "DEFAULT_QUALITY 0-100 arasında olmalı"
    assert MIN_DECOMPOSITION_LEVEL <= MAX_DECOMPOSITION_LEVEL, "Decomposition level aralığı hatalı"
    assert all(Path(p) is not None for p in [STAGES_DIR, TESTS_DIR, OUTPUT_DIR])
    return True


# Modülü import ettiğinde validasyon yap
if __name__ != "__main__":
    try:
        validate_config()
    except AssertionError as e:
        raise ValueError(f"Konfigürasyon hatasız: {e}")

if __name__ == "__main__":
    print("✅ Konfigürasyon tamam!")
    print(f"Proje kökü: {PROJECT_ROOT}")
    print(f"Varsayılan kalite: {DEFAULT_QUALITY}")
    print(f"Varsayılan dalgacık: {list(WAVELETS.keys())}")
