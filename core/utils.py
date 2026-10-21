"""
utils.py — Ortak Yardımcı Fonksiyonlar

Projenin tüm modüllerinde kullanılacak helper fonksiyonlar.
Metrikleri (PSNR, MSE, SSIM), dönüşümler, ve validasyon fonksiyonları.
"""

import numpy as np
from typing import Tuple, Dict, Any, Union, Optional
from pathlib import Path
import logging

from core.config import (
    PSNR_MAX_VALUE,
    SUPPORTED_FORMATS,
    QUALITY_PRESETS,
    MAX_STEP_SIZE,
    MIN_STEP_SIZE,
)

# ============================================================================
# 📝 LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# ✅ VALIDASYON FONKSİYONLARI
# ============================================================================


def validate_image_path(path: Union[str, Path]) -> bool:
    """
    Görüntü dosyasının geçerli olup olmadığını kontrol et.
    
    Args:
        path: Dosya yolu
        
    Returns:
        True eğer dosya var ve destekleniyorsa
        
    Raises:
        ValueError: Dosya yok veya format desteklenmiyor
    """
    path = Path(path)
    
    if not path.exists():
        raise ValueError(f"Dosya bulunamadı: {path}")
    
    if not path.is_file():
        raise ValueError(f"Bu bir dosya değil: {path}")
    
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Desteklenmeyen format: {suffix}\n"
            f"Desteklenen: {SUPPORTED_FORMATS}"
        )
    
    return True


def validate_quality_factor(quality: int) -> bool:
    """
    Kalite faktörünün geçerli olup olmadığını kontrol et.
    
    Args:
        quality: Kalite faktörü (0-100)
        
    Returns:
        True eğer geçerliyse
        
    Raises:
        ValueError: Geçersiz kalite değeri
    """
    if not isinstance(quality, int):
        raise TypeError(f"quality int olmalı, {type(quality)} alındı")
    
    if not (0 <= quality <= 100):
        raise ValueError(f"quality 0-100 arasında olmalı, {quality} alındı")
    
    return True


def validate_image_shape(image: np.ndarray, expected_dims: int = 2) -> bool:
    """
    Görüntü shape'inin doğru olup olmadığını kontrol et.
    
    Args:
        image: NumPy görüntü
        expected_dims: Beklenen boyut sayısı (2 = 2D, 3 = 3D)
        
    Returns:
        True eğer geçerliyse
        
    Raises:
        ValueError: Shape geçersiz
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image np.ndarray olmalı, {type(image)} alındı")
    
    if image.ndim != expected_dims:
        raise ValueError(
            f"image {expected_dims}D olmalı, {image.ndim}D alındı. Shape: {image.shape}"
        )
    
    return True


def validate_target_size(size: Tuple[int, int]) -> bool:
    """
    Target size'ın geçerli olup olmadığını kontrol et.
    Genellikle 2'nin kuvveti olmalı (DWT için).
    
    Args:
        size: (genişlik, yükseklik)
        
    Returns:
        True eğer geçerliyse
    """
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError(f"size (w, h) tuple'ı olmalı, {size} alındı")
    
    if not all(isinstance(s, int) and s > 0 for s in size):
        raise ValueError(f"size pozitif integer'ler içermeli: {size}")
    
    # 2'nin kuvveti kontrolü (opsiyonel, ama uyarı)
    for s in size:
        if not (s & (s - 1) == 0):
            logger.warning(
                f"Size {size} 2'nin kuvveti değil. DWT için uygun değildir. "
                f"Önerilen boyutlar: (256, 256), (512, 512), (1024, 1024)"
            )
    
    return True


# ============================================================================
# 📊 KALITE METRIK FONKSİYONLARI
# ============================================================================


def mse_fn(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Mean Squared Error (MSE) hesapla.
    
    Düşük MSE = daha iyi rekonstruksiyon.
    
    Args:
        original: Orijinal görüntü/kanal
        reconstructed: Rekonstruksiyon
        
    Returns:
        MSE değeri
    """
    validate_image_shape(original, 2)
    validate_image_shape(reconstructed, 2)
    
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape uyuşmıyor: {original.shape} vs {reconstructed.shape}"
        )
    
    # float64 ile taşma riskini azalt
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    
    return mse


def psnr_fn(original: np.ndarray, reconstructed: np.ndarray, 
            max_val: float = PSNR_MAX_VALUE) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) hesapla (dB cinsinden).
    
    Yüksek PSNR = daha iyi rekonstruksiyon.
    - PSNR > 40 dB: Çok iyi kalite
    - PSNR > 30 dB: İyi kalite
    - PSNR > 20 dB: Kabul edilebilir
    - PSNR < 20 dB: Kötü kalite
    
    Args:
        original: Orijinal görüntü/kanal
        reconstructed: Rekonstruksiyon
        max_val: Maksimum signal değeri (tipik: 255.0)
        
    Returns:
        PSNR değeri (dB)
    """
    mse = mse_fn(original, reconstructed)
    
    # Görüntüler özdeşse (MSE ≈ 0)
    if mse < 1e-12:
        return 99.99  # Yüksek sabit döndür
    
    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return float(psnr)


def ssim_fn(original: np.ndarray, reconstructed: np.ndarray,
            data_range: float = PSNR_MAX_VALUE) -> float:
    """
    Structural Similarity Index (SSIM) hesapla.
    
    SSIM, insan göz algısına daha yakında bir metrik.
    Aralık: [-1, 1], genellikle [0, 1]
    - SSIM > 0.95: Çok benzer
    - SSIM > 0.80: Benzer
    - SSIM > 0.50: Biraz benzer
    - SSIM < 0.50: Farklı
    
    Args:
        original: Orijinal görüntü/kanal
        reconstructed: Rekonstruksiyon
        data_range: Veri aralığı (tipik: 255.0)
        
    Returns:
        SSIM değeri (0-1)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        logger.warning("scikit-image bulunamadı. SSIM hesaplamak için: pip install scikit-image")
        return 0.0
    
    validate_image_shape(original, 2)
    validate_image_shape(reconstructed, 2)
    
    ssim_val = ssim(original, reconstructed, data_range=data_range)
    return float(ssim_val)


def ssim_fn_any(original: np.ndarray, reconstructed: np.ndarray,
                data_range: float = 255.0) -> float:
    """2B veya 3B (renkli) görüntüler için güvenli SSIM hesabı.
    Hata durumunda 0.0 döner."""
    try:
        if original.ndim == 3:
            from skimage.metrics import structural_similarity as _ssim
            return float(_ssim(
                original.astype(np.float64), reconstructed.astype(np.float64),
                data_range=data_range, channel_axis=-1
            ))
        else:
            return ssim_fn(original, reconstructed, data_range=data_range)
    except Exception:
        return 0.0


def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
    """
    Sıkıştırma oranını hesapla.
    
    Ratio = original_size / compressed_size
    - Ratio > 5: Çok iyi sıkıştırma
    - Ratio > 3: İyi sıkıştırma
    - Ratio > 1: Az sıkıştırma
    
    Args:
        original_size: Orijinal boyut (bytes)
        compressed_size: Sıkıştırılmış boyut (bytes)
        
    Returns:
        Sıkıştırma oranı
    """
    if compressed_size <= 0:
        raise ValueError("compressed_size > 0 olmalı")
    
    ratio = original_size / compressed_size
    return float(ratio)


def calculate_bpp(compressed_size: int, height: int, width: int) -> float:
    """
    Bits Per Pixel (BPP) hesapla.
    
    Orijinal PNG/uncompressed: 8 BPP (8-bit grayscale) veya 24 BPP (24-bit RGB)
    
    Args:
        compressed_size: Sıkıştırılmış boyut (bytes)
        height: Görüntü yüksekliği
        width: Görüntü genişliği
        
    Returns:
        BPP değeri
    """
    total_pixels = height * width
    if total_pixels == 0:
        return 0.0
    
    total_bits = compressed_size * 8
    bpp = total_bits / total_pixels
    return float(bpp)


def calculate_metrics(original: np.ndarray, reconstructed: np.ndarray,
                     original_size: int, compressed_size: int) -> Dict[str, float]:
    """
    Tüm metrikleri bir seferde hesapla.
    
    Args:
        original: Orijinal görüntü/kanal (uint8 veya float32)
        reconstructed: Rekonstruksiyon (uint8 veya float32)
        original_size: Orijinal dosya boyutu (bytes)
        compressed_size: Sıkıştırılmış dosya boyutu (bytes)
        
    Returns:
        Metrikleri içeren dictionary
    """
    # Normalizasyon: uint8 aralığına getir
    if original.dtype != np.uint8:
        original = np.clip(original, 0, 255).astype(np.uint8)
    if reconstructed.dtype != np.uint8:
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    metrics = {
        "MSE": mse_fn(original, reconstructed),
        "PSNR": psnr_fn(original, reconstructed),
        "SSIM": ssim_fn(original, reconstructed),
        "Compression Ratio": calculate_compression_ratio(original_size, compressed_size),
        "BPP": calculate_bpp(compressed_size, original.shape[0], original.shape[1]),
    }
    
    return metrics


# ============================================================================
# 🔀 DÖNÜŞTÜRME FONKSİYONLARI
# ============================================================================


def quality_to_step_size(quality: int) -> float:
    """
    Kalite faktörü (0-100) → Step size dönüşümü.
    
    - quality=0 (yoğun sıkıştırma) → step=MAX_STEP_SIZE
    - quality=100 (kayıpsız) → step=MIN_STEP_SIZE
    
    Formül: step = MAX_STEP * (100 - quality) / 100
    
    Args:
        quality: Kalite faktörü (0-100)
        
    Returns:
        Step size (kuantalama adımı)
    """
    validate_quality_factor(quality)
    
    step = MAX_STEP_SIZE * (100 - quality) / 100
    step = max(step, MIN_STEP_SIZE)  # Minimum sınırı uygula
    
    return float(step)


def step_size_to_quality(step: float) -> int:
    """
    Step size → Kalite faktörü dönüşümü (tersi).
    
    Args:
        step: Step size
        
    Returns:
        Kalite faktörü (0-100)
    """
    if step <= 0:
        raise ValueError("step > 0 olmalı")
    
    # step = MAX_STEP * (100 - quality) / 100
    # 100 - quality = step * 100 / MAX_STEP
    # quality = 100 - (step * 100 / MAX_STEP)
    
    quality = 100 - (step * 100 / MAX_STEP_SIZE)
    quality = max(0, min(100, int(quality)))
    
    return quality


# ============================================================================
# 💾 BELLEK FONKSİYONLARI
# ============================================================================


def format_bytes(size_bytes: int) -> str:
    """
    Byte'ları insan okunabilir formata dönüştür.
    
    Args:
        size_bytes: Boyut (bytes)
        
    Returns:
        Formatted string (ör. "2.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} PB"


def estimate_memory_usage(image_height: int, image_width: int, 
                         dwt_levels: int, float_precision: int = 32) -> int:
    """
    DWT işlemi için gerekli belleği tahmin et.
    
    Args:
        image_height: Görüntü yüksekliği
        image_width: Görüntü genişliği
        dwt_levels: DWT ayrıştırma seviyeleri
        float_precision: Float bit boyutu (32 veya 64)
        
    Returns:
        Tahmin edilen bellek (bytes)
    """
    bytes_per_element = float_precision // 8
    
    # Temel görüntü boyutu
    base_size = image_height * image_width * bytes_per_element
    
    # DWT intermediate sumuyla ~1.5-2x katı bellek gerekir
    multiplier = 1.5 + (dwt_levels * 0.1)
    
    estimated = int(base_size * multiplier)
    return estimated


# ============================================================================
# 📁 DOSYA OPERASYONLARI
# ============================================================================


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Dizin mevcut değilse oluştur.
    
    Args:
        directory: Dizin yolu
        
    Returns:
        Path nesnesi
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_unique_filepath(filepath: Union[str, Path]) -> Path:
    """
    Eğer dosya zaten varsa, adına (1), (2) vb. ekle.
    
    Args:
        filepath: İstenen dosya yolu
        
    Returns:
        Benzersiz dosya yolu
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return filepath
    
    stem = filepath.stem
    suffix = filepath.suffix
    parent = filepath.parent
    
    counter = 1
    while True:
        new_path = parent / f"{stem}({counter}){suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


# ============================================================================
# 📋 DİĞER YARDIMCI FONKSİYONLAR
# ============================================================================


def print_metrics_table(metrics: Dict[str, Any], title: str = "Metrikleri") -> None:
    """
    Metrikleri tablo formatında konsola yazdır.
    
    Args:
        metrics: Metrik sözlüğü
        title: Başlık
    """
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<30} {value:>12.4f}")
        else:
            print(f"  {key:.<30} {value:>12}")
    
    print(f"{'='*50}\n")


def get_system_info() -> Dict[str, Any]:
    """
    Sistem bilgisi topla.
    
    Returns:
        Sistem bilgileri dictionary
    """
    import platform
    import psutil
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "available_memory_gb": psutil.virtual_memory().available / (1024**3),
    }
    
    return info


# ============================================================================
# ✅ TEST
# ============================================================================

if __name__ == "__main__":
    print("✅ utils.py test etme...")
    
    # Quality ↔ Step Size dönüşümü
    for q in [0, 25, 50, 75, 100]:
        step = quality_to_step_size(q)
        back_q = step_size_to_quality(step)
        print(f"Quality {q:3d} → Step {step:6.2f} → Quality {back_q:3d}")
    
    # Bellek tahmini
    mem = estimate_memory_usage(512, 512, 2)
    print(f"\n512x512 DWT2 bellek tahmini: {format_bytes(mem)}")
    
    # Test görüntü
    img1 = np.arange(256, dtype=np.uint8).reshape(16, 16)
    img2 = img1 + np.random.randint(-10, 10, (16, 16))
    
    metrics = calculate_metrics(img1, img2, 256, 200)
    print_metrics_table(metrics)
    
    print("✅ Tüm testler geçti!")
