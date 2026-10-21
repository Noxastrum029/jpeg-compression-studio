"""
input_validation.py — Comprehensive Input Validation & Error Handling

JPEG-XC sisteminde kullanıcı input'larını doğrulamak için utility fonksiyonları.
"""

from pathlib import Path
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'jfif', 'jxc']
MIN_IMAGE_SIZE = 64  # pixels
MAX_IMAGE_SIZE = 8192  # pixels
MIN_QUALITY = 0
MAX_QUALITY = 100
VALID_DWT_LEVELS = [1, 2, 3]
VALID_QUANTIZATION_METHODS = ['uniform', 'scalar', 'adaptive']


def validate_input_file(file_path: Union[str, Path]) -> bool:
    """
    Giriş dosyasını doğrula
    
    Args:
        file_path: Dosya yolu
        
    Returns:
        True if valid
        
    Raises:
        FileNotFoundError: Dosya bulunamadı
        ValueError: Geçersiz format veya boyut
    """
    file_path = Path(file_path)
    
    # 1. Dosya var mı?
    if not file_path.exists():
        raise FileNotFoundError(f"❌ Dosya bulunamadı: {file_path}")
    
    # 2. Dosya mı? (directory değil)
    if not file_path.is_file():
        raise ValueError(f"❌ Bu bir dosya değildir: {file_path}")
    
    # 3. Dosya boş mu?
    if file_path.stat().st_size == 0:
        raise ValueError(f"❌ Dosya boş: {file_path}")
    
    # 4. Desteklenen format mı?
    extension = file_path.suffix.lower().lstrip('.')
    if extension not in SUPPORTED_FORMATS:
        raise ValueError(
            f"❌ Desteklenmeyen format: {extension}\n"
            f"   Desteklenenleri: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # 5. Dosya boyutu makul mı? (> 100MB check)
    max_file_size = 100 * 1024 * 1024  # 100MB
    if file_path.stat().st_size > max_file_size:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        raise ValueError(
            f"❌ Dosya çok büyük: {size_mb:.1f}MB\n"
            f"   Maksimum: 100MB"
        )
    
    return True


def validate_output_path(output_path: Union[str, Path]) -> bool:
    """
    Çıkış dosyası yolunu doğrula
    
    Args:
        output_path: Çıkış dosyası yolu
        
    Returns:
        True if valid
        
    Raises:
        ValueError: Geçersiz yol veya permission
    """
    output_path = Path(output_path)
    
    # 1. Çıkış dizini oluşturulabilir mi?
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise ValueError(
            f"❌ Yazma izni yok: {output_path.parent}\n"
            f"   Lütfen farklı bir klasör seçin"
        )
    except OSError as e:
        raise ValueError(f"❌ Dizin oluşturulamadı: {e}")
    
    # 2. JXC dosyası mı?
    extension = output_path.suffix.lower()
    if extension and extension != '.jxc':
        logger.warning(f"⚠️  Çıkış dosyası .jxc uzantısı taşımıyor: {extension}")
    
    return True


def validate_parameters(
    dwt_level: int,
    quality: int,
    wavelet: str,
    quantization_method: str
) -> bool:
    """
    Sıkıştırma parametrelerini doğrula
    
    Args:
        dwt_level: DWT seviyeleri (1-3)
        quality: Kalite faktörü (0-100)
        wavelet: Dalgacık türü
        quantization_method: Quantization yöntemi
        
    Returns:
        True if valid
        
    Raises:
        ValueError: Geçersiz parametre
    """
    errors = []
    
    # DWT Level
    if dwt_level not in VALID_DWT_LEVELS:
        errors.append(
            f"❌ DWT seviyesi geçersiz: {dwt_level}\n"
            f"   Geçerli değerler: {VALID_DWT_LEVELS}"
        )
    
    # Quality
    if not isinstance(quality, int) or quality < MIN_QUALITY or quality > MAX_QUALITY:
        errors.append(
            f"❌ Kalite geçersiz: {quality}\n"
            f"   Aralık: {MIN_QUALITY}-{MAX_QUALITY}"
        )
    
    # Wavelet
    from core.config import WAVELETS
    if wavelet not in WAVELETS:
        errors.append(
            f"❌ Dalgacık türü geçersiz: {wavelet}\n"
            f"   Geçerli: {', '.join(WAVELETS.keys())}"
        )
    
    # Quantization method
    if quantization_method not in VALID_QUANTIZATION_METHODS:
        errors.append(
            f"❌ Quantization yöntemi geçersiz: {quantization_method}\n"
            f"   Geçerli: {', '.join(VALID_QUANTIZATION_METHODS)}"
        )
    
    if errors:
        raise ValueError('\n'.join(errors))
    
    return True


def validate_image_dimensions(
    height: int,
    width: int
) -> bool:
    """
    Görüntü boyutlarını doğrula
    
    Args:
        height: Yükseklik
        width: Genişlik
        
    Returns:
        True if valid
        
    Raises:
        ValueError: Boyutlar geçersiz
    """
    errors = []
    
    for dim, val in [("Yükseklik", height), ("Genişlik", width)]:
        if val < MIN_IMAGE_SIZE:
            errors.append(
                f"❌ {dim} çok küçük: {val}px\n"
                f"   Minimum: {MIN_IMAGE_SIZE}px"
            )
        if val > MAX_IMAGE_SIZE:
            errors.append(
                f"❌ {dim} çok büyük: {val}px\n"
                f"   Maksimum: {MAX_IMAGE_SIZE}px"
            )
    
    if errors:
        raise ValueError('\n'.join(errors))
    
    return True
