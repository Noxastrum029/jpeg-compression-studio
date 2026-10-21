"""
compression_modes.py — Compression Mode Management

Sıkıştırma modları:
- Lossy: Kalite kaybı ile yüksek sıkıştırma (DWT/DCT)
- Lossless: Sıfır kaybı (tıbbi imgeler, parmak izi vb.)
"""

from enum import Enum
from typing import Optional

class CompressionMode(Enum):
    """Sıkıştırma modları"""
    LOSSY_DWT = "lossy_dwt"       # Wavelet tabanlı lossy
    LOSSY_DCT = "lossy_dct"       # DCT tabanlı lossy
    LOSSLESS = "lossless"         # Sıfır kayıp (tıbbi/biometrik)


class ModeConfig:
    """Mod yapılandırması"""
    
    def __init__(self, mode: CompressionMode):
        """
        Başlat
        
        Args:
            mode: Sıkıştırma modu
        """
        self.mode = mode
        self.quality = None
        self.dwt_level = None
        self.wavelet = None
        self.quantization_method = None
    
    def set_lossy_params(
        self,
        quality: int = 85,
        dwt_level: int = 2,
        wavelet: str = "db4",
        quantization_method: str = "scalar"
    ):
        """Lossy parametreleri ayarla"""
        if self.mode == CompressionMode.LOSSLESS:
            raise ValueError("LOSSLESS modu parametreleri desteklemez")
        
        self.quality = quality
        self.dwt_level = dwt_level
        self.wavelet = wavelet
        self.quantization_method = quantization_method
    
    def set_lossless_params(self):
        """Lossless parametreleri ayarla (yapılandırma gerekmez)"""
        if self.mode != CompressionMode.LOSSLESS:
            raise ValueError("Bu modda lossless parametreleri geçerli değil")
    
    def is_lossless(self) -> bool:
        """Lossless mi?"""
        return self.mode == CompressionMode.LOSSLESS
    
    def is_lossy(self) -> bool:
        """Lossy mi?"""
        return self.mode in (CompressionMode.LOSSY_DWT, CompressionMode.LOSSY_DCT)
    
    def get_description(self) -> str:
        """Açıklama al"""
        if self.mode == CompressionMode.LOSSLESS:
            return "Lossless (sıfır kayıp)"
        elif self.mode == CompressionMode.LOSSY_DWT:
            return f"Lossy DWT (kalite={self.quality}, {self.wavelet}, level={self.dwt_level})"
        elif self.mode == CompressionMode.LOSSY_DCT:
            return f"Lossy DCT (kalite={self.quality})"
        return "Bilinmiyor"
    
    def to_dict(self):
        """Dict'e dönüştür"""
        return {
            'mode': self.mode.value,
            'quality': self.quality,
            'dwt_level': self.dwt_level,
            'wavelet': self.wavelet,
            'quantization_method': self.quantization_method
        }


class ModeSelector:
    """Modları seç ve doğrula"""
    
    @staticmethod
    def get_mode_by_use_case(use_case: str) -> CompressionMode:
        """
        Kullanım durumuna göre modu seç
        
        Args:
            use_case: 'medical', 'fingerprint', 'natural', vb.
        
        Returns:
            Seçilen mod
        """
        use_case = use_case.lower()
        
        if use_case in ('medical', 'fingerprint', 'biometric', 'document', 'scan'):
            return CompressionMode.LOSSLESS
        elif use_case in ('natural', 'photograph', 'photo', 'natural_image'):
            return CompressionMode.LOSSY_DWT
        elif use_case in ('synthetic', 'graphic', 'artwork', 'render'):
            return CompressionMode.LOSSY_DCT
        else:
            return CompressionMode.LOSSY_DWT  # Default
    
    @staticmethod
    def recommend_for_use_case(use_case: str) -> tuple:
        """
        Kullanım durumu için önerisi al
        
        Args:
            use_case: Kullanım durumu
        
        Returns:
            (mode, quality, description)
        """
        use_case = use_case.lower()
        
        recommendations = {
            'medical': (CompressionMode.LOSSLESS, None, "Tıbbi görüntüler için sıfır kayıp gerekli"),
            'fingerprint': (CompressionMode.LOSSLESS, None, "Parmak izi tanıması için sıfır kayıp gerekli"),
            'natural': (CompressionMode.LOSSY_DWT, 85, "Doğal görüntüler için DWT yüksek kalitede iyi çalışır"),
            'synthetic': (CompressionMode.LOSSY_DCT, 85, "Sentetik görüntüler için DCT uygundur"),
            'photograph': (CompressionMode.LOSSY_DWT, 80, "Fotoğraflar için DWT iyi sıkıştırma sağlar"),
            'document': (CompressionMode.LOSSLESS, None, "Belgeler için sıfır kayıp gerekli"),
        }
        
        if use_case in recommendations:
            return recommendations[use_case]
        else:
            return (CompressionMode.LOSSY_DWT, 85, "Standart ayarlar (DWT, kalite=85)")
