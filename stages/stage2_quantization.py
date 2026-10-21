"""
stage2_quantization.py — Aşama 2: DWT Katsayılarının Kuantalanması

Kuantalama, DWT katsayılarının hassasiyetini azaltarak veri boyutunu küçültür.

Yöntemler:
1. Uniform Quantization: Tüm bantlara aynı step_size
2. Scalar Quantization: Banda göre farklı step_size (JPEG2000 standardı)
3. Adaptive Quantization: Görüntü özelliklerine göre dinamik adım

Kullanım:
    >>> from stages.stage2_quantization import QuantizationModule
    >>> quantizer = QuantizationModule(method="uniform")
    >>> step = quantizer.quality_to_step(quality=50)
    >>> quantized_coeffs = quantizer.uniform_quantize(coeffs, step)
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np

from core.logging_config import setup_logger
from core.utils import quality_to_step_size, step_size_to_quality, validate_quality_factor
from core.config import (
    SCALAR_QUANTIZATION_FACTORS,
    MAX_STEP_SIZE,
    MIN_STEP_SIZE,
    DEFAULT_QUALITY,
)

logger = setup_logger("stage2", "stage2.log")


class QuantizationModule:
    """
    DWT katsayılarını kuantalayan sınıf.
    
    Attributes:
        method (str): Kuantalama yöntemi ("uniform", "scalar", "adaptive")
        step_size (float): Uniform kuantalama adımı
        step_sizes (Dict): Banda göre adımlar (scalar kuantalama)
    """
    
    def __init__(self, method: str = "uniform"):
        """
        Initialize QuantizationModule.
        
        Args:
            method: Kuantalama yöntemi
                - "uniform": Tüm bantlara aynı step
                - "scalar": Banda göre farklı step (JPEG2000)
                - "adaptive": Dinamik step (ileri versiyon)
        """
        if method not in ["uniform", "scalar", "adaptive"]:
            raise ValueError(f"Yöntem desteklenmiyor: {method}")
        
        self.method = method
        self.step_size: Optional[float] = None
        self.step_sizes: Dict[str, float] = {}
        self.quantized_coeffs: Optional[List] = None
        
        logger.info(f"QuantizationModule initialized: method={method}")
    
    # ========================================================================
    # 🔀 KUANTİZASYON YÖNTEMLERİ
    # ========================================================================
    
    def uniform_quantize(self, coeffs: List, step_size: float) -> List:
        """
        Uniform kuantalama: Tüm bantlara aynı step_size uygulanır.
        
        Formül: q = round(c / step) * step
        
        Args:
            coeffs: DWT katsayıları (pywt.wavedec2 formatı)
            step_size: Kuantalama adımı
            
        Returns:
            Kuantalanmış katsayılar
            
        Raises:
            ValueError: step_size <= 0
        """
        if step_size <= 0:
            raise ValueError(f"step_size > 0 olmalı, {step_size} alındı")
        
        logger.info(f"Uniform kuantalama uygulanıyor: step_size={step_size:.4f}")
        
        try:
            out = []
            
            # LL bandı (yaklaşım)
            out.append(np.round(coeffs[0] / step_size) * step_size)
            
            # Detail bantları (HL, LH, HH)
            for (cH, cV, cD) in coeffs[1:]:
                out.append((
                    np.round(cH / step_size) * step_size,
                    np.round(cV / step_size) * step_size,
                    np.round(cD / step_size) * step_size,
                ))
            
            self.step_size = step_size
            self.quantized_coeffs = out
            
            logger.info(f"✓ Uniform kuantalama tamamlandı")
            return out
        
        except Exception as e:
            logger.error(f"Uniform kuantalama hatası: {e}")
            raise
    
    def scalar_quantize(self, coeffs: List,
                       ll_factor: float = 1.0,
                       hl_factor: float = 2.0,
                       lh_factor: float = 2.0,
                       hh_factor: float = 4.0,
                       base_step: float = 1.0) -> List:
        """
        Scalar kuantalama: Banda göre farklı step_size'lar.
        
        JPEG2000 standardına uygun:
        - LL (yaklaşım): Küçük step (önemli)
        - HL, LH (detay): Orta step
        - HH (diyagonal): Büyük step (az önemli)
        
        Args:
            coeffs: DWT katsayıları
            ll_factor: LL çarpanı
            hl_factor: HL çarpanı
            lh_factor: LH çarpanı
            hh_factor: HH çarpanı
            base_step: Temel adım
            
        Returns:
            Kuantalanmış katsayılar
        """
        logger.info(f"Scalar kuantalama uygulanıyor: "
                   f"LL={ll_factor}, HL={hl_factor}, LH={lh_factor}, HH={hh_factor}")
        
        try:
            out = []
            
            # LL bandı
            ll_step = base_step * ll_factor
            out.append(np.round(coeffs[0] / ll_step) * ll_step)
            
            # Detail bantları
            for (cH, cV, cD) in coeffs[1:]:
                hl_step = base_step * hl_factor
                lh_step = base_step * lh_factor
                hh_step = base_step * hh_factor
                
                out.append((
                    np.round(cH / hl_step) * hl_step,
                    np.round(cV / lh_step) * lh_step,
                    np.round(cD / hh_step) * hh_step,
                ))
            
            self.step_sizes = {
                'LL': ll_step,
                'HL': hl_step,
                'LH': lh_step,
                'HH': hh_step,
            }
            self.quantized_coeffs = out
            
            logger.info(f"✓ Scalar kuantalama tamamlandı")
            return out
        
        except Exception as e:
            logger.error(f"Scalar kuantalama hatası: {e}")
            raise
    
    def quantize_by_quality(self, coeffs: List, quality: int,
                           method: Optional[str] = None) -> List:
        """
        Kalite faktöründen kuantalama yap.
        
        Args:
            coeffs: DWT katsayıları
            quality: Kalite faktörü (0-100)
            method: Yöntem (None = self.method)
            
        Returns:
            Kuantalanmış katsayılar
        """
        validate_quality_factor(quality)
        
        method = method or self.method
        step_size = quality_to_step_size(quality)
        
        logger.info(f"Quality={quality} → step_size={step_size:.4f}")
        
        if method == "uniform":
            return self.uniform_quantize(coeffs, step_size)
        elif method == "scalar":
            # Kaliteyi facto
            factor = (100 - quality) / 50  # 0→2, 50→1, 100→0
            return self.scalar_quantize(coeffs, base_step=step_size)
        else:
            raise ValueError(f"Desteklenmeyen yöntem: {method}")
    
    # ========================================================================
    # 📊 SIKIŞTIRMA ANALİZİ
    # ========================================================================
    
    @staticmethod
    def estimate_compression(original_coeffs: List,
                             quantized_coeffs: List) -> Dict[str, float]:
        """
        Kuantalamadan sonra sıkıştırma tahminini yap.
        
        Not: Gerçek Huffman kodlaması olmadan tahmin edilir.
        
        Args:
            original_coeffs: Orijinal DWT katsayıları
            quantized_coeffs: Kuantalanmış katsayılar
            
        Returns:
            Sıkıştırma metrikleri
        """
        # Orijinal boyut (float32 = 4 bytes per element)
        original_size = original_coeffs[0].size * 4
        for (cH, cV, cD) in original_coeffs[1:]:
            original_size += cH.size * 4 + cV.size * 4 + cD.size * 4
        
        # Kuantalanmış boyut tahmin (int16 = 2 bytes per element)
        quantized_size = quantized_coeffs[0].size * 2
        for (cH, cV, cD) in quantized_coeffs[1:]:
            quantized_size += cH.size * 2 + cV.size * 2 + cD.size * 2
        
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
        
        return {
            "original_bytes": original_size,
            "quantized_bytes": quantized_size,
            "compression_ratio": compression_ratio,
            "reduction_percent": (1 - quantized_size / original_size) * 100,
        }
    
    @staticmethod
    def analyze_sparsity(quantized_coeffs: List) -> Dict[str, float]:
        """
        Kuantalamadan sonra sıfır sayısını analiz et.
        
        Arttıkça Huffman kodlaması daha etkili olur.
        
        Args:
            quantized_coeffs: Kuantalanmış katsayılar
            
        Returns:
            Sparsity metrikleri
        """
        total_elements = quantized_coeffs[0].size
        total_zeros = np.count_nonzero(quantized_coeffs[0] == 0)
        
        for (cH, cV, cD) in quantized_coeffs[1:]:
            total_elements += cH.size + cV.size + cD.size
            total_zeros += np.count_nonzero(cH == 0) + \
                          np.count_nonzero(cV == 0) + \
                          np.count_nonzero(cD == 0)
        
        sparsity_ratio = total_zeros / total_elements if total_elements > 0 else 0.0
        
        return {
            "total_elements": total_elements,
            "zero_elements": total_zeros,
            "sparsity_ratio": sparsity_ratio,
            "sparsity_percent": sparsity_ratio * 100,
        }
    
    # ========================================================================
    # 🔍 KONTROLLERİ VE BILGI
    # ========================================================================
    
    def get_coeffs_statistics(self, coeffs: List) -> Dict[str, Dict[str, float]]:
        """
        Katsayıların istatistiklerini hesapla.
        
        Args:
            coeffs: Katsayılar
            
        Returns:
            Her bant için min, max, mean, std
        """
        stats = {}
        
        # LL bandı
        ll = coeffs[0].flatten()
        stats['LL'] = {
            'min': float(np.min(ll)),
            'max': float(np.max(ll)),
            'mean': float(np.mean(ll)),
            'std': float(np.std(ll)),
        }
        
        # Detail bantları
        for i, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
            for name, band in [('HL', cH), ('LH', cV), ('HH', cD)]:
                band_flat = band.flatten()
                key = f"Level{i}_{name}"
                stats[key] = {
                    'min': float(np.min(band_flat)),
                    'max': float(np.max(band_flat)),
                    'mean': float(np.mean(band_flat)),
                    'std': float(np.std(band_flat)),
                }
        
        return stats
    
    def print_statistics(self, coeffs: List, title: str = "Katsayı İstatistikleri") -> None:
        """İstatistikleri güzel formatta yazdır."""
        stats = self.get_coeffs_statistics(coeffs)
        
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        
        for band_name, band_stats in stats.items():
            print(f"\n  {band_name:>12}:")
            for stat_name, value in band_stats.items():
                print(f"    {stat_name:>8} = {value:>12.4f}")
        
        print(f"{'='*60}\n")


# ============================================================================
# ✅ TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Test: Basit katsayılar oluştur
    print("Quantization Module Test")
    print("=" * 60)
    
    # Yapay test katsayıları
    test_coeffs = [
        np.random.randn(64, 64).astype(np.float32) * 100,  # LL
        (np.random.randn(64, 64).astype(np.float32) * 50,   # HL
         np.random.randn(64, 64).astype(np.float32) * 50,   # LH
         np.random.randn(64, 64).astype(np.float32) * 50),  # HH
    ]
    
    quantizer = QuantizationModule(method="uniform")
    
    # Test 1: Uniform kuantalama
    print("\n[TEST 1] Uniform Kuantalama")
    quantized = quantizer.uniform_quantize(test_coeffs, step_size=2.0)
    print("✓ Uniform kuantalama başarılı")
    
    # Test 2: Sıkıştırma analizi
    print("\n[TEST 2] Sıkıştırma Analizi")
    compression = quantizer.estimate_compression(test_coeffs, quantized)
    print(f"  Orijinal: {compression['original_bytes']} bytes")
    print(f"  Kuantalı: {compression['quantized_bytes']} bytes")
    print(f"  Oran: {compression['compression_ratio']:.2f}x")
    
    # Test 3: Sparsity
    print("\n[TEST 3] Sparsity Analizi")
    sparsity = quantizer.analyze_sparsity(quantized)
    print(f"  Sıfır oranı: {sparsity['sparsity_percent']:.2f}%")
    
    # Test 4: Quality → Step
    print("\n[TEST 4] Quality → Step Dönüşümü")
    for q in [0, 25, 50, 75, 100]:
        step = quality_to_step_size(q)
        print(f"  Quality {q:3d} → Step {step:6.2f}")
    
    print("\n✅ Tüm testler geçti!")
