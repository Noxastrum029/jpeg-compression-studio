"""
stage6_metrics.py — Stage 6: Performance Evaluation & Metrics

Sıkıştırma sisteminin performansını değerlendirir:
- SSIM (Structural Similarity Index)
- MSE/PSNR
- Bits per pixel (Bpp)
- Compression ratio
- Yürütme zamanı
- Karşılaştırmalı analiz

Tüm pipeline'ın kalitesini ve verimliliğini ölçer.
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

logger = logging.getLogger(__name__)


@dataclass
class CompressionMetrics:
    """Sıkıştırma performans metrikleri"""
    # Boyutlar
    original_size_bytes: int
    compressed_size_bytes: int
    
    # Kalite metrikleri
    mse: float
    psnr: float
    ssim_value: float = 0.0  # 0-1, 1 = perfect match
    max_error: float = 0.0
    mean_error: float = 0.0
    
    # Verimlilik metrikleri
    compression_ratio: float = 0.0  # original/compressed
    bpp: float = 0.0  # bits per pixel
    size_reduction_percent: float = 0.0
    
    # Zaman metrikleri
    compression_time: float = 0.0  # seconds
    decompression_time: float = 0.0  # seconds
    total_time: float = 0.0
    
    # Görüntü bilgisi
    image_width: int = 0
    image_height: int = 0
    image_pixels: int = 0
    
    def __post_init__(self):
        """Türetilmiş metrikleri hesapla"""
        if self.original_size_bytes > 0:
            self.compression_ratio = self.original_size_bytes / max(1, self.compressed_size_bytes)
            self.size_reduction_percent = (1 - self.compressed_size_bytes / self.original_size_bytes) * 100
        
        if self.image_pixels > 0:
            self.bpp = (self.compressed_size_bytes * 8) / self.image_pixels
        
        self.total_time = self.compression_time + self.decompression_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Sözlüğe dönüştür"""
        return {
            'sizes': {
                'original_bytes': self.original_size_bytes,
                'compressed_bytes': self.compressed_size_bytes,
                'compression_ratio': round(self.compression_ratio, 2),
                'size_reduction_percent': round(self.size_reduction_percent, 1)
            },
            'quality': {
                'mse': round(self.mse, 6),
                'psnr': round(self.psnr, 2),
                'ssim': round(self.ssim_value, 4),
                'max_error': round(self.max_error, 2),
                'mean_error': round(self.mean_error, 2)
            },
            'efficiency': {
                'bpp': round(self.bpp, 2),
                'compression_time_ms': round(self.compression_time * 1000, 2),
                'decompression_time_ms': round(self.decompression_time * 1000, 2),
                'total_time_ms': round(self.total_time * 1000, 2)
            },
            'image': {
                'width': self.image_width,
                'height': self.image_height,
                'pixels': self.image_pixels
            }
        }
    
    def to_json_string(self) -> str:
        """JSON string'e dönüştür"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class MetricsCalculator:
    """Sıkıştırma metrikleri hesapla"""
    
    def __init__(self):
        logger.info("MetricsCalculator initialized")
        self.has_skimage = HAS_SKIMAGE
        if not HAS_SKIMAGE:
            logger.warning("scikit-image not available, SSIM will be approximated")
    
    def calculate_all_metrics(
        self,
        original_image: np.ndarray,
        reconstructed_image: np.ndarray,
        original_file_size: int,
        compressed_file_size: int,
        compression_time: float = 0.0,
        decompression_time: float = 0.0
    ) -> CompressionMetrics:
        """
        Tüm metrikleri hesapla
        
        Args:
            original_image: Orijinal görüntü (numpy array)
            reconstructed_image: Reconstruct edilmiş görüntü
            original_file_size: Orijinal dosya boyutu (bytes)
            compressed_file_size: Sıkıştırılmış dosya boyutu (bytes)
            compression_time: Sıkıştırma zamanı (seconds)
            decompression_time: Sıkıştırma açma zamanı (seconds)
            
        Returns:
            CompressionMetrics nesnesi
        """
        try:
            # Güvenli tür dönüşümü
            orig = original_image.astype(np.float32)
            recon = reconstructed_image.astype(np.float32)
            
            # Kalite metrikleri
            mse = np.mean((orig - recon) ** 2)
            psnr = self._calculate_psnr(mse)
            ssim_val = self._calculate_ssim(orig, recon)
            max_err = np.max(np.abs(orig - recon))
            mean_err = np.mean(np.abs(orig - recon))
            
            # Görüntü bilgisi
            img_height, img_width = original_image.shape[:2]
            img_pixels = img_height * img_width
            
            metrics = CompressionMetrics(
                original_size_bytes=original_file_size,
                compressed_size_bytes=compressed_file_size,
                mse=mse,
                psnr=psnr,
                ssim_value=ssim_val,
                max_error=max_err,
                mean_error=mean_err,
                compression_time=compression_time,
                decompression_time=decompression_time,
                image_width=img_width,
                image_height=img_height,
                image_pixels=img_pixels
            )
            
            logger.info(
                f"✓ Metrikler hesaplandı: "
                f"MSE={mse:.6f}, PSNR={psnr:.2f}dB, "
                f"Ratio={metrics.compression_ratio:.2f}x"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrik hesaplaması hatası: {e}")
            raise
    
    @staticmethod
    def _calculate_psnr(mse: float) -> float:
        """PSNR hesapla"""
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def _calculate_ssim(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> float:
        """SSIM (Structural Similarity Index) hesapla"""
        try:
            if self.has_skimage:
                # data_range'i hesapla, sıfırı engelle
                data_range = original.max() - original.min()
                if data_range <= 0:
                    # Görüntü sabit renk ise
                    # İdeal SSIM = 1 (tamamen aynı), farklı = 0
                    diff = np.mean((original - reconstructed) ** 2)
                    return 1.0 if diff < 1e-6 else 0.0
                
                ssim_value = ssim(
                    original,
                    reconstructed,
                    data_range=data_range
                )
                
                # NaN kontrolü
                if np.isnan(ssim_value):
                    logger.warning("SSIM hesaplaması NaN döndürdü, 0 kullanılıyor")
                    return 0.0
                
                return np.clip(float(ssim_value), 0, 1)
            else:
                # Basitleştirilmiş yaklaşım: MSE'ye dayanan metrik
                mse = np.mean((original - reconstructed) ** 2)
                # 0-1 range'e normalize (benzer SSIM davranışı)
                ssim_approx = np.exp(-mse / 1000)  # Gaussian-like decay
                return np.clip(ssim_approx, 0, 1)
                
        except Exception as e:
            logger.warning(f"SSIM hesaplaması başarısız, 0 döndürülüyor: {e}")
            return 0.0


class PerformanceBenchmark:
    """Performans karşılaştırılması ve benchmarking"""
    
    def __init__(self):
        self.results: List[CompressionMetrics] = []
        logger.info("PerformanceBenchmark initialized")
    
    def add_result(self, metrics: CompressionMetrics) -> None:
        """Sonuç ekle"""
        self.results.append(metrics)
        logger.info(f"✓ Sonuç eklendi (toplam: {len(self.results)})")
    
    def compare_quality(self) -> Dict[str, Any]:
        """Kalite karşılaştırması"""
        if len(self.results) == 0:
            return {}
        
        metrics_list = [m.to_dict() for m in self.results]
        
        comparison = {
            'count': len(self.results),
            'best_psnr': max(m['quality']['psnr'] for m in metrics_list),
            'worst_psnr': min(m['quality']['psnr'] for m in metrics_list),
            'avg_psnr': np.mean([m['quality']['psnr'] for m in metrics_list]),
            'best_ssim': max(m['quality']['ssim'] for m in metrics_list),
            'worst_ssim': min(m['quality']['ssim'] for m in metrics_list),
            'avg_ssim': np.mean([m['quality']['ssim'] for m in metrics_list])
        }
        
        return comparison
    
    def compare_compression(self) -> Dict[str, Any]:
        """Sıkıştırma oranları karşılaştırması"""
        if len(self.results) == 0:
            return {}
        
        metrics_list = [m.to_dict() for m in self.results]
        ratios = [m['sizes']['compression_ratio'] for m in metrics_list]
        bpps = [m['efficiency']['bpp'] for m in metrics_list]
        
        comparison = {
            'best_ratio': max(ratios),
            'worst_ratio': min(ratios),
            'avg_ratio': np.mean(ratios),
            'best_bpp': min(bpps),
            'worst_bpp': max(bpps),
            'avg_bpp': np.mean(bpps)
        }
        
        return comparison
    
    def compare_speed(self) -> Dict[str, Any]:
        """Hız karşılaştırması"""
        if len(self.results) == 0:
            return {}
        
        comp_times = [m.compression_time for m in self.results]
        decomp_times = [m.decompression_time for m in self.results]
        total_times = [m.total_time for m in self.results]
        
        comparison = {
            'fastest_compression_ms': min(comp_times) * 1000,
            'slowest_compression_ms': max(comp_times) * 1000,
            'avg_compression_ms': np.mean(comp_times) * 1000,
            'fastest_decompression_ms': min(decomp_times) * 1000,
            'slowest_decompression_ms': max(decomp_times) * 1000,
            'avg_decompression_ms': np.mean(decomp_times) * 1000,
            'fastest_total_ms': min(total_times) * 1000,
            'slowest_total_ms': max(total_times) * 1000,
            'avg_total_ms': np.mean(total_times) * 1000
        }
        
        return comparison
    
    def generate_report(self) -> str:
        """Kapsamlı rapor oluştur"""
        if len(self.results) == 0:
            return "No results to report"
        
        quality_comp = self.compare_quality()
        compression_comp = self.compare_compression()
        speed_comp = self.compare_speed()
        
        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    COMPRESSION PERFORMANCE REPORT                         ║
║                          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                            ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 QUALITY METRICS
─────────────────────────────────────────────────────────────────────────────
  PSNR (dB):
    • Best:     {quality_comp.get('best_psnr', 0):.2f} dB
    • Worst:    {quality_comp.get('worst_psnr', 0):.2f} dB
    • Average:  {quality_comp.get('avg_psnr', 0):.2f} dB
  
  SSIM:
    • Best:     {quality_comp.get('best_ssim', 0):.4f}
    • Worst:    {quality_comp.get('worst_ssim', 0):.4f}
    • Average:  {quality_comp.get('avg_ssim', 0):.4f}

📦 COMPRESSION EFFICIENCY
─────────────────────────────────────────────────────────────────────────────
  Compression Ratio (original/compressed):
    • Best:     {compression_comp.get('best_ratio', 0):.2f}x
    • Worst:    {compression_comp.get('worst_ratio', 0):.2f}x
    • Average:  {compression_comp.get('avg_ratio', 0):.2f}x
  
  Bits Per Pixel (Bpp):
    • Best:     {compression_comp.get('best_bpp', 0):.2f} bpp
    • Worst:    {compression_comp.get('worst_bpp', 0):.2f} bpp
    • Average:  {compression_comp.get('avg_bpp', 0):.2f} bpp

⚡ PERFORMANCE SPEED
─────────────────────────────────────────────────────────────────────────────
  Compression (ms):
    • Fastest:  {speed_comp.get('fastest_compression_ms', 0):.2f} ms
    • Slowest:  {speed_comp.get('slowest_compression_ms', 0):.2f} ms
    • Average:  {speed_comp.get('avg_compression_ms', 0):.2f} ms
  
  Decompression (ms):
    • Fastest:  {speed_comp.get('fastest_decompression_ms', 0):.2f} ms
    • Slowest:  {speed_comp.get('slowest_decompression_ms', 0):.2f} ms
    • Average:  {speed_comp.get('avg_decompression_ms', 0):.2f} ms
  
  Total Time (ms):
    • Fastest:  {speed_comp.get('fastest_total_ms', 0):.2f} ms
    • Slowest:  {speed_comp.get('slowest_total_ms', 0):.2f} ms
    • Average:  {speed_comp.get('avg_total_ms', 0):.2f} ms

📈 SUMMARY
─────────────────────────────────────────────────────────────────────────────
  Total Tests: {len(self.results)}
  Timestamp: {datetime.now().isoformat()}
"""
        
        return report
    
    def save_report(self, output_path: str) -> None:
        """Raporu dosyaya kaydet"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            report = self.generate_report()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"✓ Rapor kaydedildi: {output_path}")
            
        except Exception as e:
            logger.error(f"Rapor kayıt hatası: {e}")


class QualityAssessment:
    """Kalite sınıflandırması ve öneriler"""
    
    @staticmethod
    def assess_quality(metrics: CompressionMetrics) -> Dict[str, Any]:
        """Kalite seviyesini değerlendir"""
        psnr = metrics.psnr
        ssim_val = metrics.ssim_value
        
        # PSNR sınıflandırması
        if psnr >= 50:
            psnr_grade = "Excellent"
            psnr_level = "Lossless-like quality"
        elif psnr >= 40:
            psnr_grade = "Good"
            psnr_level = "Visual quality acceptable"
        elif psnr >= 30:
            psnr_grade = "Acceptable"
            psnr_level = "Noticeable quality loss"
        elif psnr >= 20:
            psnr_grade = "Poor"
            psnr_level = "Significant quality loss"
        else:
            psnr_grade = "Very Poor"
            psnr_level = "Severe quality loss"
        
        # SSIM sınıflandırması
        if ssim_val >= 0.95:
            ssim_grade = "Excellent"
            ssim_level = "Visually indistinguishable"
        elif ssim_val >= 0.90:
            ssim_grade = "Good"
            ssim_level = "Minor differences"
        elif ssim_val >= 0.80:
            ssim_grade = "Acceptable"
            ssim_level = "Noticeable differences"
        elif ssim_val >= 0.60:
            ssim_grade = "Poor"
            ssim_level = "Significant differences"
        else:
            ssim_grade = "Very Poor"
            ssim_level = "Severe differences"
        
        # Kompresyon verimli mi?
        ratio = metrics.compression_ratio
        if ratio >= 5:
            efficiency = "Excellent"
            efficiency_note = f"{ratio:.1f}x smaller - highly effective"
        elif ratio >= 3:
            efficiency = "Good"
            efficiency_note = f"{ratio:.1f}x smaller - effective"
        elif ratio >= 1.5:
            efficiency = "Acceptable"
            efficiency_note = f"{ratio:.1f}x smaller - moderate"
        else:
            efficiency = "Poor"
            efficiency_note = f"{ratio:.1f}x smaller - minimal"
        
        return {
            'psnr_grade': psnr_grade,
            'psnr_level': psnr_level,
            'psnr_value': psnr,
            'ssim_grade': ssim_grade,
            'ssim_level': ssim_level,
            'ssim_value': ssim_val,
            'efficiency_grade': efficiency,
            'efficiency_note': efficiency_note,
            'overall_recommendation': f"Quality: {psnr_grade}, Efficiency: {efficiency}"
        }


if __name__ == "__main__":
    print("Stage 6: Performance Metrics & Evaluation")
    print("- Quality metrics (MSE, PSNR, SSIM)")
    print("- Compression statistics (ratio, bpp)")
    print("- Performance benchmarking")
    print("- Comparative analysis")
