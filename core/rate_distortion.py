"""
rate_distortion.py — Rate-Distortion Analysis & Optimization

Rate-Distortion (R-D) analizi:
- Farklı kalite seviyelerinde bitrate vs PSNR/SSIM grafiği
- R-D curve oluşturma ve kaydetme
- Optimal quality point bulma
- Lambda-based R-D trade-off kontrolü

Hocanın "Enable rate-distortion trade-off control" gereksinimini karşılar.
"""

import logging
import time
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from core.utils import ssim_fn_any

logger = logging.getLogger(__name__)


@dataclass
class RDPoint:
    """Tek bir Rate-Distortion noktası."""
    quality: int
    bitrate_bpp: float       # bits per pixel
    psnr_db: float           # dB
    mse: float
    ssim: float
    compression_ratio: float
    compressed_size: int     # bytes
    original_size: int       # bytes
    method: str              # DWT or DCT


class RateDistortionAnalyzer:
    """
    Rate-Distortion analizi ve trade-off kontrolü.

    Farklı kalite seviyelerinde sıkıştırma yaparak
    R-D curve oluşturur ve optimal çalışma noktasını bulur.
    """

    def __init__(self, method: str = "DWT"):
        """
        Başlat.

        Args:
            method: Transform yöntemi ('DWT' veya 'DCT')
        """
        self.method = method
        self.rd_points: List[RDPoint] = []
        self.qualities = [5, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 99]
        logger.info(f"RateDistortionAnalyzer initialized: method={method}")

    def analyze(self, image: np.ndarray) -> List[RDPoint]:
        """
        Farklı kalite seviyelerinde R-D analizi yap.

        Args:
            image: Grayscale görüntü (2D numpy array, float32)

        Returns:
            R-D noktaları listesi
        """
        self.rd_points = []
        original_size = image.nbytes

        logger.info(f"R-D analysis started: method={self.method}, "
                    f"qualities={self.qualities}")

        for quality in self.qualities:
            try:
                point = self._compress_at_quality(image, quality, original_size)
                self.rd_points.append(point)
                logger.info(
                    f"  Q{quality:3d}: bpp={point.bitrate_bpp:.3f}, "
                    f"PSNR={point.psnr_db:.2f}dB, ratio={point.compression_ratio:.2f}x"
                )
            except Exception as e:
                logger.warning(f"  Q{quality}: failed - {e}")

        logger.info(f"✓ R-D analysis complete: {len(self.rd_points)} points")
        return self.rd_points

    def _compress_at_quality(
        self, image: np.ndarray, quality: int, original_size: int
    ) -> RDPoint:
        """Belirli bir kalitede sıkıştır ve metrikleri hesapla."""

        if self.method == "DCT":
            return self._compress_dct(image, quality, original_size)
        elif self.method == "Lossless":
            return self._compress_lossless(image, quality, original_size)
        else:
            return self._compress_dwt(image, quality, original_size)

    def _compress_lossless(
        self, image: np.ndarray, quality: int, original_size: int
    ) -> RDPoint:
        """Kayıpsız (Lossless) sıkıştırma metrik hesaplama."""
        import zlib
        
        try:
            from stages.stage_lossless import LosslessCompressionPipeline
            pipeline = LosslessCompressionPipeline(compression_level=quality)
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
            compressed, _ = pipeline.compress_image_data(img_uint8)
            reconstructed = pipeline.decompress_image_data(compressed)
            compressed_size = len(compressed)
        except Exception as e:
            # Fallback: direct zlib
            logger.warning(f"Lossless pipeline failed, using zlib fallback: {e}")
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
            compressed = zlib.compress(img_uint8.tobytes(), level=quality)
            reconstructed = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(image.shape)
            compressed_size = len(compressed)

        return self._compute_rd_point(
            image, reconstructed.astype(np.float64), quality, original_size, "Lossless", compressed_size
        )

    def _compress_dwt(
        self, image: np.ndarray, quality: int, original_size: int
    ) -> RDPoint:
        """DWT sıkıştırma ve metrik hesaplama."""
        import pywt
        
        try:
            from stages.stage1_preprocessing import DWTTransformer
            from stages.stage2_quantization import QuantizationModule
            
            dwt = DWTTransformer("db4", 2)
            coeffs = dwt.apply_dwt2d(image)
            quantizer = QuantizationModule()
            quantized = quantizer.quantize_by_quality(coeffs, quality, method="scalar")
            reconstructed = pywt.waverec2(quantized, "db4", mode="periodization")
            reconstructed = reconstructed[:image.shape[0], :image.shape[1]]
        except Exception as e:
            logger.warning(f"DWT pipeline failed, using fallback: {e}")
            # Fallback: basit DWT
            try:
                coeffs = pywt.wavedec2(image, 'db4', level=2, mode='periodization')
                # Kaliteye göre katsayıları temizle (seyrekliği artır)
                def threshold_coeffs(c, quality):
                    if isinstance(c, tuple):
                        threshold = np.percentile(np.abs(c), 100 - min(quality, 95))
                        c = np.where(np.abs(c) < threshold, 0, c)
                    return c
                
                quantized = [coeffs[0]]  # LL
                for coeff_tuple in coeffs[1:]:
                    cH, cV, cD = coeff_tuple
                    quantized.append((
                        threshold_coeffs(cH, quality),
                        threshold_coeffs(cV, quality),
                        threshold_coeffs(cD, quality)
                    ))
                
                reconstructed = pywt.waverec2(quantized, 'db4', mode='periodization')
                reconstructed = reconstructed[:image.shape[0], :image.shape[1]]
            except Exception as e2:
                logger.error(f"DWT fallback also failed: {e2}")
                raise

        from huffman_codec import huffman_compressed_size
        _, huff_bytes, _ = huffman_compressed_size(quantized)
        actual_dwt_bytes = int(huff_bytes)

        return self._compute_rd_point(
            image, reconstructed, quality, original_size, "DWT", actual_dwt_bytes
        )

    def _compress_dct(
        self, image: np.ndarray, quality: int, original_size: int
    ) -> RDPoint:
        """DCT sıkıştırma ve metrik hesaplama."""
        try:
            from stages.stage_dct_transform import DCTCompressionPipeline
            pipeline = DCTCompressionPipeline(quality=quality)
            dct_coeffs = pipeline.transformer.apply_dct_2d(image)
            quantized = pipeline.quantize_dct(dct_coeffs)
            dequantized = pipeline.dequantize_dct(quantized)
            reconstructed = pipeline.reconstructor.apply_idct_2d(dequantized, image.shape)
        except Exception as e:
            logger.warning(f"DCT pipeline failed, using fallback: {e}")
            # Fallback: SciPy yoksa core.dct_utils (OpenCV blok DCT) kullanılır
            from core.dct_utils import dct2_ortho, idct2_ortho

            dct_coeffs = dct2_ortho(image)
            threshold = np.percentile(np.abs(dct_coeffs), 100 - min(quality, 99))
            dct_coeffs = np.asarray(dct_coeffs, dtype=np.float64)
            dct_coeffs[np.abs(dct_coeffs) < threshold] = 0
            reconstructed = idct2_ortho(dct_coeffs)

        image_u8 = np.clip(image, 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", image_u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        actual_dct_bytes = int(enc.nbytes) if ok else None

        return self._compute_rd_point(
            image, reconstructed, quality, original_size, "DCT", actual_dct_bytes
        )

    def _compute_rd_point(
        self, original: np.ndarray, reconstructed: np.ndarray,
        quality: int, original_size: int, method: str,
        actual_compressed_size: Optional[int] = None
    ) -> RDPoint:
        """PSNR, MSE, SSIM, bitrate hesapla."""
        orig = original.astype(np.float64)
        recon = reconstructed.astype(np.float64)

        min_h = min(orig.shape[0], recon.shape[0])
        min_w = min(orig.shape[1], recon.shape[1])
        orig = orig[:min_h, :min_w]
        recon = recon[:min_h, :min_w]

        mse = float(np.mean((orig - recon) ** 2))

        if mse == 0:
            psnr = float('inf')
        else:
            psnr = float(20 * np.log10(255.0 / np.sqrt(mse)))

        ssim_val = ssim_fn_any(orig, recon, data_range=255.0)

        total_pixels = orig.size

        if actual_compressed_size is not None:
            # Lossless sıkıştırması için gerçek boyut var
            estimated_compressed = actual_compressed_size
        else:
            # DCT/DWT için sıkıştırma boyutunu tahmin et
            # Kalite arttıkça sıkıştırma oranı düşer (daha fazla bilgi gerekir)
            if method == "DCT":
                # DCT: kalite 1-100, düşük kalite daha seyrek
                compression_ratio = 1.0 - (quality / 150.0)  # 1/3 - 2/3 oranı
            else:  # DWT
                # DWT: kalite 1-100, düşük kalite daha seyrek
                compression_ratio = 1.0 - (quality / 160.0)  # 1/3 - 2/3 oranı
            
            compression_ratio = np.clip(compression_ratio, 0.15, 0.85)
            estimated_compressed = int(original_size * (1.0 - compression_ratio))
            estimated_compressed = max(100, estimated_compressed)  # Min 100 bytes

        # Bitrate hesapla (bits per pixel)
        bpp = (estimated_compressed * 8) / total_pixels if total_pixels > 0 else 0
        ratio = original_size / estimated_compressed if estimated_compressed > 0 else 1

        return RDPoint(
            quality=quality,
            bitrate_bpp=bpp,
            psnr_db=psnr,
            mse=mse,
            ssim=ssim_val,
            compression_ratio=ratio,
            compressed_size=estimated_compressed,
            original_size=original_size,
            method=method,
        )

    # ================================================================
    # R-D CURVE OPERATIONS
    # ================================================================

    def find_optimal_quality(self, target_psnr: float = 35.0) -> Optional[int]:
        """
        Hedef PSNR için optimal kalite seviyesini bul.

        Args:
            target_psnr: Hedef PSNR (dB)

        Returns:
            Optimal quality (0-100) veya None
        """
        if not self.rd_points:
            return None

        # Target PSNR'ı karşılayan en düşük bitrate noktasını bul
        valid = [p for p in self.rd_points if p.psnr_db >= target_psnr]
        if not valid:
            # En yüksek PSNR'lı noktayı döndür
            return max(self.rd_points, key=lambda p: p.psnr_db).quality

        # En düşük bitrate
        return min(valid, key=lambda p: p.bitrate_bpp).quality

    def find_knee_point(self) -> Optional[int]:
        """
        R-D curve'deki diz noktasını bul (diminishing returns başladığı yer).

        Returns:
            Diz noktasındaki quality
        """
        if len(self.rd_points) < 3:
            return None

        # Sort by quality
        sorted_pts = sorted(self.rd_points, key=lambda p: p.quality)

        # PSNR gain per quality increment
        max_efficiency = 0
        knee_quality = sorted_pts[0].quality

        for i in range(1, len(sorted_pts)):
            dq = sorted_pts[i].quality - sorted_pts[i-1].quality
            if dq == 0:
                continue
            dp = sorted_pts[i].psnr_db - sorted_pts[i-1].psnr_db
            db = sorted_pts[i].bitrate_bpp - sorted_pts[i-1].bitrate_bpp

            # Efficiency: PSNR gain per unit bitrate increase
            if db > 0:
                efficiency = dp / db
                if efficiency > max_efficiency:
                    max_efficiency = efficiency
                    knee_quality = sorted_pts[i].quality

        return knee_quality

    def compute_lagrangian_cost(
        self, rd_point: RDPoint, lambda_val: float
    ) -> float:
        """
        Lagrangian R-D cost hesapla: J = D + λ·R

        Args:
            rd_point: R-D noktası
            lambda_val: Lagrange çarpanı

        Returns:
            J(R,D) cost
        """
        distortion = rd_point.mse
        rate = rd_point.bitrate_bpp
        return distortion + lambda_val * rate

    def find_optimal_lambda(self, target_quality: int = 85) -> float:
        """
        Hedef kalite için optimal lambda değerini bul.

        Args:
            target_quality: Hedef kalite seviyesi

        Returns:
            Optimal lambda
        """
        if len(self.rd_points) < 2:
            return 1.0

        sorted_pts = sorted(self.rd_points, key=lambda p: p.quality)

        # dD/dR slope hesapla
        for i in range(1, len(sorted_pts)):
            if sorted_pts[i].quality >= target_quality:
                dr = sorted_pts[i].bitrate_bpp - sorted_pts[i-1].bitrate_bpp
                dd = sorted_pts[i-1].mse - sorted_pts[i].mse  # MSE azalır
                if dr != 0:
                    return abs(dd / dr)
                break

        return 1.0

    # ================================================================
    # VISUALIZATION & REPORTING
    # ================================================================

    def plot_rd_curve(self, save_path: Optional[str] = None) -> str:
        """
        R-D curve grafiğini matplotlib ile oluştur ve kaydet.

        Args:
            save_path: Kayıt yolu (None ise sadece dosyaya kaydet)

        Returns:
            Kaydedilen dosya yolu
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
            return ""

        if not self.rd_points:
            return ""

        sorted_pts = sorted(self.rd_points, key=lambda p: p.bitrate_bpp)

        bpp_vals = [p.bitrate_bpp for p in sorted_pts]
        psnr_vals = [p.psnr_db for p in sorted_pts]
        quality_labels = [p.quality for p in sorted_pts]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Plot 1: Rate vs PSNR ---
        ax1.plot(bpp_vals, psnr_vals, 'b-o', linewidth=2, markersize=6, label=self.method)
        for i, q in enumerate(quality_labels):
            ax1.annotate(f'Q{q}', (bpp_vals[i], psnr_vals[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

        # Knee point
        knee = self.find_knee_point()
        if knee is not None:
            knee_pt = next((p for p in sorted_pts if p.quality == knee), None)
            if knee_pt:
                ax1.plot(knee_pt.bitrate_bpp, knee_pt.psnr_db, 'r*',
                        markersize=15, label=f'Optimal (Q{knee})')

        ax1.set_xlabel('Bitrate (bpp)', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title('Rate-Distortion Curve', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Plot 2: Quality vs PSNR ---
        q_sorted = sorted(self.rd_points, key=lambda p: p.quality)
        qualities = [p.quality for p in q_sorted]
        psnrs = [p.psnr_db for p in q_sorted]
        ratios = [p.compression_ratio for p in q_sorted]

        ax2.plot(qualities, psnrs, 'g-s', linewidth=2, markersize=6, label='PSNR (dB)')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(qualities, ratios, 'r--^', linewidth=1.5, markersize=5,
                     label='Compression Ratio', alpha=0.7)

        ax2.set_xlabel('Quality Factor', fontsize=12)
        ax2.set_ylabel('PSNR (dB)', fontsize=12, color='g')
        ax2_twin.set_ylabel('Compression Ratio', fontsize=12, color='r')
        ax2.set_title('Quality vs Performance', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = str(Path.cwd() / "output" / "reports" / "rd_curve.png")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ R-D curve saved: {save_path}")
        return save_path

    def generate_rd_report(self) -> str:
        """R-D analiz raporu oluştur (text format)."""
        if not self.rd_points:
            return "No R-D data available"

        knee = self.find_knee_point()
        opt_35 = self.find_optimal_quality(35.0)

        lines = [
            "=" * 70,
            "  RATE-DISTORTION ANALYSIS REPORT",
            f"  Method: {self.method}",
            f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            f"{'Quality':>8} {'Bitrate(bpp)':>13} {'PSNR(dB)':>10} "
            f"{'MSE':>10} {'SSIM':>8} {'Ratio':>8}",
            "-" * 70,
        ]

        for p in sorted(self.rd_points, key=lambda x: x.quality):
            psnr_str = f"{p.psnr_db:.2f}" if p.psnr_db != float('inf') else "∞"
            lines.append(
                f"{p.quality:>8d} {p.bitrate_bpp:>13.4f} {psnr_str:>10s} "
                f"{p.mse:>10.2f} {p.ssim:>8.4f} {p.compression_ratio:>7.2f}x"
            )

        lines.extend([
            "",
            "-" * 70,
            f"  Optimal quality for 35dB PSNR: Q{opt_35}",
            f"  Knee point (best efficiency):  Q{knee}",
            f"  Lambda at Q85:                 {self.find_optimal_lambda(85):.4f}",
            "=" * 70,
        ])

        return "\n".join(lines)

    def to_json(self) -> str:
        """R-D verisini JSON olarak döndür."""
        return json.dumps({
            "method": self.method,
            "date": datetime.now().isoformat(),
            "points": [asdict(p) for p in self.rd_points],
            "optimal_q35": self.find_optimal_quality(35.0),
            "knee_point": self.find_knee_point(),
        }, indent=2, default=str)


# ============================================================================
# OPTIMUM BIT ALLOCATION
# ============================================================================

class OptimumBitAllocator:
    """
    Alt bantlara optimum bit dağılımı.

    Variance-based bit allocation: Her alt bandın varyansına göre
    bit sayısını optimum şekilde dağıtır.

    Formül (JPEG2000 standardına uygun):
        bits_i = bits_avg + 0.5 * log2(variance_i / geometric_mean_variance)
    """

    def __init__(self, total_bits_per_pixel: float = 8.0):
        """
        Başlat.

        Args:
            total_bits_per_pixel: Toplam hedef bpp
        """
        self.total_bpp = total_bits_per_pixel
        self.band_allocations: Dict[str, float] = {}
        logger.info(f"OptimumBitAllocator initialized: target_bpp={total_bits_per_pixel}")

    def compute_allocation(self, coeffs: list) -> Dict[str, float]:
        """
        DWT katsayılarına variance-based bit allocation uygula.

        Args:
            coeffs: DWT katsayıları (pywt.wavedec2 formatı)

        Returns:
            Her bant için ayrılan bit sayısı (bpp)
        """
        # Her bandın varyansını hesapla
        band_variances = {}
        band_sizes = {}

        # LL bandı
        ll = coeffs[0].astype(np.float64)
        band_variances['LL'] = float(np.var(ll)) + 1e-10
        band_sizes['LL'] = ll.size

        # Detail bantları
        for level_idx, detail_tuple in enumerate(coeffs[1:], start=1):
            cH, cV, cD = detail_tuple
            band_variances[f'HL{level_idx}'] = float(np.var(cH.astype(np.float64))) + 1e-10
            band_variances[f'LH{level_idx}'] = float(np.var(cV.astype(np.float64))) + 1e-10
            band_variances[f'HH{level_idx}'] = float(np.var(cD.astype(np.float64))) + 1e-10

            band_sizes[f'HL{level_idx}'] = cH.size
            band_sizes[f'LH{level_idx}'] = cV.size
            band_sizes[f'HH{level_idx}'] = cD.size

        # Geometric mean of variances
        variances = np.array(list(band_variances.values()))
        geo_mean = float(np.exp(np.mean(np.log(variances))))

        # Optimal bit allocation (water-filling)
        total_pixels = sum(band_sizes.values())
        avg_bpp = self.total_bpp

        self.band_allocations = {}
        for band_name, variance in band_variances.items():
            # bits_i = avg + 0.5 * log2(sigma_i^2 / geo_mean)
            bits = avg_bpp + 0.5 * np.log2(variance / geo_mean)
            bits = max(0.0, bits)  # Negatif olamaz
            self.band_allocations[band_name] = float(round(bits, 3))

        logger.info(f"✓ Bit allocation computed for {len(self.band_allocations)} bands")
        return self.band_allocations

    def get_step_sizes(self, quality: int = 85) -> Dict[str, float]:
        """
        Bit allocation'a göre her bant için quantization step size hesapla.

        Args:
            quality: Base quality factor

        Returns:
            Her bant için step size
        """
        if not self.band_allocations:
            return {}

        from core.utils import quality_to_step_size
        base_step = quality_to_step_size(quality)

        step_sizes = {}
        max_bits = max(self.band_allocations.values()) if self.band_allocations else 1

        for band_name, bits in self.band_allocations.items():
            if max_bits > 0:
                # Daha çok bit → daha küçük step (daha az kayıp)
                ratio = bits / max_bits
                step_sizes[band_name] = base_step / max(0.1, ratio)
            else:
                step_sizes[band_name] = base_step

        return step_sizes

    def generate_report(self) -> str:
        """Bit allocation raporu."""
        if not self.band_allocations:
            return "No allocation computed yet"

        lines = [
            "=" * 50,
            "  OPTIMUM BIT ALLOCATION REPORT",
            "=" * 50,
            "",
            f"  Target bpp: {self.total_bpp}",
            "",
            f"  {'Band':<10} {'Allocated bpp':>15}",
            "  " + "-" * 30,
        ]

        for band, bits in sorted(self.band_allocations.items()):
            bar = "█" * int(bits * 3)
            lines.append(f"  {band:<10} {bits:>13.3f}  {bar}")

        total = sum(self.band_allocations.values())
        lines.extend([
            "  " + "-" * 30,
            f"  {'Total':<10} {total:>13.3f}",
            "",
            "  LL bands get more bits (important approximation)",
            "  HH bands get fewer bits (less important detail)",
            "=" * 50,
        ])

        return "\n".join(lines)
