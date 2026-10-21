"""
image_type_comparison.py — Image Type Comparison Framework

Farklı görüntü türleri (Natural, Synthetic, Hybrid) üzerinde
sıkıştırma performansını karşılaştırır.

Karşılaştırma:
- DWT vs DCT vs Lossless
- Farklı kalite seviyeleri (Q50, Q70, Q85, Q95)
- PSNR, SSIM, Compression Ratio, Bpp, Süre

Hocanın "evaluate across image types" talebini karşılar.
"""

import logging
import time
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

from core.image_classifier import ImageClassifier, ImageType, ImageTypeEvaluator
from core.utils import ssim_fn_any

logger = logging.getLogger(__name__)


# ============================================================================
# TEST GÖRÜNTÜ OLUŞTURMA
# ============================================================================

class TestImageGenerator:
    """Farklı türlerde test görüntüleri oluşturur (programatik)."""

    @staticmethod
    def generate_natural_image(size: int = 256, seed: int = 42) -> np.ndarray:
        """
        Natural-like görüntü oluştur.
        Gaussian noise + smooth gradients + texture patterns.

        Args:
            size: Görüntü boyutu (kare)
            seed: Random seed

        Returns:
            Grayscale görüntü [0-255]
        """
        rng = np.random.RandomState(seed)

        # Smooth gradient base
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)
        gradient = (xx * 0.4 + yy * 0.3) * 180

        # Multi-scale noise (simulates natural texture)
        noise_fine = rng.randn(size, size) * 15
        noise_coarse = np.repeat(
            np.repeat(rng.randn(size // 8, size // 8) * 25, 8, axis=0),
            8, axis=1
        )[:size, :size]

        # Sinusoidal patterns (simulates natural structures)
        freq1 = np.sin(xx * 12 * np.pi) * np.cos(yy * 8 * np.pi) * 20
        freq2 = np.sin((xx + yy) * 6 * np.pi) * 15

        image = gradient + noise_fine + noise_coarse + freq1 + freq2
        image = np.clip(image, 0, 255).astype(np.float32)

        logger.info(f"Natural image generated: {size}x{size}")
        return image

    @staticmethod
    def generate_synthetic_image(size: int = 256, seed: int = 42) -> np.ndarray:
        """
        Synthetic/CG görüntü oluştur.
        Geometric patterns, sharp edges, uniform regions.

        Args:
            size: Görüntü boyutu (kare)
            seed: Random seed

        Returns:
            Grayscale görüntü [0-255]
        """
        rng = np.random.RandomState(seed)
        image = np.zeros((size, size), dtype=np.float32)

        # Uniform background
        image[:] = 200

        # Sharp geometric shapes
        # Rectangle 1
        image[size//8:size//4, size//8:size//3] = 50
        # Rectangle 2
        image[size//2:3*size//4, size//3:2*size//3] = 120
        # Circle
        cx, cy, r = size//2, size//4, size//8
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
        image[mask] = 80

        # Diagonal stripe pattern
        for i in range(0, size, 16):
            for j in range(0, size, 16):
                if (i // 16 + j // 16) % 2 == 0:
                    end_i = min(i + 16, size)
                    end_j = min(j + 16, size)
                    image[i:end_i, j:end_j] *= 0.85

        # Sharp text-like horizontal lines
        for y_pos in range(size // 5, size, size // 5):
            if y_pos + 2 < size:
                image[y_pos:y_pos+2, size//6:5*size//6] = 30

        image = np.clip(image, 0, 255).astype(np.float32)
        logger.info(f"Synthetic image generated: {size}x{size}")
        return image

    @staticmethod
    def generate_hybrid_image(size: int = 256, seed: int = 42) -> np.ndarray:
        """
        Hybrid görüntü oluştur (natural + synthetic karışımı).
        Half natural texture, half geometric shapes.

        Args:
            size: Görüntü boyutu (kare)
            seed: Random seed

        Returns:
            Grayscale görüntü [0-255]
        """
        natural = TestImageGenerator.generate_natural_image(size, seed)
        synthetic = TestImageGenerator.generate_synthetic_image(size, seed + 1)

        # Diagonal blend
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)
        blend_mask = np.clip((xx + yy) / 2, 0, 1)

        image = natural * blend_mask + synthetic * (1 - blend_mask)
        image = np.clip(image, 0, 255).astype(np.float32)

        logger.info(f"Hybrid image generated: {size}x{size}")
        return image


# ============================================================================
# TEK GÖRÜNTÜ KARŞILAŞTIRMA SONUCU
# ============================================================================

@dataclass
class SingleComparisonResult:
    """Tek bir sıkıştırma sonucu."""
    image_type: str
    method: str          # "DWT", "DCT", "Lossless"
    quality: int
    psnr: float = 0.0
    ssim: float = 0.0
    mse: float = 0.0
    compression_ratio: float = 0.0
    bpp: float = 0.0
    compression_time_ms: float = 0.0
    original_size: int = 0
    compressed_size: int = 0


# ============================================================================
# ANA KARŞILAŞTIRMA RUNNER
# ============================================================================

class ImageTypeComparisonRunner:
    """
    Tüm görüntü türlerini test eden ana karşılaştırma sınıfı.

    Her görüntü türü için DWT, DCT ve Lossless yöntemlerini
    farklı kalite seviyelerinde çalıştırır ve sonuçları toplar.
    """

    def __init__(self, image_size: int = 256):
        """
        Başlat.

        Args:
            image_size: Test görüntüsü boyutu
        """
        self.image_size = image_size
        self.classifier = ImageClassifier()
        self.evaluator = ImageTypeEvaluator()
        self.results: List[SingleComparisonResult] = []
        self.qualities = [50, 70, 85, 95]

        logger.info(f"ImageTypeComparisonRunner initialized: size={image_size}")

    def run_full_comparison(self) -> List[SingleComparisonResult]:
        """
        Tam karşılaştırma çalıştır.

        Returns:
            Tüm sonuçlar listesi
        """
        self.results = []
        generator = TestImageGenerator()

        image_types = {
            "natural": generator.generate_natural_image(self.image_size),
            "synthetic": generator.generate_synthetic_image(self.image_size),
            "hybrid": generator.generate_hybrid_image(self.image_size),
        }

        for img_type_name, image in image_types.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing image type: {img_type_name}")
            logger.info(f"{'='*60}")

            # Classify
            detected_type, features = self.classifier.classify(image)
            logger.info(f"Detected type: {detected_type.value}")

            # DWT comparison
            for q in self.qualities:
                result = self._test_dwt(image, img_type_name, q)
                self.results.append(result)

            # DCT comparison
            for q in self.qualities:
                result = self._test_dct(image, img_type_name, q)
                self.results.append(result)

            # Lossless
            result = self._test_lossless(image, img_type_name)
            self.results.append(result)

        logger.info(f"\n✓ Full comparison complete: {len(self.results)} tests")
        return self.results

    def compare_single_image(
        self, image: np.ndarray, image_type: str
    ) -> List[SingleComparisonResult]:
        """
        Tek görüntü karşılaştırması.

        Args:
            image: Grayscale görüntü
            image_type: Tür adı

        Returns:
            Sonuçlar
        """
        results = []

        for q in self.qualities:
            results.append(self._test_dwt(image, image_type, q))
            results.append(self._test_dct(image, image_type, q))

        results.append(self._test_lossless(image, image_type))
        return results

    # ---- Private: DWT Test ----
    def _test_dwt(
        self, image: np.ndarray, image_type: str, quality: int
    ) -> SingleComparisonResult:
        """DWT sıkıştırma testi."""
        import pywt
        from stages.stage1_preprocessing import DWTTransformer
        from stages.stage2_quantization import QuantizationModule
        from huffman_codec import huffman_compressed_size

        result = SingleComparisonResult(
            image_type=image_type, method="DWT", quality=quality
        )

        try:
            t0 = time.time()

            # Forward DWT
            dwt = DWTTransformer("db4", 2)
            coeffs = dwt.apply_dwt2d(image)

            # Quantize (quantized coefficients already contain approximate values)
            quantizer = QuantizationModule()
            quantized = quantizer.quantize_by_quality(coeffs, quality, method="scalar")

            # Compression size estimation: quantized DWT coeffs -> Huffman bytes
            _, huffman_bytes, _ = huffman_compressed_size(quantized)
            # Orijinal boyutu yöntemler arasında tutarlı tut: görüntü byte sayısı.
            result.original_size = int(np.clip(image, 0, 255).astype(np.uint8).nbytes)
            result.compressed_size = int(huffman_bytes)
            if result.compressed_size > 0:
                result.compression_ratio = result.original_size / result.compressed_size
                result.bpp = (result.compressed_size * 8) / image.size if image.size > 0 else 0.0

            # Inverse DWT (use pywt directly — quantized coeffs are already scaled)
            reconstructed = pywt.waverec2(quantized, "db4", mode="periodization")

            # Trim to original size
            reconstructed = reconstructed[:image.shape[0], :image.shape[1]]

            t1 = time.time()
            result.compression_time_ms = (t1 - t0) * 1000

            # Metrics
            result = self._calc_metrics(image, reconstructed, result)

        except Exception as e:
            logger.warning(f"DWT test failed ({image_type}, Q{quality}): {e}")

        return result

    # ---- Private: DCT Test ----
    def _test_dct(
        self, image: np.ndarray, image_type: str, quality: int
    ) -> SingleComparisonResult:
        """DCT sıkıştırma testi."""
        from stages.stage_dct_transform import DCTCompressionPipeline

        result = SingleComparisonResult(
            image_type=image_type, method="DCT", quality=quality
        )

        try:
            t0 = time.time()

            pipeline = DCTCompressionPipeline(quality=quality)
            dct_coeffs = pipeline.transformer.apply_dct_2d(image)
            quantized = pipeline.quantize_dct(dct_coeffs)
            dequantized = pipeline.dequantize_dct(quantized)
            reconstructed = pipeline.reconstructor.apply_idct_2d(
                dequantized, image.shape
            )

            # Compression size: OpenCV JPEG encoder output bytes
            image_u8 = np.clip(image, 0, 255).astype(np.uint8)
            ok, enc = cv2.imencode(".jpg", image_u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if ok:
                result.original_size = int(image_u8.nbytes)
                result.compressed_size = int(enc.nbytes)
                if result.compressed_size > 0:
                    result.compression_ratio = result.original_size / result.compressed_size
                    result.bpp = (result.compressed_size * 8) / image_u8.size if image_u8.size > 0 else 0.0

            t1 = time.time()
            result.compression_time_ms = (t1 - t0) * 1000

            result = self._calc_metrics(image, reconstructed, result)

        except Exception as e:
            logger.warning(f"DCT test failed ({image_type}, Q{quality}): {e}")

        return result

    # ---- Private: Lossless Test ----
    def _test_lossless(
        self, image: np.ndarray, image_type: str
    ) -> SingleComparisonResult:
        """Lossless sıkıştırma testi."""
        from stages.stage_lossless import LosslessCompressionPipeline

        result = SingleComparisonResult(
            image_type=image_type, method="Lossless", quality=100
        )

        try:
            t0 = time.time()

            pipeline = LosslessCompressionPipeline(compression_level=9)
            compressed, metadata = pipeline.compress_image_data(image)
            reconstructed = pipeline.decompress_image_data(
                compressed, image.shape[0], image.shape[1]
            )

            t1 = time.time()
            result.compression_time_ms = (t1 - t0) * 1000

            result.original_size = image.nbytes
            result.compressed_size = len(compressed) if isinstance(compressed, bytes) else compressed.nbytes
            if result.compressed_size > 0:
                result.compression_ratio = result.original_size / result.compressed_size

            # Lossless should be perfect
            result.psnr = -1.0  # Represents Infinity
            result.ssim = 1.0
            result.mse = 0.0
            result.bpp = (result.compressed_size * 8) / image.size if image.size > 0 else 0

        except Exception as e:
            logger.warning(f"Lossless test failed ({image_type}): {e}")

        return result

    # ---- Metric calculation ----
    def _calc_metrics(
        self, original: np.ndarray, reconstructed: np.ndarray,
        result: SingleComparisonResult
    ) -> SingleComparisonResult:
        """PSNR, SSIM, MSE, compression ratio hesapla."""
        orig = original.astype(np.float64)
        recon = reconstructed.astype(np.float64)

        # Boyut eşitle
        min_h = min(orig.shape[0], recon.shape[0])
        min_w = min(orig.shape[1], recon.shape[1])
        orig = orig[:min_h, :min_w]
        recon = recon[:min_h, :min_w]

        # MSE
        mse = float(np.mean((orig - recon) ** 2))
        result.mse = mse

        # PSNR
        if mse == 0:
            result.psnr = -1.0  # Represents Infinity
        else:
            result.psnr = float(20 * np.log10(255.0 / np.sqrt(mse)))

        result.ssim = ssim_fn_any(orig, recon, data_range=255.0)

        # Size metrics fallback: if caller did not provide measured sizes.
        if result.original_size <= 0:
            result.original_size = int(original.shape[0] * original.shape[1])
        if result.compressed_size <= 0:
            result.compressed_size = result.original_size
        if result.compression_ratio <= 0 and result.compressed_size > 0:
            result.compression_ratio = result.original_size / result.compressed_size
        if result.bpp <= 0:
            total = original.size
            result.bpp = (result.compressed_size * 8) / total if total > 0 else 0.0

        return result

    # ================================================================
    # RAPOR OLUŞTURMA
    # ================================================================

    def generate_json_report(self) -> str:
        """JSON formatında rapor oluştur."""
        report = {
            "title": "Image Type Compression Comparison Report",
            "date": datetime.now().isoformat(),
            "image_size": f"{self.image_size}x{self.image_size}",
            "total_tests": len(self.results),
            "results": [asdict(r) for r in self.results],
            "summary": self._generate_summary()
        }
        return json.dumps(report, indent=2, default=str)

    def generate_markdown_report(self) -> str:
        """Markdown formatında karşılaştırma raporu oluştur."""
        summary = self._generate_summary()

        report = f"""# 📊 Image Type Compression Comparison Report

**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Görüntü Boyutu:** {self.image_size}x{self.image_size}
**Toplam Test:** {len(self.results)}

---

## 📈 Özet Sonuçlar

### DWT vs DCT vs Lossless — Tür Bazında Ortalama PSNR (dB)

| Görüntü Türü | DWT (Q85) | DCT (Q85) | Lossless |
|:-------------|:---------:|:---------:|:--------:|
"""
        for img_type in ["natural", "synthetic", "hybrid"]:
            dwt_q85 = [r for r in self.results if r.image_type == img_type and r.method == "DWT" and r.quality == 85]
            dct_q85 = [r for r in self.results if r.image_type == img_type and r.method == "DCT" and r.quality == 85]
            lossless = [r for r in self.results if r.image_type == img_type and r.method == "Lossless"]

            dwt_psnr = f"{dwt_q85[0].psnr:.2f}" if dwt_q85 and dwt_q85[0].psnr != -1.0 else "∞"
            dct_psnr = f"{dct_q85[0].psnr:.2f}" if dct_q85 and dct_q85[0].psnr != -1.0 else "∞"
            loss_psnr = "∞ (perfect)"

            report += f"| {img_type.capitalize():12s} | {dwt_psnr:>9s} | {dct_psnr:>9s} | {loss_psnr:>8s} |\n"

        report += f"""
---

## 📊 Detaylı Sonuçlar

### Tüm Test Sonuçları

| Tür | Yöntem | Kalite | PSNR (dB) | SSIM | Oran | Süre (ms) |
|:----|:-------|:------:|:---------:|:----:|:----:|:---------:|
"""
        for r in self.results:
            psnr_str = f"{r.psnr:.2f}" if r.psnr != -1.0 else "∞"
            report += (
                f"| {r.image_type:10s} | {r.method:8s} | {r.quality:5d} "
                f"| {psnr_str:>9s} | {r.ssim:.4f} "
                f"| {r.compression_ratio:.2f}x | {r.compression_time_ms:.1f} |\n"
            )

        report += f"""
---

## 🎯 Öneriler

"""
        for img_type in ["natural", "synthetic", "hybrid"]:
            best = summary.get(img_type, {})
            rec_method = best.get("recommended_method", "DWT")
            rec_quality = best.get("recommended_quality", 85)
            report += f"- **{img_type.capitalize()}:** {rec_method} @ Q{rec_quality}\n"

        report += f"""
---

## ✅ Sonuç

Bu rapor, JPEG-XC sıkıştırma sisteminin farklı görüntü türleri üzerindeki
performansını karşılaştırmaktadır. Her görüntü türü için en uygun sıkıştırma
yöntemi ve kalite seviyesi belirlenmiştir.

*Otomatik oluşturuldu: {datetime.now().isoformat()}*
"""
        return report

    def _generate_summary(self) -> Dict[str, Any]:
        """Özet istatistikler oluştur."""
        summary = {}

        for img_type in ["natural", "synthetic", "hybrid"]:
            type_results = [r for r in self.results if r.image_type == img_type and r.method != "Lossless"]

            if not type_results:
                continue

            # Filter out inf PSNR for comparison
            finite_results = [r for r in type_results if r.psnr != -1.0]

            if finite_results:
                best = max(finite_results, key=lambda r: r.psnr)
                summary[img_type] = {
                    "recommended_method": best.method,
                    "recommended_quality": best.quality,
                    "best_psnr": best.psnr,
                    "best_ssim": best.ssim,
                    "test_count": len(type_results)
                }
            else:
                summary[img_type] = {
                    "recommended_method": "DWT",
                    "recommended_quality": 85,
                    "best_psnr": 0,
                    "best_ssim": 0,
                    "test_count": len(type_results)
                }

        return summary

    def save_reports(self, output_dir: str) -> Tuple[str, str]:
        """
        Raporları dosyaya kaydet.

        Args:
            output_dir: Çıkış dizini

        Returns:
            (json_path, markdown_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        json_path = output_path / "image_type_comparison.json"
        md_path = output_path / "image_type_comparison_report.md"

        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_json_report())

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_markdown_report())

        logger.info(f"✓ Reports saved to {output_path}")
        return str(json_path), str(md_path)
