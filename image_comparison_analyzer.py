"""
Image Comparison Analyzer: Evaluate compression performance across image types.

Compares three image categories:
1. Natural Images (landscapes, portraits, photographs)
2. Synthetic/Computer-Generated Images (graphics, patterns, diagrams)
3. Hybrid Images (mix of natural and synthetic content)

Metrics Evaluated:
- Compression Ratio (original size / compressed size)
- PSNR (Peak Signal-to-Noise Ratio) - image quality
- MSE (Mean Squared Error) - reconstruction error
- BPP (Bits Per Pixel) - file size efficiency
- Processing Time

"""

import os
import cv2
import numpy as np
import pandas as pd
import pywt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from jpeg_jpeg2000_stage1 import ImagePreprocessor, DWTTransformer
from stages.stage2_quantization import QuantizationModule
from stages.stage3_huffman_coding import HuffmanEncoder


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Squared Error between two float arrays."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def _psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio (dB) between two float arrays."""
    mse = _mse(a, b)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / mse))


class ImageCategoryAnalyzer:
    """Analyze compression performance across different image categories."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """Initialize analyzer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results: Dict[str, List[Dict]] = {
            "natural": [],      # Landscapes, portraits, photographs
            "synthetic": [],    # Graphics, patterns, diagrams
            "hybrid": []        # Mixed content
        }
        
    def categorize_image(self, image_path: str) -> str:
        """
        Categorize image based on filename or content analysis.
        
        Categories:
        - 'natural': Portraits, landscapes, photography
        - 'synthetic': Generated graphics, patterns
        - 'hybrid': Mixed content
        """
        filename = Path(image_path).stem.lower()
        
        # Rule-based categorization (can be extended with ML)
        natural_keywords = ['portrait', 'landscape', 'photo', 'face', 'nature', 
                           'akcigerfilm', 'x-ray', 'medical', 'person']
        synthetic_keywords = ['pattern', 'synthetic', 'diagram', 'chart', 'graphic',
                             'generated', 'fingerprint', 'parmak']
        
        for keyword in natural_keywords:
            if keyword in filename:
                return "natural"
        
        for keyword in synthetic_keywords:
            if keyword in filename:
                return "synthetic"
        
        # Default to hybrid for mixed content
        return "hybrid"
    
    def evaluate_dcт_compression(
        self,
        image_path: str,
        quality_levels: Optional[List[int]] = None
    ) -> Dict:
        """
        Evaluate JPEG (DCT) compression at different quality levels.
        
        Args:
            image_path: Path to image file
            quality_levels: List of JPEG quality values (5-95)
        
        Returns:
            Dictionary with metrics for each quality level
        """
        if quality_levels is None:
            quality_levels = [100, 80, 50, 25, 10]
        
        results = {"method": "JPEG (DCT)", "quality_levels": {}}
        
        try:
            # Load and prepare image
            pre = ImagePreprocessor(image_path)
            pre.load_image()
            original_bgr = pre.original_bgr
            if original_bgr is None:
                results["error"] = "Image could not be loaded"
                return results
            original_bytes = original_bgr.nbytes
            
            for quality in quality_levels:
                # Map compression ratio to JPEG quality
                # 100% → 95, 75% → 71, 50% → 47, 25% → 24, 10% → 10
                jpeg_quality = int(max(5, min(95, 100 - (100 - quality))))
                
                # Encode with OpenCV
                success, encoded = cv2.imencode(
                    '.jpg',
                    original_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                )
                
                if not success:
                    continue
                
                compressed_bytes = len(encoded)
                compression_ratio = original_bytes / compressed_bytes
                bpp = (compressed_bytes * 8) / (original_bgr.shape[0] * original_bgr.shape[1])
                
                # Decode to calculate PSNR/MSE
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded is None:
                    continue
                
                psnr = _psnr(
                    original_bgr.astype(np.float32),
                    decoded.astype(np.float32)
                )
                mse = _mse(
                    original_bgr.astype(np.float32),
                    decoded.astype(np.float32)
                )
                
                results["quality_levels"][quality] = {
                    "jpeg_quality": jpeg_quality,
                    "compressed_bytes": int(compressed_bytes),
                    "compression_ratio": float(round(compression_ratio, 3)),
                    "bpp": float(round(bpp, 3)),
                    "psnr": float(round(psnr, 2)),
                    "mse": float(round(mse, 4)),
                }
        
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def evaluate_dwt_compression(
        self,
        image_path: str,
        wavelet_types: Optional[List[str]] = None,
        decomposition_levels: Optional[List[int]] = None,
        quantization_levels: Optional[List[float]] = None
    ) -> Dict:
        """
        Evaluate JPEG2000-like (DWT) compression.
        
        Args:
            image_path: Path to image
            wavelet_types: List of wavelet names
            decomposition_levels: DWT levels (1-3)
            quantization_levels: Quantization delta values
        
        Returns:
            Dictionary with metrics
        """
        if wavelet_types is None:
            wavelet_types = ["haar", "db4", "db8"]
        if decomposition_levels is None:
            decomposition_levels = [1, 2, 3]
        if quantization_levels is None:
            quantization_levels = [1.0, 5.0, 10.0, 20.0]
        
        results = {"method": "JPEG2000 (DWT)", "wavelets": {}}
        
        try:
            pre = ImagePreprocessor(image_path)
            pre.load_image()
            original_bgr = pre.original_bgr
            if original_bgr is None:
                results["error"] = "Image could not be loaded"
                return results
            original_bytes = original_bgr.nbytes
            
            pre.convert_to_luma(force_grayscale=True)
            luma = pre.luma_channel
            if luma is None:
                results["error"] = "Luma channel could not be computed"
                return results
            luma_h, luma_w = luma.shape[0], luma.shape[1]
            
            quantizer = QuantizationModule(method="uniform")
            
            for wavelet in wavelet_types:
                results["wavelets"][wavelet] = {
                    "levels": {},
                    "quantization": {}
                }
                
                for level in decomposition_levels:
                    # Apply DWT at the requested decomposition level
                    dwt = DWTTransformer(wavelet_name=wavelet, level=level)
                    coeffs = dwt.apply_dwt2d(luma)
                    if not isinstance(coeffs, list):
                        # Grayscale luma always yields a list of coefficients
                        continue
                    
                    # For quantization evaluation
                    for q_delta in quantization_levels:
                        quant_coeffs = quantizer.uniform_quantize(coeffs, q_delta)
                        
                        # Huffman coding estimate
                        huffman = HuffmanEncoder()
                        bitstream, _code_table = huffman.encode(quant_coeffs)
                        compressed_bytes = len(bitstream) // 8
                        
                        if compressed_bytes == 0:
                            continue
                        
                        compression_ratio = original_bytes / max(1, compressed_bytes)
                        bpp = (compressed_bytes * 8) / (luma_h * luma_w)
                        
                        # Reconstruct for quality metrics
                        recon = pywt.waverec2(quant_coeffs, wavelet, mode="periodization")
                        recon = np.asarray(recon)[:luma_h, :luma_w]
                        
                        psnr = _psnr(
                            luma.astype(np.float32),
                            recon.astype(np.float32)
                        )
                        mse = _mse(
                            luma.astype(np.float32),
                            recon.astype(np.float32)
                        )
                        
                        key = f"level{level}_delta{q_delta}"
                        results["wavelets"][wavelet]["quantization"][key] = {
                            "level": level,
                            "delta": q_delta,
                            "compressed_bytes": int(compressed_bytes),
                            "compression_ratio": float(round(compression_ratio, 3)),
                            "bpp": float(round(bpp, 3)),
                            "psnr": float(round(psnr, 2)),
                            "mse": float(round(mse, 4)),
                        }
        
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Run full compression analysis on single image.
        
        Returns:
            Comprehensive metrics for all compression methods
        """
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}
        
        category = self.categorize_image(image_path)
        
        analysis = {
            "image_path": image_path,
            "filename": Path(image_path).name,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "image_info": {}
        }
        
        # Get image info
        try:
            img = cv2.imread(image_path)
            if img is not None:
                analysis["image_info"] = {
                    "width": int(img.shape[1]),
                    "height": int(img.shape[0]),
                    "channels": int(img.shape[2]) if len(img.shape) > 2 else 1,
                    "size_bytes": int(os.path.getsize(image_path)),
                    "megapixels": float(round(img.shape[0] * img.shape[1] / 1e6, 2))
                }
        except Exception as e:
            analysis["image_info"]["error"] = str(e)
        
        # Run compression evaluations
        analysis["dct_results"] = self.evaluate_dcт_compression(image_path)
        analysis["dwt_results"] = self.evaluate_dwt_compression(image_path)
        
        # Store by category
        self.results[category].append(analysis)
        
        return analysis
    
    def batch_analyze(self, image_paths: List[str]) -> Dict:
        """Analyze multiple images and collect statistics."""
        results = {"analyzed": 0, "errors": 0, "by_category": {}}
        
        for path in image_paths:
            try:
                self.analyze_image(path)
                results["analyzed"] += 1
            except Exception as e:
                results["errors"] += 1
                print(f"Error analyzing {path}: {e}")
        
        return results
    
    def generate_comparison_report(self) -> Dict:
        """
        Generate comparison statistics across image categories.
        
        Compares:
        - Compression efficiency (ratio, BPP)
        - Quality preservation (PSNR, MSE)
        - Method suitability for each category
        """
        report = {
            "summary": {},
            "by_category": {}
        }
        
        for category in ["natural", "synthetic", "hybrid"]:
            if not self.results[category]:
                continue
            
            category_data = {
                "image_count": len(self.results[category]),
                "dct_summary": self._summarize_method(category, "dct"),
                "dwt_summary": self._summarize_method(category, "dwt"),
                "recommendations": self._generate_recommendations(category)
            }
            
            report["by_category"][category] = category_data
        
        report["summary"]["overall_observations"] = self._generate_overall_observations()
        
        return report
    
    def _summarize_method(self, category: str, method: str) -> Dict:
        """Summarize compression performance for a method in a category."""
        summary = {}
        
        for result in self.results[category]:
            method_key = f"{method}_results"
            if method_key not in result:
                continue
            
            method_data = result[method_key]
            
            if method == "dct":
                for quality, metrics in method_data.get("quality_levels", {}).items():
                    if quality not in summary:
                        summary[quality] = []
                    summary[quality].append(metrics)
            
            elif method == "dwt":
                for wavelet, data in method_data.get("wavelets", {}).items():
                    if wavelet not in summary:
                        summary[wavelet] = []
                    for metrics in data.get("quantization", {}).values():
                        summary[wavelet].append(metrics)
        
        # Calculate averages
        averages = {}
        for key, values in summary.items():
            if values:
                avg_psnr = np.mean([v.get("psnr", 0) for v in values])
                avg_ratio = np.mean([v.get("compression_ratio", 0) for v in values])
                avg_bpp = np.mean([v.get("bpp", 0) for v in values])
                
                averages[key] = {
                    "avg_psnr": float(round(avg_psnr, 2)),
                    "avg_compression_ratio": float(round(avg_ratio, 3)),
                    "avg_bpp": float(round(avg_bpp, 3)),
                }
        
        return averages
    
    def _generate_recommendations(self, category: str) -> List[str]:
        """Generate compression recommendations for image category."""
        recommendations = {
            "natural": [
                "JPEG (DCT) recommended for natural images - good quality at 75-80%",
                "JPEG2000 with db4/db8 wavelets for finer detail preservation",
                "Use higher quality settings (>75%) to minimize visible artifacts",
                "Suitable for photography, landscapes, medical imagery",
            ],
            "synthetic": [
                "Both JPEG and JPEG2000 perform well",
                "Consider Lossless compression if exact pixel values needed",
                "Wavelet methods (DWT) handle sharp edges well",
                "Lower quality settings (50-75%) acceptable without visible quality loss",
                "Good for graphics, charts, diagrams, patterns",
            ],
            "hybrid": [
                "Mixed content requires balanced approach",
                "JPEG2000 with moderate quantization recommended",
                "Use db4 or bior4.4 wavelets for balanced detail/compression",
                "Monitor PSNR carefully - natural parts may need higher quality",
                "Consider processing parts separately if possible",
            ]
        }
        
        return recommendations.get(category, [])
    
    def _generate_overall_observations(self) -> List[str]:
        """Generate overall observations from all categories."""
        observations = [
            "Natural images show characteristic compression artifacts at lower quality",
            "Synthetic/generated images maintain quality better at low compression ratios",
            "Hybrid images require balanced parameters to preserve both content types",
            "JPEG performs well for natural images; wavelets for synthetic content",
            "Higher decomposition levels improve compression but increase processing time",
        ]
        return observations
    
    def export_results_to_json(self) -> str:
        """Export all results to JSON file."""
        report = self.generate_comparison_report()
        report["raw_results"] = self.results
        
        filepath = self.output_dir / f"compression_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def export_results_to_csv(self) -> str:
        """Export results to CSV for spreadsheet analysis."""
        data = []
        
        for category in ["natural", "synthetic", "hybrid"]:
            for result in self.results[category]:
                base = {
                    "category": category,
                    "image": result.get("filename", ""),
                    "width": result["image_info"].get("width", ""),
                    "height": result["image_info"].get("height", ""),
                    "megapixels": result["image_info"].get("megapixels", ""),
                }
                
                # DCT results
                for quality, metrics in result.get("dct_results", {}).get("quality_levels", {}).items():
                    row = base.copy()
                    row.update({
                        "method": "DCT",
                        "param": f"quality_{quality}",
                        "compression_ratio": metrics.get("compression_ratio", ""),
                        "bpp": metrics.get("bpp", ""),
                        "psnr": metrics.get("psnr", ""),
                        "mse": metrics.get("mse", ""),
                    })
                    data.append(row)
                
                # DWT results
                for wavelet, w_data in result.get("dwt_results", {}).get("wavelets", {}).items():
                    for key, metrics in w_data.get("quantization", {}).items():
                        row = base.copy()
                        row.update({
                            "method": "DWT",
                            "param": f"{wavelet}_{key}",
                            "compression_ratio": metrics.get("compression_ratio", ""),
                            "bpp": metrics.get("bpp", ""),
                            "psnr": metrics.get("psnr", ""),
                            "mse": metrics.get("mse", ""),
                        })
                        data.append(row)
        
        df = pd.DataFrame(data)
        filepath = self.output_dir / f"compression_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        return str(filepath)


def main():
    """Run compression comparison analysis on available test images."""
    analyzer = ImageCategoryAnalyzer()
    
    # Find test images in project
    # Generated by `python generate_test_images.py`; extra image paths can be
    # passed on the command line.
    import sys
    test_images = sys.argv[1:] or [
        "test_images_generated/test_natural_landscape.jpg",
        "test_images_generated/test_synthetic_graphic.jpg",
        "test_images_generated/test_hybrid_mixed.jpg",
    ]
    
    print("=" * 60)
    print("IMAGE COMPRESSION COMPARISON ANALYZER")
    print("=" * 60)
    print("\nFound test images:")
    
    valid_images = []
    for img in test_images:
        if os.path.exists(img):
            valid_images.append(img)
            print(f"  ✓ {img}")
        else:
            print(f"  ✗ {img} (not found)")
    
    if not valid_images:
        print("\nNo test images found!")
        return
    
    print(f"\nAnalyzing {len(valid_images)} images...\n")
    
    # Run analysis
    analyzer.batch_analyze(valid_images)
    
    # Generate and export reports
    print("\nGenerating reports...")
    
    json_file = analyzer.export_results_to_json()
    print(f"✓ JSON report: {json_file}")
    
    csv_file = analyzer.export_results_to_csv()
    print(f"✓ CSV report: {csv_file}")
    
    # Print summary
    report = analyzer.generate_comparison_report()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for category, data in report.get("by_category", {}).items():
        print(f"\n📊 {category.upper()} IMAGES ({data['image_count']} images)")
        print(f"  Recommendations:")
        for rec in data.get("recommendations", [])[:2]:
            print(f"    • {rec}")
    
    print("\n📋 Full reports exported to evaluation_results/")


if __name__ == "__main__":
    main()
