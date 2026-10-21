"""
image_classifier.py — Image Type Classification

Görüntü türü sınıflandırması:
- Natural: Doğal fotograflar (camera, phone)
- Synthetic: Sentetik görüntüler (graphics, render, diagram)
- Hybrid: Karışık (natural + synthetic elements)

Sınıflandırma kriterleri:
- Entropy analysis
- Texture analysis
- Edge distribution
- Color space variance
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any, List
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageType(Enum):
    """Görüntü türleri"""
    NATURAL = "natural"
    SYNTHETIC = "synthetic"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class ImageClassifier:
    """Görüntü sınıflandırıcı"""
    
    def __init__(self):
        """Başlat"""
        logger.info("ImageClassifier initialized")
    
    def classify(self, image: np.ndarray) -> Tuple[ImageType, Dict[str, float]]:
        """Görüntüyü sınıflandır"""
        # Grayscale'e dönüştür
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        # Değer aralığını kontrol et (0-1 ise 0-255'e çek)
        if image.max() <= 1.01:
            image = (image * 255.0)
            
        image = image.astype(np.float64)
        
        # Özellik hesapla
        features = self._extract_features(image)
        
        # Sınıflandır
        image_type = self._classify_features(features)
        
        logger.info(f"Image classified as: {image_type.value}")
        return image_type, features
    
    def _extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Görüntü özelliklerini çıkar (Max Hassasiyet)"""
        features = {}
        
        # 0. Kesin Sayım (uint8)
        img_u8 = np.clip(image, 0, 255).astype(np.uint8)
        unique_colors = len(np.unique(img_u8))
        features['unique_colors'] = unique_colors
        
        # 1. Entropy
        features['entropy'] = self._calculate_entropy(image)
        
        # 2. Color Sparsity (128 renk altı sentetiklik kanıtıdır)
        features['color_sparsity'] = 1.0 - (min(unique_colors, 128) / 128.0)
        
        # 3. Kenar Keskinliği (Eşik 10 -> 2'ye çekildi)
        features['edge_sharpness'] = self._calculate_edge_sharpness(image, threshold=2)
        
        # 4. Boş Alan Oranı
        features['flat_areas'] = self._calculate_flat_areas(image)
        
        # 5. Texture Regularity
        features['regularity'] = self._calculate_regularity_fast(image)
        
        return features

    def _calculate_edge_sharpness(self, image: np.ndarray, threshold: int = 2) -> float:
        """İnce çizgileri yakalamak için düşük eşik."""
        dx = np.abs(np.diff(image, axis=1))
        dy = np.abs(np.diff(image, axis=0))
        edges = np.concatenate([dx.ravel(), dy.ravel()])
        
        strong_edges = edges[edges > threshold]
        if len(strong_edges) == 0: return 0.0
        
        # Keskinlik: Kenarların ne kadar 'net' olduğu
        sharpness = np.mean(strong_edges) / 255.0
        return float(np.clip(sharpness * 3.0, 0, 1))

    def _calculate_entropy(self, image: np.ndarray) -> float:
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        return float(np.clip(entropy / 8.0, 0, 1))

    def _calculate_flat_areas(self, image: np.ndarray) -> float:
        """Yan yana aynı olan piksel oranı."""
        dx = np.diff(image, axis=1) == 0
        dy = np.diff(image, axis=0) == 0
        flat_ratio = (np.sum(dx) + np.sum(dy)) / (dx.size + dy.size)
        return float(np.clip(flat_ratio, 0, 1))

    def _calculate_regularity_fast(self, image: np.ndarray) -> float:
        """Lokal varyansın homojenliği (Vektörel hızlı versiyon)."""
        if image.shape[0] < 16 or image.shape[1] < 16: return 0.5
        h, w = image.shape
        h_trim, w_trim = (h // 8) * 8, (w // 8) * 8
        blocks = image[:h_trim, :w_trim].reshape(h_trim // 8, 8, w_trim // 8, 8).swapaxes(1, 2)
        block_vars = np.var(blocks, axis=(2, 3))
        regularity = 1.0 - (np.std(block_vars) / (np.mean(block_vars) + 1.0))
        return float(np.clip(regularity, 0, 1))

    def _classify_features(self, features: Dict[str, float]) -> ImageType:
        """
        Sistematik Karar Mekanizması (Decision Tree):
        Hocanın istediği değerlendirme kriterlerine göre kesin filtreleme.
        """
        u_colors = features['unique_colors']
        flatness = features['flat_areas']
        sharpness = features['edge_sharpness']
        entropy = features['entropy']
        
        # DEBUG LOG (Konsola yazar, analiz için)
        logger.warning(f"--- ANALİZ: Renk={u_colors}, Düzlük={flatness:.2f}, Keskinlik={sharpness:.2f}, Entropi={entropy:.2f}")
        
        # 1. Kural: Renk Sayısı (Siyah-beyaz resimde 80'den az ton kesinlikle sentetiktir)
        if u_colors < 80:
            return ImageType.SYNTHETIC
            
        # 2. Kural: Yüksek Düzlük ve Düşük Entropi (Pixel Art / Grafik)
        if flatness > 0.45 and entropy < 0.6:
            return ImageType.SYNTHETIC
            
        # 3. Kural: Hibrit Tespiti (Fotoğraf üzerine yazı veya grafik)
        if 0.25 < flatness < 0.5 and u_colors > 120:
            return ImageType.HYBRID
            
        # 4. Kural: Doğallık Testi (Çok fazla renk tonu ve düşük düzlük)
        if u_colors > 180 and flatness < 0.25:
            return ImageType.NATURAL
            
        # 5. Kural: Karmaşık Sentetikler
        if sharpness > 0.6 and flatness > 0.3:
            return ImageType.SYNTHETIC
            
        # Varsayılan Karar
        if u_colors > 150:
            return ImageType.NATURAL
        else:
            return ImageType.HYBRID
    
    def get_classification_confidence(self, features: Dict[str, float]) -> float:
        """
        Sınıflandırma güvenini hesapla (0-1)
        
        Args:
            features: Özellik scores
        
        Returns:
            Confidence score
        """
        # Özellikler arasındaki tutarlılığı kontrol et
        natural_consistency = [
            features['entropy'] > 0.6,
            features['histogram_uniformity'] < 0.6,
            features['edge_complexity'] > 0.5,
            features['color_variance'] > 0.5,
            features['texture_regularity'] < 0.5,
            features['frequency_spread'] > 0.6
        ]
        
        synthetic_consistency = [
            features['entropy'] < 0.5,
            features['histogram_uniformity'] > 0.6,
            features['edge_complexity'] < 0.5,
            features['color_variance'] < 0.5,
            features['texture_regularity'] > 0.5,
            features['frequency_spread'] < 0.5
        ]
        
        natural_matches = sum(natural_consistency) / len(natural_consistency)
        synthetic_matches = sum(synthetic_consistency) / len(synthetic_consistency)
        
        # Confidence: maximum match ratio
        confidence = max(natural_matches, synthetic_matches)
        
        return float(confidence)


class ImageTypeEvaluator:
    """Görüntü türü değerlendirmesi"""
    
    def __init__(self):
        """Başlat"""
        self.classifier = ImageClassifier()
    
    def evaluate_compression_impact(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        image_type: ImageType,
        quality_metric: float
    ) -> Dict[str, Any]:
        """
        Sıkıştırmanın görüntü türüne göre etkisini değerlendir
        
        Args:
            original: Orijinal görüntü
            reconstructed: Reconstructed görüntü
            image_type: Görüntü türü
            quality_metric: Kalite metriği (PSNR, SSIM, vb.)
        
        Returns:
            Evaluation results
        """
        results = {
            'image_type': image_type.value,
            'quality_metric': quality_metric,
            'acceptable': False,
            'recommendation': ''
        }
        
        # Image type-based quality thresholds
        thresholds = {
            ImageType.NATURAL: 28,      # Natural: 28+ dB is good
            ImageType.SYNTHETIC: 32,    # Synthetic: 32+ dB needed
            ImageType.HYBRID: 30,       # Hybrid: 30+ dB
            ImageType.UNKNOWN: 30
        }
        
        threshold = thresholds.get(image_type, 30)
        
        results['threshold'] = threshold
        results['acceptable'] = quality_metric >= threshold
        
        if results['acceptable']:
            results['recommendation'] = f"✓ Lossy acceptable ({quality_metric:.1f} dB >= {threshold} dB)"
        else:
            results['recommendation'] = f"✗ Use lossless ({quality_metric:.1f} dB < {threshold} dB)"
        
        return results
    
    def get_recommended_mode(self, image_type: ImageType) -> str:
        """
        Görüntü türüne göre önerilen modu al
        
        Args:
            image_type: Görüntü türü
        
        Returns:
            'lossy' or 'lossless'
        """
        if image_type in (ImageType.SYNTHETIC, ImageType.HYBRID):
            return 'lossy'  # DCT recommended
        elif image_type == ImageType.NATURAL:
            return 'lossy'  # DWT recommended
        else:
            return 'lossy'  # Default
    
    def get_recommended_quality(self, image_type: ImageType) -> int:
        """
        Görüntü türüne göre önerilen kaliteyi al
        
        Args:
            image_type: Görüntü türü
        
        Returns:
            Quality factor (0-100)
        """
        recommendations = {
            ImageType.NATURAL: 85,      # High quality for natural
            ImageType.SYNTHETIC: 90,    # Very high for synthetic
            ImageType.HYBRID: 85,       # High for mixed
            ImageType.UNKNOWN: 85
        }
        
        return recommendations.get(image_type, 85)
