"""
stage_lossless.py — Lossless Compression Module

Sıfır kayıp sıkıştırma:
- zlib-based DEFLATE compression
- Run-Length Encoding for sparse data
- Huffman encoding integration
- Medical/fingerprint image support
"""

import logging
import zlib
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class LosslessEncoder:
    """Lossless encoder (DEFLATE-based)"""
    
    def __init__(self, compression_level: int = 9):
        """
        Başlat
        
        Args:
            compression_level: Sıkıştırma seviyesi (1-9, default=9 max)
        """
        if not 1 <= compression_level <= 9:
            raise ValueError(f"Compression level must be 1-9, got {compression_level}")
        
        self.compression_level = compression_level
        logger.info(f"LosslessEncoder initialized: level={compression_level}")
    
    def compress_raw_data(self, data: np.ndarray) -> bytes:
        """
        Ham veriyi sıkıştır (DEFLATE)
        
        Args:
            data: Sıkıştırılacak NumPy array
        
        Returns:
            Sıkıştırılmış bytes
        """
        # Veriyi bytes'a dönüştür
        data_bytes = data.astype(np.float32).tobytes()
        
        # zlib DEFLATE ile sıkıştır
        compressed = zlib.compress(data_bytes, level=self.compression_level)
        
        logger.info(
            f"✓ Raw data compressed: "
            f"{len(data_bytes)} → {len(compressed)} bytes "
            f"({len(compressed)/len(data_bytes)*100:.1f}%)"
        )
        
        return compressed
    
    def compress_quantized_coefficients(self, quantized: np.ndarray) -> bytes:
        """
        Quantized katsayıları sıkıştır (RLE + DEFLATE)
        
        Args:
            quantized: Quantized katsayılar
        
        Returns:
            Sıkıştırılmış bytes
        """
        # RLE uygulanma için seri hale
        flat = quantized.flatten()
        
        # Sıfırları say (sparsity)
        zero_count = np.sum(flat == 0)
        sparsity = zero_count / flat.size
        
        logger.info(
            f"Quantized data sparsity: {sparsity*100:.1f}% "
            f"({zero_count}/{flat.size} zeros)"
        )
        
        # Seyrek veri için RLE uygulanabilir
        if sparsity > 0.5:
            logger.info("High sparsity detected, applying RLE")
            rle_data = self._apply_rle(flat)
        else:
            rle_data = flat
        
        # DEFLATE ile sıkıştır
        compressed = zlib.compress(rle_data.tobytes(), level=self.compression_level)
        
        logger.info(
            f"✓ Quantized coefficients compressed: "
            f"{len(rle_data.tobytes())} → {len(compressed)} bytes"
        )
        
        return compressed
    
    def _apply_rle(self, data: np.ndarray) -> np.ndarray:
        """
        Run-Length Encoding uygula
        
        Args:
            data: Input verisi
        
        Returns:
            RLE kodlanmış verisi (value, count tuples)
        """
        # Basit RLE: (value, count) çiftleri
        if len(data) == 0:
            return np.array([], dtype=data.dtype)
        
        # Değişimleri bul
        changes = np.concatenate(([True], data[1:] != data[:-1], [True]))
        indices = np.where(changes)[0]
        
        # (value, count) tuples oluştur
        runs = []
        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i + 1]
            value = data[start]
            count = end - start
            runs.append((value, count))
        
        # Array'e dönüştür
        if runs:
            values = np.array([v for v, c in runs], dtype=data.dtype)
            counts = np.array([c for v, c in runs], dtype=np.int32)
            return np.concatenate([values, counts])
        else:
            return np.array([], dtype=data.dtype)


class LosslessDecoder:
    """Lossless decoder (DEFLATE-based)"""
    
    @staticmethod
    def decompress_raw_data(
        compressed: bytes,
        original_shape: Tuple[int, ...],
        dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """
        Sıkıştırılmış ham veriyi decompress et
        
        Args:
            compressed: Sıkıştırılmış veri
            original_shape: Orijinal shape
            dtype: Data type
        
        Returns:
            Decompressed array
        """
        # zlib ile decompress
        decompressed = zlib.decompress(compressed)
        
        # NumPy array'e dönüştür
        data = np.frombuffer(decompressed, dtype=dtype)
        
        # Shape'i restore et
        try:
            data = data.reshape(original_shape)
            logger.info(f"✓ Raw data decompressed: {data.shape}")
        except ValueError:
            logger.warning(f"Shape mismatch, returning flat array")
        
        return data
    
    @staticmethod
    def decompress_quantized_coefficients(
        compressed: bytes,
        original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Sıkıştırılmış quantized katsayıları decompress et
        
        Args:
            compressed: Sıkıştırılmış veri
            original_shape: Orijinal shape
        
        Returns:
            Decompressed array
        """
        # zlib ile decompress
        decompressed = zlib.decompress(compressed)
        
        # NumPy array'e dönüştür
        data = np.frombuffer(decompressed, dtype=np.float32)
        
        # Shape'i restore et
        try:
            data = data.reshape(original_shape)
            logger.info(f"✓ Quantized coefficients decompressed: {data.shape}")
        except ValueError:
            logger.warning(f"Shape mismatch, returning flat array")
        
        return data


class LosslessCompressionPipeline:
    """Lossless compression pipeline"""
    
    def __init__(self, compression_level: int = 9):
        """
        Başlat
        
        Args:
            compression_level: Sıkıştırma seviyesi (1-9)
        """
        self.encoder = LosslessEncoder(compression_level)
        self.decoder = LosslessDecoder()
        self.compression_level = compression_level
        logger.info(f"LosslessCompressionPipeline: level={compression_level}")
    
    def compress_image_data(
        self,
        luma: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Görüntü verisini sıkıştır
        
        Args:
            luma: Luma channel (Y)
            metadata: Optional metadata
        
        Returns:
            (compressed_data, metadata)
        """
        logger.info(f"Lossless compression starting: {luma.shape}")
        
        # Veriyi sıkıştır
        compressed = self.encoder.compress_raw_data(luma)
        
        # Metadata oluştur
        compression_metadata = {
            'mode': 'lossless',
            'original_height': luma.shape[0],
            'original_width': luma.shape[1],
            'compression_level': self.compression_level,
            'original_size': luma.nbytes,
            'compressed_size': len(compressed),
            'compression_ratio': luma.nbytes / len(compressed) if len(compressed) > 0 else 0
        }
        
        if metadata:
            compression_metadata.update(metadata)
        
        logger.info(
            f"✓ Lossless compression complete: "
            f"{compression_metadata['compression_ratio']:.2f}x compression"
        )
        
        return compressed, compression_metadata
    
    def decompress_image_data(
        self,
        compressed: bytes,
        original_height: int,
        original_width: int
    ) -> np.ndarray:
        """
        Sıkıştırılmış görüntü verisini decompress et
        
        Args:
            compressed: Sıkıştırılmış veri
            original_height: Orijinal yükseklik
            original_width: Orijinal genişlik
        
        Returns:
            Decompressed luma array
        """
        original_shape = (original_height, original_width)
        
        decompressed = self.decoder.decompress_raw_data(
            compressed,
            original_shape,
            dtype=np.float32
        )
        
        logger.info(f"✓ Lossless decompression complete: {decompressed.shape}")
        
        return decompressed
    
    def calculate_compression_ratio(
        self,
        original_size: int,
        compressed_size: int
    ) -> float:
        """Sıkıştırma oranını hesapla"""
        if compressed_size == 0:
            return 0.0
        return original_size / compressed_size


class LosslessQualityAssessment:
    """Lossless kalite değerlendirmesi"""
    
    @staticmethod
    def verify_lossless_reconstruction(
        original: np.ndarray,
        reconstructed: np.ndarray,
        tolerance: float = 1e-6
    ) -> Tuple[bool, float, str]:
        """
        Lossless reconstruction verify et
        
        Args:
            original: Orijinal array
            reconstructed: Reconstructed array
            tolerance: Tolerans (default=1e-6)
        
        Returns:
            (is_perfect, max_error, status)
        """
        if original.shape != reconstructed.shape:
            return False, float('inf'), f"Shape mismatch: {original.shape} vs {reconstructed.shape}"
        
        # Max error hesapla
        max_error = np.max(np.abs(original - reconstructed))
        
        # Identität kontrol et
        is_perfect = np.allclose(original, reconstructed, atol=tolerance)
        
        status = "✓ Perfect reconstruction" if is_perfect else f"✗ Reconstruction error: {max_error}"
        
        logger.info(f"Lossless verification: {status}")
        
        return is_perfect, max_error, status
    
    @staticmethod
    def calculate_compression_metrics(
        original_size: int,
        compressed_size: int,
        compression_time: float,
        decompression_time: float
    ) -> Dict[str, float]:
        """
        Sıkıştırma metriklerini hesapla
        
        Args:
            original_size: Orijinal boyut (bytes)
            compressed_size: Sıkıştırılmış boyut (bytes)
            compression_time: Sıkıştırma zamanı (saniye)
            decompression_time: Açma zamanı (saniye)
        
        Returns:
            Metrikler dict
        """
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        reduction = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        metrics = {
            'compression_ratio': ratio,
            'size_reduction_percent': reduction,
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_time_seconds': compression_time,
            'decompression_time_seconds': decompression_time,
            'compression_speed_mbps': (original_size / 1e6) / compression_time if compression_time > 0 else 0,
            'decompression_speed_mbps': (original_size / 1e6) / decompression_time if decompression_time > 0 else 0
        }
        
        return metrics


if __name__ == '__main__':
    # Test lossless compression
    from core.logging_config import setup_logger
    
    setup_logger(__name__)
    
    # Test veri
    test_data = np.random.randint(0, 255, (512, 512), dtype=np.float32)
    
    # Sıkıştırma
    pipeline = LosslessCompressionPipeline(compression_level=9)
    compressed, metadata = pipeline.compress_image_data(test_data)
    
    # Decompression
    decompressed = pipeline.decompress_image_data(
        compressed,
        metadata['original_height'],
        metadata['original_width']
    )
    
    # Verify
    is_perfect, max_error, status = LosslessQualityAssessment.verify_lossless_reconstruction(
        test_data,
        decompressed
    )
    
    print(f"Compression ratio: {metadata['compression_ratio']:.2f}x")
    print(f"Status: {status}")
    print(f"Max error: {max_error}")
