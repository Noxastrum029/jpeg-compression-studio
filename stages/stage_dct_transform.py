"""
stage_dct_transform.py — Discrete Cosine Transform (DCT) Implementation

JPEG-style 8x8 block-based DCT transformation module.

DCT (Discrete Cosine Transform):
- Used in JPEG standard (block-based image compression)
- Converts spatial domain to frequency domain
- Concentrates energy in low-frequency coefficients
- Better for natural images with sharp edges
- Comparison point with DWT for compression analysis

Multi-level support:
- Single-level: Full image 8x8 blocks
- Can be applied recursively (though JPEG uses single-level)
"""

import numpy as np
import logging
from typing import Tuple

from core.dct_utils import dct2_ortho_block, idct2_ortho_block
from core.logging_config import setup_logger

logger = setup_logger("dct", "dct.log")


class DCTTransformer:
    """
    DCT Transform - JPEG-style block-based compression.
    
    Implements 2D DCT on 8x8 blocks (JPEG standard).
    Suitable for comparison with DWT methods.
    
    Attributes:
        block_size: Size of blocks (default 8, JPEG standard)
        blocks: Generated DCT blocks
    """
    
    def __init__(self, block_size: int = 8):
        """
        Initialize DCT Transformer.
        
        Args:
            block_size: Block size for DCT (8 for JPEG)
        """
        self.block_size = block_size
        self.blocks: list = []
        
        logger.info(f"DCTTransformer initialized: block_size={block_size}")
    
    def apply_dct_2d(self, image: np.ndarray) -> np.ndarray:
        """
        Apply 2D DCT to image using 8x8 blocks.
        
        Args:
            image: Input image (2D array, grayscale)
            
        Returns:
            DCT coefficients (same size as input)
            
        Raises:
            ValueError: Invalid image dimensions
        """
        if len(image.shape) != 2:
            raise ValueError(f"Expected 2D image, got {image.shape}")
        
        height, width = image.shape
        
        logger.info(f"Applying DCT: image_shape={image.shape}, "
                   f"block_size={self.block_size}")
        
        # Pad image to multiple of block_size
        padded_height = ((height + self.block_size - 1) // self.block_size) * self.block_size
        padded_width = ((width + self.block_size - 1) // self.block_size) * self.block_size
        
        padded_image = np.zeros((padded_height, padded_width), dtype=np.float32)
        padded_image[:height, :width] = image.astype(np.float32)
        
        logger.info(f"Image padded: {image.shape} → {padded_image.shape}")
        
        # Process blocks
        dct_result = np.zeros_like(padded_image)
        self.blocks = []
        
        for i in range(0, padded_height, self.block_size):
            for j in range(0, padded_width, self.block_size):
                # Extract block
                block = padded_image[i:i+self.block_size, j:j+self.block_size]
                
                # Apply 2D DCT (Type-II DCT, orthonormal; SciPy veya OpenCV)
                dct_block = dct2_ortho_block(block, self.block_size)
                
                dct_result[i:i+self.block_size, j:j+self.block_size] = dct_block
                self.blocks.append(dct_block)
        
        logger.info(f"✓ DCT applied: {len(self.blocks)} blocks processed")
        
        return dct_result
    
    def get_dct_visualization(self, dct_coeffs: np.ndarray) -> np.ndarray:
        """
        Prepare DCT coefficients for visualization.
        
        Args:
            dct_coeffs: DCT coefficient matrix
            
        Returns:
            Normalized coefficients [0, 1] for display
        """
        # Use log scale for better visualization (DCT has large dynamic range)
        # Prevent log(0) by adding small epsilon
        dct_magnitude = np.abs(dct_coeffs) + 1e-10
        log_magnitude = np.log(dct_magnitude)
        
        # Normalize to [0, 1]
        min_val = log_magnitude.min()
        max_val = log_magnitude.max()
        
        if max_val > min_val:
            normalized = (log_magnitude - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(log_magnitude)
        
        return normalized


class InverseDCT:
    """
    Inverse DCT Transform - Reconstruct image from DCT coefficients.
    
    Implements 2D IDCT using block processing.
    Reverses the forward DCT transformation.
    
    Attributes:
        block_size: Size of blocks (must match forward DCT)
    """
    
    def __init__(self, block_size: int = 8):
        """
        Initialize Inverse DCT Transformer.
        
        Args:
            block_size: Block size (must match forward DCT)
        """
        self.block_size = block_size
        logger.info(f"InverseDCT initialized: block_size={block_size}")
    
    def apply_idct_2d(self, dct_coeffs: np.ndarray, 
                      original_shape: Tuple[int, int] = None) -> np.ndarray:
        """
        Apply inverse 2D DCT to reconstruct image.
        
        Args:
            dct_coeffs: DCT coefficients (output from DCTTransformer)
            original_shape: Original image shape (for removing padding)
            
        Returns:
            Reconstructed image
            
        Raises:
            ValueError: Invalid DCT coefficient dimensions
        """
        if len(dct_coeffs.shape) != 2:
            raise ValueError(f"Expected 2D DCT coefficients, got {dct_coeffs.shape}")
        
        height, width = dct_coeffs.shape
        
        logger.info(f"Applying IDCT: shape={dct_coeffs.shape}, "
                   f"block_size={self.block_size}")
        
        # Process blocks with inverse DCT
        reconstructed = np.zeros_like(dct_coeffs, dtype=np.float32)
        
        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                # Extract DCT block
                dct_block = dct_coeffs[i:i+self.block_size, j:j+self.block_size]
                
                idct_block = idct2_ortho_block(dct_block, self.block_size)
                
                reconstructed[i:i+self.block_size, j:j+self.block_size] = idct_block
        
        logger.info(f"✓ IDCT applied: Image reconstructed")
        
        # Remove padding if original shape provided
        if original_shape is not None:
            orig_height, orig_width = original_shape
            reconstructed = reconstructed[:orig_height, :orig_width]
            logger.info(f"Padding removed: {dct_coeffs.shape} → {reconstructed.shape}")
        
        return reconstructed
    
    def apply_idct_2d_fast(self, dct_coeffs_padded: np.ndarray,
                           original_height: int,
                           original_width: int) -> np.ndarray:
        """
        Fast IDCT with automatic padding removal.
        
        Args:
            dct_coeffs_padded: Padded DCT coefficients
            original_height: Original image height
            original_width: Original image width
            
        Returns:
            Reconstructed image (original size)
        """
        reconstructed = self.apply_idct_2d(dct_coeffs_padded)
        return reconstructed[:original_height, :original_width]


class DCTCompressionPipeline:
    """
    Complete DCT-based compression pipeline.
    
    Combines DCT transformation with quantization.
    Similar structure to DWT pipeline for consistency.
    """
    
    def __init__(self, quality: int = 85, block_size: int = 8):
        """
        Initialize DCT compression pipeline.
        
        Args:
            quality: Quality factor (0-100)
            block_size: Block size for DCT
        """
        self.quality = quality
        self.block_size = block_size
        self.transformer = DCTTransformer(block_size)
        self.reconstructor = InverseDCT(block_size)
        
        logger.info(f"DCTCompressionPipeline: quality={quality}, block_size={block_size}")
    
    def get_quantization_matrix(self) -> np.ndarray:
        """
        Get JPEG-style quantization matrix.
        
        Returns:
            8x8 quantization matrix based on quality factor
            
        Note:
            JPEG uses luminance quantization matrix.
            Quality 85 maps to JPEG quality level.
        """
        # Standard JPEG luminance quantization matrix
        # (scaled for quality factor)
        jpeg_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)
        
        # Scale by quality factor
        # Quality 50: scale factor = 1.0
        # Quality > 50: scale down (less quantization, higher quality)
        # Quality < 50: scale up (more quantization, lower quality)
        if self.quality >= 50:
            scale = (100 - self.quality) / 50.0
        else:
            scale = 50.0 / self.quality
        
        quantization_matrix = np.round(jpeg_matrix * scale).astype(np.float32)
        quantization_matrix = np.clip(quantization_matrix, 1, 255)
        
        return quantization_matrix
    
    def quantize_dct(self, dct_coeffs: np.ndarray) -> np.ndarray:
        """
        Apply JPEG-style quantization to DCT coefficients.
        
        Args:
            dct_coeffs: DCT coefficients
            
        Returns:
            Quantized coefficients
        """
        q_matrix = self.get_quantization_matrix()
        
        quantized = np.zeros_like(dct_coeffs, dtype=np.int16)
        
        height, width = dct_coeffs.shape
        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                # Get block
                block = dct_coeffs[i:i+self.block_size, j:j+self.block_size]
                
                # Quantize: round(block / q_matrix) * q_matrix
                if block.shape == (self.block_size, self.block_size):
                    quantized_block = np.round(block / q_matrix) * q_matrix
                    quantized[i:i+self.block_size, j:j+self.block_size] = quantized_block.astype(np.int16)
        
        logger.info(f"✓ DCT quantization complete: quality={self.quality}")
        
        return quantized
    
    def dequantize_dct(self, quantized_coeffs: np.ndarray) -> np.ndarray:
        """
        Dequantize DCT coefficients (inverse of quantization).
        
        Args:
            quantized_coeffs: Quantized DCT coefficients
            
        Returns:
            Dequantized coefficients
        """
        # For DCT with JPEG-style quantization, dequantization
        # is simply converting back to float (no scaling needed)
        return quantized_coeffs.astype(np.float32)
