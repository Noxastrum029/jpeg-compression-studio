"""
Orthonormal 2D DCT-II helpers.

SciPy (scipy.fftpack) is preferred. If SciPy is not installed, OpenCV 8×8 block
DCT is used so imports succeed and the GUI (tür testi, DCT pipeline) keeps working.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

_dct2_ortho: Callable[[np.ndarray], np.ndarray]
_idct2_ortho: Callable[[np.ndarray], np.ndarray]

try:
    from scipy.fftpack import dct as _dct1
    from scipy.fftpack import idct as _idct1

    def _dct2_ortho_scipy(x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float64)
        return _dct1(_dct1(a, axis=0, norm="ortho"), axis=1, norm="ortho")

    def _idct2_ortho_scipy(x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float64)
        return _idct1(_idct1(a, axis=0, norm="ortho"), axis=1, norm="ortho")

    _dct2_ortho = _dct2_ortho_scipy
    _idct2_ortho = _idct2_ortho_scipy
except ImportError:
    import cv2

    def _dct2_idct2_blocks(
        x: np.ndarray,
        block_size: int,
        forward: bool,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        h, w = x.shape
        ph = ((h + block_size - 1) // block_size) * block_size
        pw = ((w + block_size - 1) // block_size) * block_size
        padded = np.zeros((ph, pw), dtype=np.float32)
        padded[:h, :w] = x
        out = np.zeros_like(padded)
        flags = 0 if forward else cv2.DCT_INVERSE
        for i in range(0, ph, block_size):
            for j in range(0, pw, block_size):
                blk = padded[i : i + block_size, j : j + block_size]
                out[i : i + block_size, j : j + block_size] = cv2.dct(blk, flags=flags)
        return out[:h, :w].astype(np.float64)

    def _dct2_ortho_opencv(x: np.ndarray) -> np.ndarray:
        return _dct2_idct2_blocks(x, 8, forward=True)

    def _idct2_ortho_opencv(x: np.ndarray) -> np.ndarray:
        return _dct2_idct2_blocks(x, 8, forward=False)

    _dct2_ortho = _dct2_ortho_opencv
    _idct2_ortho = _idct2_ortho_opencv
    logger.warning(
        "SciPy bulunamadı; DCT için OpenCV 8×8 blok yolu kullanılıyor. "
        "Öneri: pip install scipy"
    )


def dct2_ortho(x: np.ndarray) -> np.ndarray:
    """Full-image 2D orthonormal DCT-II (rate_distortion fallback)."""
    return _dct2_ortho(x)


def idct2_ortho(x: np.ndarray) -> np.ndarray:
    """Inverse of dct2_ortho."""
    return _idct2_ortho(x)


def dct2_ortho_block(block: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Single block DCT (JPEG-style)."""
    b = np.asarray(block, dtype=np.float64)
    if b.shape != (block_size, block_size):
        raise ValueError(f"Beklenen ({block_size},{block_size}), gelen {b.shape}")
    try:
        from scipy.fftpack import dct as _dct1

        return _dct1(_dct1(b, axis=0, norm="ortho"), axis=1, norm="ortho")
    except ImportError:
        import cv2

        return cv2.dct(b.astype(np.float32)).astype(np.float64)


def idct2_ortho_block(block: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Single block inverse DCT."""
    b = np.asarray(block, dtype=np.float64)
    if b.shape != (block_size, block_size):
        raise ValueError(f"Beklenen ({block_size},{block_size}), gelen {b.shape}")
    try:
        from scipy.fftpack import idct as _idct1

        return _idct1(_idct1(b, axis=0, norm="ortho"), axis=1, norm="ortho")
    except ImportError:
        import cv2

        return cv2.dct(b.astype(np.float32), flags=cv2.DCT_INVERSE).astype(np.float64)
