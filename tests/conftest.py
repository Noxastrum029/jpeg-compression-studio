import numpy as np
import pytest

from stages.stage4_bitstream import CompressionMetadata


@pytest.fixture
def sample_metadata():
    return CompressionMetadata(
        image_width=64,
        image_height=64,
        dwt_level=2,
        wavelet="db4",
        quantization_method="uniform",
        quality_factor=50,
    )


@pytest.fixture
def sample_bitstream():
    # 8'e tam bolunen uzunluk, paketleme testleri icin ideal.
    return "01001101" * 16


@pytest.fixture
def sample_code_table():
    return {0: "0", 1: "10", 2: "110", 3: "111"}


@pytest.fixture
def sample_images():
    original = np.full((16, 16), 128, dtype=np.uint8)
    reconstructed = original.copy()
    reconstructed[0, 0] = 120
    return original, reconstructed
