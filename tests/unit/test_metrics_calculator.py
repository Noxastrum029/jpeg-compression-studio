import numpy as np
import pytest

from stages.stage6_metrics import MetricsCalculator


def test_metrics_with_identical_images_returns_zero_mse_and_infinite_psnr():
    calculator = MetricsCalculator()
    image = np.full((8, 8), 100, dtype=np.uint8)

    metrics = calculator.calculate_all_metrics(
        original_image=image,
        reconstructed_image=image,
        original_file_size=1024,
        compressed_file_size=256,
    )

    assert metrics.mse == 0.0
    assert np.isinf(metrics.psnr)
    assert metrics.compression_ratio == 4.0
    assert metrics.image_pixels == 64


def test_metrics_with_small_difference_produces_positive_mse(sample_images):
    calculator = MetricsCalculator()
    original, reconstructed = sample_images

    metrics = calculator.calculate_all_metrics(
        original_image=original,
        reconstructed_image=reconstructed,
        original_file_size=2048,
        compressed_file_size=512,
        compression_time=0.01,
        decompression_time=0.02,
    )

    assert metrics.mse > 0
    assert metrics.psnr < float("inf")
    assert 0 <= metrics.ssim_value <= 1
    assert metrics.total_time == pytest.approx(0.03)
