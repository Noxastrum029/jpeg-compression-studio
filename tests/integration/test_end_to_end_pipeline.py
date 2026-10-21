"""
Uçtan uca (ince) entegrasyon: DWT → kuantalama → Huffman → decode → ters DWT.

GUI veya komut satırı çalıştırma şeklini değiştirmez; yalnızca pytest ile doğrulanır.
"""

import numpy as np
import pywt
import pytest

from huffman_codec import decode_coeffs, encode_coeffs
from stages.stage1_preprocessing import DWTTransformer
from stages.stage2_quantization import QuantizationModule


@pytest.mark.integration
def test_dwt_quantize_huffman_decode_inverse_dwt() -> None:
    wavelet = "db4"
    level = 2
    image = np.linspace(0, 255, 64 * 64, dtype=np.float64).reshape(64, 64)

    dwt = DWTTransformer(wavelet_name=wavelet, level=level)
    coeffs = dwt.apply_dwt2d(image)

    quantizer = QuantizationModule(method="scalar")
    q_coeffs = quantizer.quantize_by_quality(coeffs, quality=50)

    bitstream, _ = encode_coeffs(q_coeffs)
    assert len(bitstream) > 0

    decoded = decode_coeffs(bitstream, wavelet=wavelet, level=level)

    # Huffman yolu int16 yuvarlaması kullanır; decode çıktısı yuvarlanmış katsayılarla örtüşmeli.
    np.testing.assert_allclose(
        decoded[0], np.round(q_coeffs[0]).astype(np.float64), rtol=0, atol=0
    )
    for dec_level, q_level in zip(decoded[1:], q_coeffs[1:]):
        for a, b in zip(dec_level, q_level):
            np.testing.assert_allclose(
                a, np.round(b).astype(np.float64), rtol=0, atol=0
            )

    recon = pywt.waverec2(decoded, wavelet, mode="periodization")
    recon = recon[: image.shape[0], : image.shape[1]]
    assert recon.shape == image.shape
    assert np.all(np.isfinite(recon))
