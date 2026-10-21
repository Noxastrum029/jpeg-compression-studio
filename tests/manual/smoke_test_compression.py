#!/usr/bin/env python3
"""Test compression GUI functions without GUI"""

import numpy as np
import cv2
import sys

print("🧪 Sıkıştırma sistemini test et...")

# Test görüntüsü oluştur
print("\n1️⃣  Test görüntüsü oluştur...")
test_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
cv2.imwrite("test_image.jpg", test_img)
print("   ✓ test_image.jpg oluşturuldu")

# Module import kontrol et
print("\n2️⃣  Modülleri import et...")
try:
    from jpeg_jpeg2000_stage1 import ImagePreprocessor, DWTTransformer
    from huffman_codec import huffman_compressed_size
    print("   ✓ Core modülleri yüklendi")
except Exception as e:
    print(f"   ✗ Hata: {e}")
    sys.exit(1)

# Görüntü yükleme testi
print("\n3️⃣  Görüntü yükleme testi...")
try:
    pre = ImagePreprocessor("test_image.jpg")
    pre.load_image()
    pre.convert_to_luma(force_grayscale=True)
    luma = pre.luma_channel
    print(f"   ✓ Görüntü yüklendi: {luma.shape}, dtype={luma.dtype}")
except Exception as e:
    print(f"   ✗ Hata: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# DWT testi
print("\n4️⃣  DWT dönüşümü testi...")
try:
    dwt = DWTTransformer(wavelet_name="db4", level=2)
    coeffs = dwt.apply_dwt2d(luma.astype(np.float64))
    print(f"   ✓ DWT başarılı: {type(coeffs)}")
except Exception as e:
    print(f"   ✗ Hata: {e}")
    import traceback
    traceback.print_exc()

# Kuantizasyon testi
print("\n5️⃣  Kuantizasyon testi...")
try:
    from compression_gui import quantize_coeffs
    q_coeffs = quantize_coeffs(coeffs, 5.0)
    print(f"   ✓ Kuantizasyon başarılı")
except Exception as e:
    print(f"   ✗ Hata: {e}")
    import traceback
    traceback.print_exc()

# Huffman testi
print("\n6️⃣  Huffman kodlaması testi...")
try:
    raw_bytes, huff_bytes, saving = huffman_compressed_size(q_coeffs)
    print(f"   ✓ Huffman: {raw_bytes} → {huff_bytes} bytes (↓{saving:.1f}%)")
except Exception as e:
    print(f"   ✗ Hata: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Tüm testler tamamlandı!")
