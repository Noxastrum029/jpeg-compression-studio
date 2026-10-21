#!/usr/bin/env python3
"""Compression GUI - Başlatma testi"""

import sys
print("Python:", sys.version)

# Gerekli modülleri kontrol et
required = [
    ("cv2", "OpenCV"),
    ("customtkinter", "CustomTkinter"),
    ("numpy", "NumPy"),
    ("pywt", "PyWavelets"),
    ("matplotlib", "Matplotlib"),
    ("PIL", "Pillow"),
]

print("\n✓ Modül kontrolleri:")
for module, name in required:
    try:
        __import__(module)
        print(f"  ✓ {name} ({module})")
    except ImportError as e:
        print(f"  ✗ {name} ({module}): {e}")

# Yerel modülleri kontrol et
print("\n✓ Yerel modüller:")
try:
    from jpeg_jpeg2000_stage1 import DWTTransformer, ImagePreprocessor
    print("  ✓ jpeg_jpeg2000_stage1")
except Exception as e:
    print(f"  ✗ jpeg_jpeg2000_stage1: {e}")

try:
    from huffman_codec import huffman_compressed_size
    print("  ✓ huffman_codec")
except Exception as e:
    print(f"  ✗ huffman_codec: {e}")

try:
    from core.image_type_comparison import ImageTypeComparisonRunner
    print("  ✓ core.image_type_comparison")
except Exception as e:
    print(f"  ✗ core.image_type_comparison: {e}")

try:
    from core.rate_distortion import RateDistortionAnalyzer
    print("  ✓ core.rate_distortion")
except Exception as e:
    print(f"  ✗ core.rate_distortion: {e}")

print("\n✓ GUI başlatılıyor...")
try:
    from compression_gui import CompressionApp
    print("  ✓ CompressionApp import edildi")

    app = CompressionApp()
    print("  ✓ GUI penceresine oluşturuldu")
    print("\n✅ GUI başarıyla başlatılıyor!")
    print("Görüntü yüklemek için sol tarafta 'Görüntü Yükle' butonuna tıklayın.")

    app.run()
except Exception as e:
    import traceback
    print(f"\n❌ HATA: {e}")
    traceback.print_exc()
    sys.exit(1)
