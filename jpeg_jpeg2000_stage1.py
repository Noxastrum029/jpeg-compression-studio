"""
jpeg_jpeg2000_stage1.py

Bu dosya, JPEG/JPEG2000 benzeri görüntü sıkıştırma sisteminin
1. aşamasını (Input Handling + Source Encoding / DWT) gerçekleştirir.

Ana özellikler:
- PNG / BMP / TIFF vb. formatlardan görüntü okuma
- 512x512 veya 1024x1024 boyutlandırma
- RGB -> YCbCr dönüştürme, sadece Y (parlaklık) kanalı ile çalışma
- Çok seviyeli (multi-level) Haar veya Daubechies (db2, db4, ...) DWT
- DWT alt bantlarını (LL, LH, HL, HH) çoklu kare halinde görselleştirme
- Çıktı: Kuantalama aşamasına gidecek DWT katsayıları (coeffs)
"""

import os
from typing import Tuple, List, Dict, cast

import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

from core.pr_qmf import is_pr_qmf, wavedec2_prqmf, waverec2_prqmf, max_level_prqmf


class ImagePreprocessor:
    """
    Giriş görüntüsünü okuma, boyutlandırma ve uygun renk uzayına dönüştürme
    işlemlerini kapsayan sınıf.

    Genel kullanım:
    - Renkli görüntülerde, YCbCr renk uzayına dönüp sadece Y (parlaklık)
      kanalı ile çalışmak yaygın JPEG / JPEG2000 yaklaşımıdır.
    - Parmak izi / tıbbi gibi tek kanallı (grayscale) görüntüler için
      doğrudan gri seviye ile çalışmak mümkündür.
    """

    def __init__(self, image_path: str) -> None:
        """
        image_path:
            Giriş görüntüsünün dosya yolu.
        """
        self.image_path: str = image_path
        self.original_bgr: np.ndarray | None = None
        self.luma_channel: np.ndarray | None = None

    def load_image(self) -> None:
        """
        Görüntüyü BGR formatında okur. Türkçe karakter içeren yollar (Ü, ü, ğ, ş vb.)
        Windows'ta OpenCV'nin cv2.imread ile sorun çıkardığı için dosya Python ile
        binary okunup cv2.imdecode ile açılır; böylece Unicode yol sorunu olmaz.
        """
        if not os.path.isfile(self.image_path):
            raise ValueError(f"Görüntü dosyası bulunamadı: {self.image_path}")

        with open(self.image_path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Görüntü okunamadı (desteklenmeyen format?): {self.image_path}")

        self.original_bgr = img

    def convert_to_luma(
        self,
        target_size: Tuple[int, int] | None = None,
        force_grayscale: bool = False,
    ) -> np.ndarray:
        """
        Orijinal BGR görüntüyü, DWT aşamasında kullanılacak verilere dönüştürür.
        
        Parametreler:
        - target_size:
            Görüntünün yeniden boyutlandırılacağı (genişlik, yükseklik) boyutu. None ise yeniden boyutlandırılmaz.
        - force_grayscale:
            True  -> BGR görüntü gri tonlamaya dönüştürülür (parmak izi / medikal / Lossless).
            False -> BGR görüntü iken renkli görüntülere uygun Y channel çıkarılır.
                    (NOT: Renkli görüntülerin RGB/BGR 3-channel korunması tercihli)

        Dönüş:
        - Grayscale: 2B numpy dizisi (float32) [0, 255]
        - Renkli: 3B numpy dizisi BGR (float32) [0, 255]
        """
        if self.original_bgr is None:
            raise RuntimeError("Görüntü henüz yüklenmedi. Önce load_image() çağrılmalı.")

        if target_size is not None:
            img_to_convert = cv2.resize(self.original_bgr, target_size, interpolation=cv2.INTER_AREA)
        else:
            img_to_convert = self.original_bgr

        if force_grayscale:
            # Tıbbi/Parmak izi: gri seviye
            # Eğer zaten grayscale ise direkt kullan
            if len(img_to_convert.shape) == 2 or (len(img_to_convert.shape) == 3 and img_to_convert.shape[2] == 1):
                # Zaten grayscale
                gray = img_to_convert.squeeze() if len(img_to_convert.shape) == 3 else img_to_convert
            else:
                # BGR/RGB to grayscale dönüşümü
                gray = cv2.cvtColor(img_to_convert, cv2.COLOR_BGR2GRAY)
            self.luma_channel = gray.astype(np.float32)
        else:
            # Renkli görüntüler: 3-channel BGR kalsın (renk bilgisi korunur)
            # Lossless için Y channel tercihli, ancak JPEG/JPEG2000 için BGR 3-channel uygun
            if len(img_to_convert.shape) == 3 and img_to_convert.shape[2] == 3:
                # Renkli görüntü: BGR kalsın
                self.luma_channel = img_to_convert.astype(np.float32)
            else:
                # Grayscale görüntü: Y channel
                self.luma_channel = img_to_convert.astype(np.float32)

        return self.luma_channel


class DWTTransformer:
    """
    2B DWT (Discrete Wavelet Transform) ve görselleştirme işlemlerini kapsayan sınıf.

    Bu sınıf, ileride kuantalama ve Huffman aşamalarının doğrudan kullanacağı
    katsayıları (coeffs) üretir.
    """

    def __init__(self, wavelet_name: str = "haar", level: int = 2) -> None:
        """
        wavelet_name:
            "haar", "db2", "db4", "bior4.4" vb. PyWavelets dalgacık isimleri.
        level:
            Çok seviyeli ayrıştırma derecesi (1, 2, 3, ...).
        """
        self.wavelet_name: str = wavelet_name
        self.level: int = level

    def apply_dwt2d(self, image_2d: np.ndarray) -> List:
        """
        2B veya 3B DWT uygular ve pywt.wavedec2'nin döndürdüğü katsayı yapısını verir.

        image_2d:
            2B numpy dizisi (ör. Y kanalı) veya 3B (BGR) görüntü, float32/float64.
            3B ise: her channel'a ayrı DWT uygulanır.

        level=2 için tek kanal çıktı yapısı:
            coeffs[0] -> cA2 (LL2)
            coeffs[1] -> (cH2, cV2, cD2) -> (HL2, LH2, HH2)
            coeffs[2] -> (cH1, cV1, cD1) -> (HL1, LH1, HH1)
            
        3B (BGR) input için:
            coeffs = {
                'B': [...],  # Blue channel coeffs
                'G': [...],  # Green channel coeffs
                'R': [...]   # Red channel coeffs
            }
        """
        # 3-channel support
        if image_2d.ndim == 3 and image_2d.shape[2] == 3:
            # 3B (BGR) görüntü: her channel'a ayrı DWT
            channels = ['B', 'G', 'R']
            coeffs_dict = {}
            for ch_idx, ch_name in enumerate(channels):
                ch_data = image_2d[:, :, ch_idx]
                if is_pr_qmf(self.wavelet_name):
                    coeffs_dict[ch_name] = wavedec2_prqmf(
                        ch_data, self.wavelet_name, level=self.level
                    )
                else:
                    coeffs_dict[ch_name] = pywt.wavedec2(
                        data=ch_data,
                        wavelet=self.wavelet_name,
                        level=self.level,
                        mode="periodization",
                    )
            # Renkli (3 kanal) durumda kanal bazlı katsayı sözlüğü döner.
            # Çağıranlar tek kanal (2B) için List bekler; tip uyumu için cast edilir.
            return cast(List, coeffs_dict)
        
        elif image_2d.ndim == 2:
            # 2B (Y channel / Grayscale)
            if is_pr_qmf(self.wavelet_name):
                coeffs = wavedec2_prqmf(
                    image_2d, self.wavelet_name, level=self.level
                )
            else:
                coeffs = pywt.wavedec2(
                    data=image_2d,
                    wavelet=self.wavelet_name,
                    level=self.level,
                    mode="periodization",  # JPEG2000'e benzeyen sınır koşulu
                )
            return coeffs
        
        else:
            raise ValueError("DWT için giriş 2B (tek kanallı) veya 3B (BGR) bir görüntü olmalıdır.")

    @staticmethod
    def _normalize_for_display(band: np.ndarray) -> np.ndarray:
        """
        Bir alt bandı görselleştirebilmek için [0, 1] aralığına normalleştirir.
        Bu fonksiyon sadece çizim için kullanılır.
        """
        band = band.astype(np.float32)
        min_val = band.min()
        max_val = band.max()

        if max_val - min_val < 1e-9:
            return np.zeros_like(band, dtype=np.float32)

        norm = (band - min_val) / (max_val - min_val)
        return norm

    def visualize_multilevel_coeffs(
        self,
        coeffs: List,
        show: bool = True,
        save_path: str | None = None,
        figure_title: str = "DWT Alt Bantları",
    ) -> None:
        """
        Çok seviyeli DWT katsayılarını 4–8 kare halinde görselleştirir.

        level = 1 için:
            LL1, LH1, HL1, HH1
        level = 2 için:
            LL2, HL2, LH2, HH2, HL1, LH1, HH1 (toplam 7 görüntü)
        """
        cA = coeffs[0]
        detail_coeffs = coeffs[1:]

        images: List[np.ndarray] = []
        titles: List[str] = []

        images.append(self._normalize_for_display(cA))
        titles.append(f"LL (Seviye {self.level})")

        current_level = self.level
        for (cH, cV, cD) in detail_coeffs:
            images.append(self._normalize_for_display(cH))
            titles.append(f"HL (Seviye {current_level})")

            images.append(self._normalize_for_display(cV))
            titles.append(f"LH (Seviye {current_level})")

            images.append(self._normalize_for_display(cD))
            titles.append(f"HH (Seviye {current_level})")

            current_level -= 1

        num_images = len(images)
        n_cols = int(np.ceil(np.sqrt(num_images)))
        n_rows = int(np.ceil(num_images / n_cols))

        plt.figure(figsize=(3 * n_cols, 3 * n_rows))
        plt.suptitle(figure_title, fontsize=14)

        for idx, (img, title) in enumerate(zip(images, titles), start=1):
            ax = plt.subplot(n_rows, n_cols, idx)
            ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        if save_path is not None:
            plt.savefig(save_path, dpi=150)

        if show:
            plt.show()


def _print_coeffs_info(coeffs: List, wavelet_name: str, level: int) -> None:
    """
    DWT katsayılarının yapısını ve bant boyutlarını konsola yazar.
    Kuantalama aşamasına geçmeden önce doğrulama için kullanılır.
    """
    print(f"  DWT katsayı yapısı ({wavelet_name}, seviye {level}):")
    print(f"    - coeffs[0] (LL): shape {coeffs[0].shape}, dtype {coeffs[0].dtype}")
    for i, det in enumerate(coeffs[1:], start=1):
        cH, cV, cD = det
        print(f"    - coeffs[{i}] (HL,LH,HH): shapes {cH.shape}, {cV.shape}, {cD.shape}")


def run_example(
    image_path: str,
    target_size: Tuple[int, int] = (512, 512),
    wavelet_name: str = "haar",
    level: int = 2,
    force_grayscale: bool = False,
):
    """
    1. aşamayı uçtan uca çalıştıran örnek fonksiyon.

    Adımlar:
    1) Görüntüyü oku
    2) 512x512 / 1024x1024 boyutlandır
    3) Y (veya gri seviye) kanalını çıkar
    4) Seviye-2 Haar (veya seçilen dalgacık) ile DWT uygula
    5) LL, LH, HL, HH alt bantlarını çok seviyeli yapıda görselleştir
    6) Kuantalama aşamasına verilecek katsayıları döndür
    """
    print("[1/4] Görüntü yükleniyor...")
    preprocessor = ImagePreprocessor(image_path=image_path)
    preprocessor.load_image()
    original_bgr = preprocessor.original_bgr
    assert original_bgr is not None, "Görüntü yüklenemedi."
    h_orig, w_orig = original_bgr.shape[:2]
    print(f"  Orijinal görüntü: shape {original_bgr.shape}, dtype {original_bgr.dtype}")

    print(f"[2/4] Luma kanalı üretiliyor (hedef boyut {target_size})...")
    luma = preprocessor.convert_to_luma(
        target_size=target_size,
        force_grayscale=force_grayscale,
    )
    assert luma.shape == (target_size[1], target_size[0]), (
        f"Luma boyutu beklenen {target_size} ile uyuşmuyor: {luma.shape}"
    )
    print(f"  Luma shape: {luma.shape}, dtype: {luma.dtype}, min={float(luma.min()):.1f}, max={float(luma.max()):.1f}")

    plt.figure(figsize=(4, 4))
    plt.title("İşlenecek Parlaklık (Y) Kanalı")
    plt.imshow(luma, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"[3/4] DWT uygulanıyor ({wavelet_name}, seviye {level})...")
    dwt = DWTTransformer(wavelet_name=wavelet_name, level=level)
    coeffs = dwt.apply_dwt2d(luma)
    _print_coeffs_info(coeffs, wavelet_name, level)

    print("[4/4] Alt bantlar görselleştiriliyor...")
    dwt.visualize_multilevel_coeffs(
        coeffs=coeffs,
        show=True,
        save_path=None,
        figure_title=f"{wavelet_name} DWT (Seviye {level}) Alt Bantları",
    )

    print("Aşama 1 tamamlandı. DWT katsayıları (coeffs) kuantalama modülüne verilmek üzere hazır.")
    return coeffs


if __name__ == "__main__":
    """
    Basit komut satırı testi.

    Bu dosyanın bulunduğu klasörde terminal açıp:

        python jpeg_jpeg2000_stage1.py

    şeklinde çalıştırabilirsiniz.
    Farklı bir görsel için yolu argüman olarak verin:

        python jpeg_jpeg2000_stage1.py path/to/image.jpg
    """
    import sys

    example_image_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "test_images_generated", "test_natural_landscape.jpg")
    )

    target_size = (512, 512)
    wavelet_name = "haar"
    level = 2
    force_grayscale = True  # Demo tek kanallı Y (parlaklık) üzerinde çalışır

    if not os.path.isfile(example_image_path):
        raise SystemExit(
            "Giriş görüntüsü bulunamadı.\n"
            "Önce 'python generate_test_images.py' çalıştırın ya da görsel yolunu "
            f"argüman olarak verin. Aranan yol: {example_image_path!r}"
        )

    coeffs = run_example(
        image_path=example_image_path,
        target_size=target_size,
        wavelet_name=wavelet_name,
        level=level,
        force_grayscale=force_grayscale,
    )

    # 'coeffs' değişkeni, kuantalama ve Huffman kodlama aşamalarına
    # girdi olarak verilebilecek DWT katsayılarını içerir.

