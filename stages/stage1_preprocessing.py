"""
stage1_preprocessing.py — Aşama 1: Giriş İşlemleri & DWT

Bu modül, JPEG/JPEG2000 benzeri görüntü sıkıştırma sisteminin
1. aşamasını (Input Handling + Source Encoding / DWT) gerçekleştirir.

Ana özellikler:
- PNG / BMP / TIFF vb. formatlardan görüntü okuma
- 512x512 veya 1024x1024 boyutlandırma
- RGB → YCbCr dönüştürme, sadece Y (parlaklık) kanalı ile çalışma
- Çok seviyeli (multi-level) Haar veya Daubechies (db2, db4, ...) DWT
- DWT alt bantlarını (LL, LH, HL, HH) çoklu kare halinde görselleştirme
- Çıktı: Kuantalama aşamasına gidecek DWT katsayıları (coeffs)

Kullanım:
    >>> from stages.stage1_preprocessing import ImagePreprocessor, DWTTransformer
    >>> proc = ImagePreprocessor("image.png")
    >>> proc.load_image()
    >>> luma = proc.convert_to_luma((512, 512))
    >>> dwt = DWTTransformer("db4", level=2)
    >>> coeffs = dwt.apply_dwt2d(luma)
"""

from pathlib import Path
from typing import Tuple, List, Optional, Union

import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

from core.logging_config import setup_logger
from core.utils import validate_image_path, validate_image_shape

logger = setup_logger("stage1", "stage1.log")


class ImagePreprocessor:
    """
    Giriş görüntüsünü okuma, boyutlandırma ve uygun renk uzayına dönüştürme
    işlemlerini kapsayan sınıf.

    Genel kullanım:
    - Renkli görüntülerde, YCbCr renk uzayına dönüp sadece Y (parlaklık)
      kanalı ile çalışmak yaygın JPEG / JPEG2000 yaklaşımıdır.
    - Parmak izi / tıbbi gibi tek kanallı (grayscale) görüntüler için
      doğrudan gri seviye ile çalışmak mümkündür.
    
    Attributes:
        image_path (str): Giriş görüntüsünün dosya yolu
        original_bgr (np.ndarray): BGR formatında orijinal görüntü
        luma_channel (np.ndarray): İşlenmiş Y kanalı (float32)
    """

    def __init__(self, image_path: Union[str, Path]) -> None:
        """
        Initialize ImagePreprocessor.
        
        Args:
            image_path: Giriş görüntüsünün dosya yolu
        """
        self.image_path = Path(image_path)
        self.original_bgr: Optional[np.ndarray] = None
        self.luma_channel: Optional[np.ndarray] = None
        
        # Path doğrulaması
        try:
            validate_image_path(self.image_path)
        except ValueError as e:
            logger.error(f"Geçersiz dosya yolu: {e}")
            raise

    def load_image(self) -> np.ndarray:
        """
        Görüntüyü BGR formatında okur.
        
        Türkçe karakter içeren yollar (Ü, ü, ğ, ş vb.) Windows'ta
        OpenCV'nin cv2.imread ile sorun çıkardığı için dosya Python ile
        binary okunup cv2.imdecode ile açılır; böylece Unicode yol sorunu olmaz.
        
        Returns:
            BGR formatında görüntü (np.ndarray)
            
        Raises:
            ValueError: Dosya okunamadı veya desteklenmeyen format
        """
        logger.info(f"Görüntü yükleniyor: {self.image_path}")
        
        try:
            with open(self.image_path, "rb") as f:
                buf = np.frombuffer(f.read(), dtype=np.uint8)
            
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                raise ValueError(f"Görüntü okunamadı (desteklenmeyen format?): {self.image_path}")
            
            self.original_bgr = img
            logger.info(f"✓ Görüntü yüklendi: shape={img.shape}, dtype={img.dtype}")
            
            return img
        
        except Exception as e:
            logger.error(f"Görüntü yükleme hatası: {e}")
            raise

    def convert_to_luma(
        self,
        target_size: Tuple[int, int] = (512, 512),
        force_grayscale: bool = False,
    ) -> np.ndarray:
        """
        Orijinal BGR görüntüyü, DWT aşamasında kullanılacak 2B parlaklık (luma)
        kanalına dönüştürür ve istenen boyuta ölçekler.

        Parameters:
            target_size: (genişlik, yükseklik) olarak hedef boyut
            force_grayscale: 
                - True: BGR görüntü gri tonlamaya dönüştürülür (parmak izi / medikal)
                - False: BGR görüntü YCrCb'ye dönüştürülür, yalnızca Y kanalı kullanılır

        Returns:
            target_size boyutunda, 2B numpy dizisi (float32) [0, 255] aralığında
            
        Raises:
            RuntimeError: Görüntü henüz yüklenmedi
            ValueError: target_size geçersiz
        """
        if self.original_bgr is None:
            raise RuntimeError("Görüntü henüz yüklenmedi. Önce load_image() çağrılmalı.")
        
        logger.info(f"Luma kanalı üretiliyor: target_size={target_size}, "
                    f"force_grayscale={force_grayscale}")
        
        try:
            # Resize
            resized = cv2.resize(self.original_bgr, target_size, interpolation=cv2.INTER_AREA)
            
            # Renk dönüşümü
            if force_grayscale:
                luma = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
                luma = ycrcb[:, :, 0]
            
            self.luma_channel = luma.astype(np.float32)
            
            logger.info(f"✓ Luma kanalı hazır: shape={self.luma_channel.shape}, "
                       f"min={self.luma_channel.min():.1f}, max={self.luma_channel.max():.1f}")
            
            return self.luma_channel
        
        except Exception as e:
            logger.error(f"Luma dönüşüm hatası: {e}")
            raise


class DWTTransformer:
    """
    2B DWT (Discrete Wavelet Transform) ve görselleştirme işlemlerini
    kapsayan sınıf.

    Bu sınıf, ileride kuantalama ve Huffman aşamalarının doğrudan
    kullanacağı katsayıları (coeffs) üretir.
    
    Attributes:
        wavelet_name (str): Dalgacık tipi (haar, db2, db4, bior4.4, ...)
        level (int): Çok seviyeli ayrıştırma derecesi (1, 2, 3, ...)
    """

    def __init__(self, wavelet_name: str = "haar", level: int = 2) -> None:
        """
        Initialize DWTTransformer.
        
        Args:
            wavelet_name: PyWavelets dalgacık isimleri
            level: Çok seviyeli ayrıştırma seviyeleri
        """
        self.wavelet_name = wavelet_name
        self.level = level
        
        logger.info(f"DWTTransformer initialized: wavelet={wavelet_name}, level={level}")

    def apply_dwt2d(self, image_2d: np.ndarray) -> List:
        """
        2B DWT uygular ve pywt.wavedec2'nin döndürdüğü katsayı yapısını verir.

        Parameters:
            image_2d: 2B numpy dizisi (ör. Y kanalı), float32 veya float64

        Returns:
            Katsayı listesi (pywt.wavedec2 formatı)
            
        Örnek (level=2):
            coeffs[0] → cA2 (LL2)
            coeffs[1] → (cH2, cV2, cD2) → (HL2, LH2, HH2)
            coeffs[2] → (cH1, cV1, cD1) → (HL1, LH1, HH1)
            
        Raises:
            ValueError: Giriş 2D değil
        """
        try:
            validate_image_shape(image_2d, expected_dims=2)
        except ValueError as e:
            logger.error(f"DWT input validation hatası: {e}")
            raise
        
        logger.info(f"DWT uygulanıyor: wavelet={self.wavelet_name}, level={self.level}, "
                   f"image_shape={image_2d.shape}")
        
        try:
            coeffs = pywt.wavedec2(
                data=image_2d,
                wavelet=self.wavelet_name,
                level=self.level,
                mode="periodization",  # JPEG2000 uyumu
            )
            
            logger.info(f"✓ DWT tamamlandı: {len(coeffs)} alt bant")
            self._log_coeffs_info(coeffs)
            
            return coeffs
        
        except Exception as e:
            logger.error(f"DWT uygulama hatası: {e}")
            raise

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
        save_path: Optional[Union[str, Path]] = None,
        figure_title: str = "DWT Alt Bantları",
    ) -> None:
        """
        Çok seviyeli DWT katsayılarını grid halinde görselleştirir.

        Parameters:
            coeffs: DWT katsayıları (pywt.wavedec2 formatı)
            show: Pencerede göster
            save_path: Dosyaya kaydet (None = kaydetme)
            figure_title: Figür başlığı
        """
        try:
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

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            if save_path is not None:
                plt.savefig(save_path, dpi=150)
                logger.info(f"✓ Görselleştirme kaydedildi: {save_path}")

            if show:
                plt.show()
        
        except Exception as e:
            logger.error(f"Görselleştirme hatası: {e}")
            raise

    @staticmethod
    def _log_coeffs_info(coeffs: List) -> None:
        """DWT katsayılarının yapısını loga yaz."""
        logger.debug(f"Katsayı yapısı:")
        logger.debug(f"  - coeffs[0] (LL): shape {coeffs[0].shape}, dtype {coeffs[0].dtype}")
        for i, det in enumerate(coeffs[1:], start=1):
            cH, cV, cD = det
            logger.debug(f"  - coeffs[{i}]: HL{cH.shape}, LH{cV.shape}, HH{cD.shape}")


# ============================================================================
# ✅ ÖRNEK KULLANIM
# ============================================================================

def run_example(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (512, 512),
    wavelet_name: str = "haar",
    level: int = 2,
    force_grayscale: bool = False,
) -> List:
    """
    Aşama 1'i uçtan uca çalıştıran örnek fonksiyon.
    
    Args:
        image_path: Giriş görüntüsü
        target_size: Hedef boyut
        wavelet_name: Dalgacık tipi
        level: DWT seviyeleri
        force_grayscale: Gri tonlama modu
        
    Returns:
        DWT katsayıları (coeffs)
    """
    logger.info("=" * 60)
    logger.info("AŞAMA 1: GİRİŞ + DWT BAŞLATILIYOR")
    logger.info("=" * 60)
    
    try:
        # 1. Görüntü yükle
        logger.info("[1/4] Görüntü yükleniyor...")
        proc = ImagePreprocessor(image_path)
        proc.load_image()
        
        # 2. Luma kanalı
        logger.info(f"[2/4] Luma kanalı üretiliyor...")
        luma = proc.convert_to_luma(target_size=target_size, force_grayscale=force_grayscale)
        
        # 3. DWT uygula
        logger.info(f"[3/4] DWT uygulanıyor...")
        dwt = DWTTransformer(wavelet_name=wavelet_name, level=level)
        coeffs = dwt.apply_dwt2d(luma)
        
        # 4. Görselleştir
        logger.info("[4/4] Alt bantlar görselleştiriliyor...")
        dwt.visualize_multilevel_coeffs(
            coeffs=coeffs,
            show=True,
            figure_title=f"{wavelet_name} DWT (Seviye {level}) Alt Bantları",
        )
        
        logger.info("✅ AŞAMA 1 TAMAMLANDI")
        logger.info("=" * 60)
        
        return coeffs
    
    except Exception as e:
        logger.error(f"❌ AŞAMA 1 HATASI: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Test örneği
    print("Stage 1 Preprocessing Module")
    print("=" * 60)
    
    # Test görselleri bulunabilir mi?
    test_images_dir = Path(__file__).parent.parent / "test_images_generated"
    test_file = test_images_dir / "test_natural_landscape.jpg"
    
    if test_file.exists():
        logger.info(f"Test dosyası bulundu: {test_file}")
        run_example(test_file, target_size=(512, 512), wavelet_name="db4", level=2)
    else:
        logger.error(f"Test dosyası bulunamadı: {test_file}")
        logger.info("Önce 'python generate_test_images.py' çalıştırın")
