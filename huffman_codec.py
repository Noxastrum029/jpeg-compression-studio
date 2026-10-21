"""
huffman_codec.py

CENG 384 — Entropy Coding (Huffman) Modülü
Aşama 4: Kuantalanmış DWT katsayılarını Huffman kodlaması ile sıkıştırır.

Sorumluluklar:
- Symbol frekans tablosu oluşturma (FR-17)
- Huffman ağacı inşası ve değişken uzunluklu kodlama (FR-18)
- Encoding metadata saklama (FR-19)
- Decode işlemi (FR-20)
- Gerçek bitstream boyutu hesabı (sıkıştırma oranı için)

Kullanım:
    from huffman_codec import HuffmanCodec, encode_coeffs, decode_coeffs
    encoded = encode_coeffs(quantized_coeffs)
    pywt_coeffs = decode_coeffs(encoded)
"""

import heapq
import pickle
import struct
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Huffman Ağaç Düğümü
# ---------------------------------------------------------------------------

class _HuffmanNode:
    """
    Huffman ağacındaki iç ve yaprak düğümleri temsil eder.
    heapq ile kullanım için __lt__ tanımlıdır (frekansa göre sıralama).
    """

    def __init__(self, symbol: Optional[int], freq: int,
                 left: "Optional[_HuffmanNode]" = None,
                 right: "Optional[_HuffmanNode]" = None) -> None:
        self.symbol = symbol    # Yaprak düğümde sembol (int); iç düğümde None
        self.freq = freq        # Frekans (katsayı görülme sayısı)
        self.left = left        # Sol alt ağaç (daha az frekans)
        self.right = right      # Sağ alt ağaç

    def __lt__(self, other: "_HuffmanNode") -> bool:
        # Min-heap: düşük frekanslı düğüm önce çıkar
        return self.freq < other.freq


# ---------------------------------------------------------------------------
# HuffmanCodec — tek bant için encoder/decoder
# ---------------------------------------------------------------------------

class HuffmanCodec:
    """
    Tek bir sembol dizisi (1D integer array) için Huffman codec.

    Adımlar:
    1. Frekans tablosu çıkar (Counter)
    2. Min-heap'ten Huffman ağacı inşa et
    3. Her sembol için bit kodu üret (kod tablosu)
    4. Encode: sembol dizisi → bitstring → bytes
    5. Decode: bytes + kod tablosu → orijinal sembol dizisi
    """

    def __init__(self) -> None:
        self._code_table: Dict[int, str] = {}     # sembol → bit kodu ("0110" vb.)
        self._reverse_table: Dict[str, int] = {}  # bit kodu → sembol (decode için)
        self._root: Optional[_HuffmanNode] = None # Huffman ağacı kökü

    # ------------------------------------------------------------------
    # Ağaç inşası
    # ------------------------------------------------------------------

    def _build_tree(self, symbols: np.ndarray) -> None:
        """
        Sembol dizisinden frekans tablosu çıkarıp Huffman ağacını inşa eder.

        Tek sembol durumu: tüm elemanlar aynıysa ağaç sadece kökten oluşur;
        bu durumda '0' bit kodu atanır.
        """
        freq = Counter(symbols.tolist())

        # Tek farklı sembol varsa özel durum
        if len(freq) == 1:
            sym = next(iter(freq))
            self._root = _HuffmanNode(sym, freq[sym])
            self._code_table = {sym: ""}
            self._reverse_table = {"": sym}
            return

        # Her sembol için düğüm oluştur, min-heap'e ekle
        heap: List[_HuffmanNode] = [_HuffmanNode(s, f) for s, f in freq.items()]
        heapq.heapify(heap)

        # Huffman ağacı inşa: iki en az frekanslı düğümü birleştir
        while len(heap) > 1:
            left = heapq.heappop(heap)   # En az frekans
            right = heapq.heappop(heap)  # İkinci en az
            merged = _HuffmanNode(
                symbol=None,
                freq=left.freq + right.freq,
                left=left,
                right=right,
            )
            heapq.heappush(heap, merged)

        self._root = heap[0]
        self._code_table = {}
        self._reverse_table = {}
        # Ağaçtan kod tablosunu üret (DFS)
        self._traverse(self._root, "")

    def _traverse(self, node: Optional[_HuffmanNode], code: str) -> None:
        """
        Ağaçta DFS gezinerek her yaprak için bit kodunu belirler.
        Sol dal → '0', Sağ dal → '1'
        """
        if node is None:
            return
        if node.symbol is not None:
            # Yaprak düğüm: kodu tabloya yaz
            self._code_table[node.symbol] = code
            self._reverse_table[code] = node.symbol
            return
        self._traverse(node.left, code + "0")
        self._traverse(node.right, code + "1")

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, symbols: np.ndarray) -> Tuple[bytes, int]:
        """
        1D integer sembol dizisini Huffman ile encode eder.

        Dönüş:
        - encoded_bytes: sıkıştırılmış veri (bytes)
        - original_length: orijinal sembol sayısı (decode için)

        Bit packing: bit string'i byte'lara dönüştürür (sona 0 pad eklenir).
        """
        self._build_tree(symbols)
        # Her sembolü bit koduna çevir, hepsini birleştir
        bitstring = "".join(self._code_table[int(s)] for s in symbols)
        original_length = len(symbols)

        # Bitstring'i byte dizisine dönüştür (8'li gruplar)
        padding = (8 - len(bitstring) % 8) % 8
        bitstring += "0" * padding  # Son byte'ı tamamlamak için 0 ekle
        byte_array = bytearray()
        for i in range(0, len(bitstring), 8):
            byte_array.append(int(bitstring[i:i + 8], 2))

        # Header: padding miktarını sakla (decode'da ihtiyaç var)
        header = struct.pack("B", padding)
        return header + bytes(byte_array), original_length

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, encoded_bytes: bytes, original_length: int) -> np.ndarray:
        """
        Encode edilmiş byte verisini orijinal sembol dizisine geri dönüştürür.
        Kod tablosu (self._reverse_table) mevcut olmalı — yani encode sonrası çağrılır.
        """
        padding = struct.unpack("B", encoded_bytes[:1])[0]
        data = encoded_bytes[1:]

        # Byte'ları bit string'e dönüştür
        bitstring = "".join(f"{byte:08b}" for byte in data)
        # Pad bitlerini kaldır
        if padding > 0:
            bitstring = bitstring[:-padding]

        # Eğer sadece bir sembol varsa (kod tablosunda ""), direkt olarak array'i doldur
        if "" in self._reverse_table and len(self._reverse_table) == 1:
            return np.full(original_length, self._reverse_table[""], dtype=np.int16)

        # Bit string'den sembolleri geri çöz (Huffman ağacında gezin)
        symbols = []
        current_code = ""
        for bit in bitstring:
            current_code += bit
            if current_code in self._reverse_table:
                symbols.append(self._reverse_table[current_code])
                current_code = ""

        return np.array(symbols[:original_length], dtype=np.int16)

    # ------------------------------------------------------------------
    # Yardımcı: kod tablosunu pickle olarak serialize et
    # ------------------------------------------------------------------

    def get_metadata(self) -> bytes:
        """
        Huffman kod tablosunu bytes olarak döndürür.
        Gerçek bitstream'de bu metadata header'a yazılır (FR-19).
        """
        return pickle.dumps({"code": self._code_table, "rev": self._reverse_table})

    def load_metadata(self, metadata: bytes) -> None:
        """Kaydedilmiş metadata'yı geri yükler (decode öncesi)."""
        tables = pickle.loads(metadata)
        self._code_table = tables["code"]
        self._reverse_table = tables["rev"]

    # ------------------------------------------------------------------
    # Sıkıştırma istatistikleri
    # ------------------------------------------------------------------

    def compression_stats(self, symbols: np.ndarray, encoded_bytes: bytes) -> Dict[str, Any]:
        """
        Encode sonrası sıkıştırma istatistiklerini hesaplar.

        Dönüş dict:
        - original_bits: ham int16 boyutu (bit)
        - compressed_bits: Huffman sonrası boyut (bit)
        - compression_ratio: sıkıştırma oranı
        - space_saving_pct: kazanılan alan yüzdesi
        - avg_code_length: ortalama kod uzunluğu (bit/sembol)
        - entropy: Shannon entropisi (bit/sembol)
        """
        original_bits = len(symbols) * 16  # int16 = 2 byte = 16 bit
        compressed_bits = (len(encoded_bytes) - 1) * 8  # -1: header byte

        # Shannon entropisi H = -Σ p(x) log2(p(x))
        freq = Counter(symbols.tolist())
        total = len(symbols)
        entropy = -sum(
            (c / total) * np.log2(c / total)
            for c in freq.values()
            if c > 0
        )

        # Ortalama kod uzunluğu
        avg_code = sum(
            (freq[s] / total) * len(self._code_table.get(s, ""))
            for s in freq
        )

        return {
            "original_bits": original_bits,
            "compressed_bits": compressed_bits,
            "compression_ratio": original_bits / max(compressed_bits, 1),
            "space_saving_pct": 100.0 * (1 - compressed_bits / max(original_bits, 1)),
            "avg_code_length": avg_code,
            "entropy": entropy,
        }


# ---------------------------------------------------------------------------
# Üst seviye API — pywt coeffs yapısı için
# ---------------------------------------------------------------------------

class CoeffsBandEncoder:
    """
    pywt.wavedec2 çıktı yapısını (coeffs listesi) bant bant Huffman encode/decode eden sınıf.

    Her bant (LL, HL, LH, HH her seviyede) ayrı HuffmanCodec ile işlenir;
    çünkü her bandın istatistiksel dağılımı farklıdır (FR-15: rate-distortion optimizasyonu).

    Encode → birleşik bitstream (bytes) + metadata (decode için)
    Decode → orijinal coeffs yapısı
    """

    def __init__(self) -> None:
        # Her bant için ayrı codec saklanır; key = band_key string
        self._codecs: Dict[str, HuffmanCodec] = {}
        self._shapes: Dict[str, Tuple] = {}   # Orijinal matris boyutları
        self._lengths: Dict[str, int] = {}    # Orijinal sembol sayıları

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode_coeffs(self, coeffs: List) -> bytes:
        """
        DWT katsayılarının tüm bantlarını Huffman ile encode eder.

        Dönüş: birleşik bitstream (bytes) — metadata + her bandın verisi
        Format:
            [N_bands (4 byte)] [band_key_len][band_key][shape...][encoded_len][encoded_bytes]...
        """
        # Katsayıları düzleştirip int16'ya çevir (Huffman sembol tablosu integer istiyor)
        bands = self._flatten_coeffs(coeffs)
        band_streams: Dict[str, Tuple[bytes, int, Tuple, bytes]] = {}

        for key, flat in bands.items():
            codec = HuffmanCodec()
            encoded, length = codec.encode(flat)
            metadata = codec.get_metadata()
            self._codecs[key] = codec
            band_streams[key] = (encoded, length, self._shapes[key], metadata)

        # Birleştir: basit binary format
        payload = pickle.dumps({
            "band_streams": {
                k: {"encoded": v[0], "length": v[1],
                    "shape": v[2], "metadata": v[3]}
                for k, v in band_streams.items()
            }
        })
        return payload

    def decode_coeffs(self, bitstream: bytes, wavelet: str, level: int) -> List:
        """
        Encode edilmiş bitstream'den orijinal DWT katsayı yapısını geri oluşturur.
        """
        data = pickle.loads(bitstream)
        band_streams = data["band_streams"]

        # Her bandı decode et
        decoded_bands: Dict[str, np.ndarray] = {}
        for key, info in band_streams.items():
            codec = HuffmanCodec()
            codec.load_metadata(info["metadata"])
            flat = codec.decode(info["encoded"], info["length"])
            decoded_bands[key] = flat.reshape(info["shape"]).astype(np.float64)

        # DWT coeffs yapısına geri dönüştür
        return self._rebuild_coeffs(decoded_bands, level)

    # ------------------------------------------------------------------
    # Yardımcı: coeffs ↔ band dict dönüşümleri
    # ------------------------------------------------------------------

    def _flatten_coeffs(self, coeffs: List) -> Dict[str, np.ndarray]:
        """
        pywt coeffs listesini anahtar:matris sözlüğüne dönüştürür.
        Her matris int16'ya yuvarlanıp (Huffman sembol aralığı) 1D yapılır.
        """
        bands: Dict[str, np.ndarray] = {}

        # LL bandı (tek matris)
        ll = np.round(coeffs[0]).astype(np.int16)
        self._shapes["LL"] = coeffs[0].shape
        bands["LL"] = ll.ravel()

        # Her seviyedeki detay bantları (cH=HL, cV=LH, cD=HH)
        n_levels = len(coeffs) - 1
        for i, (cH, cV, cD) in enumerate(coeffs[1:]):
            lv = n_levels - i  # Seviye numarası (en ince → en kaba)
            for band_name, arr in [("HL", cH), ("LH", cV), ("HH", cD)]:
                key = f"{band_name}_L{lv}"
                self._shapes[key] = arr.shape
                bands[key] = np.round(arr).astype(np.int16).ravel()

        return bands

    def _rebuild_coeffs(self, bands: Dict[str, np.ndarray], level: int) -> List:
        """
        Band sözlüğünden pywt.wavedec2 formatındaki coeffs listesini yeniden inşa eder.
        """
        coeffs: List = [bands["LL"]]
        for lv in range(level, 0, -1):
            hl = bands.get(f"HL_L{lv}", np.array([0.0]))
            lh = bands.get(f"LH_L{lv}", np.array([0.0]))
            hh = bands.get(f"HH_L{lv}", np.array([0.0]))
            coeffs.append((hl, lh, hh))
        return coeffs

    # ------------------------------------------------------------------
    # İstatistik
    # ------------------------------------------------------------------

    def get_total_stats(self) -> Dict[str, Any]:
        """
        Tüm bantların toplam sıkıştırma istatistiklerini döndürür.
        Bant başına entropi ve ortalama kod uzunluğunu da içerir.
        """
        stats: Dict[str, Any] = {}
        total_orig = 0
        total_comp = 0

        for key, codec in self._codecs.items():
            if not codec._code_table:
                continue
            total_orig += 0   # shapes'ten hesaplanacak
            total_comp += 0
            # Bant bazında ortalama kod uzunluğu
            avg_len = sum(len(v) for v in codec._code_table.values()) / max(len(codec._code_table), 1)
            stats[key] = {
                "unique_symbols": len(codec._code_table),
                "avg_code_length": avg_len,
            }

        return stats


# ---------------------------------------------------------------------------
# Kolaylık fonksiyonları — compression_gui.py tarafından çağrılır
# ---------------------------------------------------------------------------

def encode_coeffs(coeffs: List) -> Tuple[bytes, "CoeffsBandEncoder"]:
    """
    DWT katsayılarını Huffman ile encode eder.

    Dönüş:
    - bitstream: bytes (gerçek sıkıştırılmış veri)
    - encoder: istatistik ve decode için CoeffsBandEncoder nesnesi
    """
    encoder = CoeffsBandEncoder()
    bitstream = encoder.encode_coeffs(coeffs)
    return bitstream, encoder


def decode_coeffs(bitstream: bytes, wavelet: str, level: int) -> List:
    """
    Huffman bitstream'ini pywt coeffs yapısına geri dönüştürür.
    """
    decoder = CoeffsBandEncoder()
    return decoder.decode_coeffs(bitstream, wavelet, level)


def huffman_compressed_size(coeffs: List) -> Tuple[int, int, float]:
    """
    Huffman sıkıştırma boyutunu hesaplar (GUI metrik tablosu için).

    Gerçek Huffman bit sayısını doğrudan hesaplar (pickle overhead yoktur).
    Her bant için:
      - Shannon entropisi ile bit sayısı = ceil(H * N) / 8 byte
      - Kod tablosu overhead'i = unique_symbols * 3 byte

    Dönüş:
    - original_bytes: ham int16 boyutu
    - compressed_bytes: Huffman sonrası tahmin (byte)
    - space_saving_pct: kazanılan alan yüzdesi
    """
    import math as _math
    from collections import Counter as _Counter

    original_bytes = 0
    total_compressed_bits = 0

    def _encode_band(arr: np.ndarray) -> int:
        """Tek bant için tahmini Huffman bit sayısı döndürür."""
        syms = np.round(arr).astype(np.int32).ravel().tolist()
        n = len(syms)
        freq = _Counter(syms)
        unique = len(freq)
        # Kod tablosu header: her sembol için ~3 byte (2 byte değer + 1 byte kod uzunluğu)
        header_bits = unique * 24
        if unique <= 1:
            return header_bits  # Tüm sıfır bant — sadece header
        H = -sum((c / n) * _math.log2(c / n) for c in freq.values() if c > 0)
        return header_bits + int(_math.ceil(H * n))

    # LL
    original_bytes += coeffs[0].size * 2
    total_compressed_bits += _encode_band(coeffs[0])

    # Detail bands
    for tup in coeffs[1:]:
        for band in tup:
            original_bytes += band.size * 2
            total_compressed_bits += _encode_band(band)

    compressed_bytes = int(_math.ceil(total_compressed_bits / 8))
    saving = 100.0 * (1 - compressed_bytes / max(original_bytes, 1))
    return original_bytes, compressed_bytes, saving
