"""
stage3_huffman_coding.py — Aşama 3: Entropy Kodlama (Huffman)

Huffman kodlaması, kuantalanmış DWT katsayılarını bit akışına dönüştürür.
Bu sayede sıkışmalı (lossless) veri sıkıştırması yapılır.

İki ana işlem:
1. Huffman Kodlama: Integer değerler → Variable-length binary codes
2. Huffman Dekodlama: Binary codes → Integer değerler

Kullanım:
    >>> from stages.stage3_huffman_coding import HuffmanEncoder, HuffmanDecoder
    >>> encoder = HuffmanEncoder()
    >>> encoder.build_tree(quantized_coeffs)
    >>> bitstream, code_table = encoder.encode(quantized_coeffs)
    >>> decoder = HuffmanDecoder(code_table)
    >>> decoded = decoder.decode(bitstream)
"""

import heapq
import json
from collections import defaultdict, Counter
from typing import Any, List, Dict, Tuple, Optional, Union
import numpy as np

from core.logging_config import setup_logger
from core.config import MAX_CODE_LENGTH

logger = setup_logger("stage3", "stage3.log")


class HuffmanNode:
    """
    Huffman ağacının düğümü.
    
    Attributes:
        symbol (int): İnt
egral value
        frequency (int): Frekans
        left (HuffmanNode): Sol alt ağaç
        right (HuffmanNode): Sağ alt ağaç
    """
    
    def __init__(self, symbol: Optional[int] = None, frequency: int = 0,
                 left: Optional['HuffmanNode'] = None,
                 right: Optional['HuffmanNode'] = None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = left
        self.right = right
    
    def __lt__(self, other: 'HuffmanNode') -> bool:
        """Heap sıralaması için karşılaştırma"""
        return self.frequency < other.frequency
    
    def __repr__(self) -> str:
        if self.symbol is not None:
            return f"Node(sym={self.symbol}, freq={self.frequency})"
        return f"Node(internal, freq={self.frequency})"


class HuffmanEncoder:
    """
    Huffman kodlama: Değerler → Binary kodlar
    
    Attributes:
        code_table (Dict): Symbol → binary code mapping
        reverse_table (Dict): Binary code → symbol mapping
        root (HuffmanNode): Huffman ağacının kökü
    """
    
    def __init__(self):
        self.code_table: Dict[int, str] = {}
        self.reverse_table: Dict[str, int] = {}
        self.root: Optional[HuffmanNode] = None
        self.frequencies: Counter = Counter()
    
    # ========================================================================
    # AĞAÇ İNŞASI
    # ========================================================================
    
    def _get_flat_values(self, coeffs: Any) -> List[int]:
        """Her türlü girişi (DWT listesi, DCT array, vb.) düz bir int listesine çevir"""
        flat = []
        
        def _recursive_extract(item):
            if isinstance(item, (list, tuple)):
                for sub in item:
                    _recursive_extract(sub)
            elif isinstance(item, np.ndarray):
                flat.extend(item.flatten())
            else:
                flat.append(item)
        
        _recursive_extract(coeffs)
        # Yuvarla ve int'e çevir
        return [int(np.round(x)) for x in flat]

    def build_tree_from_coefficients(self, coeffs: Any) -> None:
        """
        DWT/DCT katsayılarından frekans tablosu oluşturarak Huffman ağacını inşa et.
        """
        logger.info("Huffman ağacı inşa ediliyor (coefficients'tan)...")
        
        try:
            int_values = self._get_flat_values(coeffs)
            self.frequencies = Counter(int_values)
            self._build_tree_from_frequencies()
        
        except Exception as e:
            logger.error(f"Ağaç inşası hatası: {e}")
            raise
    
    def build_tree_from_frequencies(self, frequencies: Dict[int, int]) -> None:
        """
        Verilen frekans tablosundan Huffman ağacını inşa et.
        
        Args:
            frequencies: Symbol → frequency mapping
        """
        logger.info(f"Huffman ağacı inşa ediliyor ({len(frequencies)} symbol)...")
        
        try:
            self.frequencies = Counter(frequencies)
            self._build_tree_from_frequencies()
        except Exception as e:
            logger.error(f"Ağaç inşası hatası: {e}")
            raise
    
    def _build_tree_from_frequencies(self) -> None:
        """İç metod: frekans tablosundan ağaç inşası"""
        if not self.frequencies:
            raise ValueError("Frekans tablosu boş")
        
        # Öncelik kuyruğu (heap)
        heap = [
            HuffmanNode(symbol=symbol, frequency=freq)
            for symbol, freq in self.frequencies.items()
        ]
        heapq.heapify(heap)
        
        # Ağaç inşası (bottom-up)
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            parent = HuffmanNode(
                frequency=left.frequency + right.frequency,
                left=left,
                right=right
            )
            heapq.heappush(heap, parent)
        
        self.root = heap[0] if heap else None
        
        # Kod tablosu oluştur
        self.code_table = {}
        self._generate_codes(self.root, "")
        self.reverse_table = {v: k for k, v in self.code_table.items()}
        
        logger.info(f"✓ Ağaç hazır: {len(self.code_table)} kod ilişkisi")
    
    def _generate_codes(self, node: Optional[HuffmanNode], code: str) -> None:
        """
        Ağacı traverse ederek kod tablosunu üret.
        
        Args:
            node: Traverse edilecek düğüm
            code: Şimdiye kadar oluşturulan kod
        """
        if node is None:
            return
        
        # Yaprak düğümü (symbol)
        if node.symbol is not None:
            if not code:
                code = "0"  # Tek sembol için fallback
            self.code_table[node.symbol] = code
        else:
            # İç düğümler
            self._generate_codes(node.left, code + "0")
            self._generate_codes(node.right, code + "1")
    
    # ========================================================================
    # KODLAMA
    # ========================================================================
    
    def encode(self, coeffs: Any) -> Tuple[str, Dict[int, str]]:
        """
        Katsayıları Huffman kodlaması ile encode et.
        """
        logger.info("Huffman kodlama başlıyor...")
        
        try:
            # Ağaç henüz inşa edilmediyse inşa et
            if not self.code_table:
                self.build_tree_from_coefficients(coeffs)
            
            int_values = self._get_flat_values(coeffs)
            
            bitstream_list = []
            total_symbols = 0
            
            for val in int_values:
                if val in self.code_table:
                    bitstream_list.append(self.code_table[val])
                    total_symbols += 1
            
            bitstream = "".join(bitstream_list)
            
            logger.info(f"✓ Kodlama tamamlandı: {total_symbols} symbol, {len(bitstream)} bit")
            return bitstream, self.code_table
        
        except Exception as e:
            logger.error(f"Kodlama hatası: {e}")
            raise
    
    def get_code_table_bytes(self) -> bytes:
        """
        Kod tablosunu JSON olarak serialize et.
        
        Returns:
            JSON bytes
        """
        # Integer anahtarları string'e dönüştür (JSON uyumluluğu)
        json_table = {str(k): v for k, v in self.code_table.items()}
        return json.dumps(json_table).encode('utf-8')
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Kodlama istatistikleri.
        
        Returns:
            İstatistik sözlüğü
        """
        if not self.code_table:
            return {}
        
        code_lengths = [len(code) for code in self.code_table.values()]
        
        return {
            'num_symbols': len(self.code_table),
            'min_code_length': min(code_lengths) if code_lengths else 0,
            'max_code_length': max(code_lengths) if code_lengths else 0,
            'avg_code_length': sum(code_lengths) / len(code_lengths) if code_lengths else 0,
            'entropy': self._calculate_entropy(),
        }
    
    def _calculate_entropy(self) -> float:
        """Shannon entropy hesapla"""
        if not self.frequencies:
            return 0.0
        
        total = sum(self.frequencies.values())
        entropy = 0.0
        
        for freq in self.frequencies.values():
            p = freq / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return float(entropy)
    
    # ========================================================================
    # PRETTY PRINT
    # ========================================================================
    
    def print_code_table(self, limit: int = 20) -> None:
        """
        Kod tablosunu tablo formatında yazdır.
        
        Args:
            limit: Gösterilecek satır limitin sayısı
        """
        print(f"\n{'='*60}")
        print(f"  Huffman Kod Tablosu (İlk {limit} satır)")
        print(f"{'='*60}")
        print(f"{'Symbol':>10} | {'Code':>30} | {'Length':>6}")
        print(f"{'-'*60}")
        
        count = 0
        for symbol in sorted(self.code_table.keys())[:limit]:
            code = self.code_table[symbol]
            print(f"{symbol:>10} | {code:>30} | {len(code):>6}")
            count += 1
        
        if len(self.code_table) > limit:
            print(f"{'-'*60}")
            print(f"... ve {len(self.code_table) - limit} daha")
        
        print(f"{'='*60}\n")


class HuffmanDecoder:
    """
    Huffman dekolaması: Binary kodlar → Değerler
    
    Attributes:
        code_table (Dict): Symbol → binary code mapping
        reverse_table (Dict): Binary code → symbol mapping
    """
    
    def __init__(self, code_table: Dict[Union[int, str], str]):
        """
        Initialize decoder.
        
        Args:
            code_table: Kodlama tablosu
        """
        logger.info("Huffman dekodlayıcı initialize ediliyor...")
        
        # String anahtarları integer'a dönüştür
        self.code_table: Dict[int, str] = {}
        for k, v in code_table.items():
            if isinstance(k, str):
                try:
                    self.code_table[int(k)] = v
                except ValueError:
                    self.code_table[int(k)] = v
            else:
                self.code_table[k] = v
        
        # Ters tablo: kod → symbol
        self.reverse_table: Dict[str, int] = {}
        for symbol, code in self.code_table.items():
            self.reverse_table[code] = symbol
        
        logger.info(f"✓ Dekodlayıcı hazır: {len(self.reverse_table)} kod")
    
    def decode(self, bitstream: str) -> List[int]:
        """
        Bitstream'i dekod ederek sembol listesi döndür.
        
        Args:
            bitstream: Binary string
            
        Returns:
            Decoded integer list
        """
        logger.info(f"Huffman dekolaması başlıyor ({len(bitstream)} bit)...")
        
        try:
            result = []
            current_code = ""
            
            for bit in bitstream:
                current_code += bit
                
                if current_code in self.reverse_table:
                    result.append(self.reverse_table[current_code])
                    current_code = ""
                
                # Sonsuz döngüden kaçın
                if len(current_code) > MAX_CODE_LENGTH:
                    logger.warning(f"Kod limit aşıldı: {current_code[:50]}...")
                    break
            
            # Sonra kalan code
            if current_code:
                logger.warning(f"Eksik kod: {current_code}")
            
            logger.info(f"✓ Dekolaması tamamlandı: {len(result)} symbol")
            
            return result
        
        except Exception as e:
            logger.error(f"Dekolaması hatası: {e}")
            raise
    
    @staticmethod
    def load_code_table_from_json(json_bytes: bytes) -> Dict[int, str]:
        """
        JSON'dan kod tablosu yükle.
        
        Args:
            json_bytes: JSON encoded code table
            
        Returns:
            Code table dictionary
        """
        json_dict = json.loads(json_bytes.decode('utf-8'))
        return {int(k): v for k, v in json_dict.items()}


# ============================================================================
# ✅ TEST
# ============================================================================

if __name__ == "__main__":
    print("Huffman Coding Module Test")
    print("=" * 60)
    
    # Test 1: Basit kodlama
    print("\n[TEST 1] Basit Kodlama ve Dekolaması")
    test_values = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    
    encoder = HuffmanEncoder()
    encoder.build_tree_from_frequencies(Counter(test_values))
    
    # Basit encode
    simple_codes = "".join(encoder.code_table[v] for v in test_values)
    print(f"  Orijinal: {test_values}")
    print(f"  Bitstream: {simple_codes[:50]}... ({len(simple_codes)} bit)")
    
    # Decode
    decoder = HuffmanDecoder(encoder.code_table)
    decoded = decoder.decode(simple_codes)
    print(f"  Decoded: {decoded}")
    assert decoded == test_values, "Roundtrip başarısız!"
    print("  ✓ Roundtrip başarılı")
    
    # Test 2: İstatistik
    print("\n[TEST 2] Kod Tablosu İstatistikleri")
    stats = encoder.get_statistics()
    print(f"  Symbol sayısı: {stats['num_symbols']}")
    print(f"  Ortalama kod uzunluğu: {stats['avg_code_length']:.2f} bit")
    print(f"  Entropy: {stats['entropy']:.2f} bit")
    
    # Test 3: Yapay DWT katsayıları
    print("\n[TEST 3] Yapay DWT Katsayıları")
    fake_coeffs = [
        np.random.randn(16, 16).astype(np.float32) * 10,  # LL
        (np.random.randn(16, 16).astype(np.float32) * 5,   # HL
         np.random.randn(16, 16).astype(np.float32) * 5,   # LH
         np.random.randn(16, 16).astype(np.float32) * 5),  # HH
    ]
    
    enc2 = HuffmanEncoder()
    bitstream, code_table = enc2.encode(fake_coeffs)
    print(f"  Bitstream boyutu: {len(bitstream)} bit")
    
    dec2 = HuffmanDecoder(code_table)
    decoded2 = dec2.decode(bitstream)
    print(f"  Decoded symbol sayısı: {len(decoded2)}")
    print("  ✓ Başarılı")
    
    print("\n✅ Tüm testler geçti!")
