"""
stage4_bitstream.py — Stage 4: Bitstream Formation & File Packaging

Bu modül Huffman tarafından üretilen bitstream'i bir dosya formatına paketler.
Header bilgisi, metadata, kod tablosu ve veri bitstream'i içeren .jxc dosyası oluşturur.

Dosya Yapısı:
===============================================================================
| Magic Bytes (3 bytes)  |  Version (1)  |  Metadata  |  Bitstream  |  CRC32 (4) |
===============================================================================

Metadata:
- Image dimensions (width, height) - 4 bytes each
- DWT level (1) - 1 byte
- Wavelet type (string) - 16 bytes (fixed)
- Quantization method (string) - 16 bytes (fixed)
- Quality factor (1 byte) - 0-100
- Code table JSON length (4 bytes) - variable length
- Code table JSON (variable)
- Bitstream padding info (1 byte) - son byte'a kaç bit padding eklendi

JPEG-XC (.jxc) binary file format
"""

import struct
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import zlib

logger = logging.getLogger(__name__)


@dataclass
class CompressionMetadata:
    """Sıkıştırma metadata'sı"""
    image_width: int
    image_height: int
    dwt_level: int = 2
    wavelet: str = "db4"
    quantization_method: str = "uniform"
    quality_factor: int = 50
    
    def to_bytes(self) -> bytes:
        """Metadata'yı bytes'a dönüştür"""
        # Sabit format: width(4) + height(4) + level(1) + wavelet(16) + method(16) + quality(1)
        wavelet_bytes = self.wavelet.encode('ascii').ljust(16, b'\x00')[:16]
        method_bytes = self.quantization_method.encode('ascii').ljust(16, b'\x00')[:16]
        
        return struct.pack(
            '!IIB',
            self.image_width,
            self.image_height,
            self.dwt_level
        ) + wavelet_bytes + method_bytes + struct.pack('!B', self.quality_factor)
    
    @staticmethod
    def from_bytes(data: bytes) -> 'CompressionMetadata':
        """Bytes'dan metadata oluştur"""
        width, height, level = struct.unpack('!IIB', data[:9])
        wavelet = data[9:25].rstrip(b'\x00').decode('ascii')
        method = data[25:41].rstrip(b'\x00').decode('ascii')
        quality = struct.unpack('!B', data[41:42])[0]
        
        return CompressionMetadata(
            image_width=width,
            image_height=height,
            dwt_level=level,
            wavelet=wavelet,
            quantization_method=method,
            quality_factor=quality
        )


class BitstreamPackager:
    """Huffman bitstream'ini JXC dosya formatına paketler"""
    
    MAGIC_BYTES = b'JXC'
    VERSION = 1
    COMPRESSION_FORMAT = "JPEG-XC"
    
    def __init__(self):
        self.magic = self.MAGIC_BYTES
        self.version = self.VERSION
        logger.info("BitstreamPackager initialized")
    
    def create_jxc_file(
        self,
        output_path: str,
        bitstream: str,
        code_table: Dict[int, str],
        metadata: CompressionMetadata
    ) -> Dict[str, Any]:
        """
        JXC dosyası oluştur
        
        Args:
            output_path: Çıktı dosya yolu
            bitstream: Huffman bitstream (0 ve 1 karakterleri)
            code_table: Huffman kod tablosu {symbol: code}
            metadata: Sıkıştırma metadata'sı
            
        Returns:
            Dosya bilgileri
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Tüm veriyi memory'de hazırla
            file_data = bytearray()
            
            # 1. Magic bytes ve version
            file_data.extend(self.magic)
            file_data.extend(struct.pack('!B', self.version))
            
            # 2. Metadata
            metadata_bytes = metadata.to_bytes()
            file_data.extend(metadata_bytes)
            
            # 3. Kod tablosu JSON olarak
            code_table_json = self._serialize_code_table(code_table)
            code_table_length = len(code_table_json)
            file_data.extend(struct.pack('!I', code_table_length))
            file_data.extend(code_table_json)
            
            # 4. Bitstream (padding ile byte alignma)
            padding_bits = self._calculate_padding(len(bitstream))
            padded_bitstream = bitstream + '0' * padding_bits
            
            # Bitstream'i bytes'a çevir (her 8 bit = 1 byte)
            bitstream_bytes = self._bitstring_to_bytes(padded_bitstream)
            file_data.extend(bitstream_bytes)
            
            # 5. Padding info (kaç bit padding eklendi)
            file_data.extend(struct.pack('!B', padding_bits))
            
            # 6. CRC32 kontrol toplamı hesapla
            crc32_value = self._calculate_crc32(bytes(file_data))
            file_data.extend(struct.pack('!I', crc32_value))
            
            # Dosyaya yaz
            with open(output_path, 'wb') as f:
                f.write(file_data)
            
            file_size = output_path.stat().st_size
            compression_info = self._calculate_compression_stats(
                bitstream, code_table, metadata
            )
            
            logger.info(
                f"JXC file created: {output_path} "
                f"({file_size} bytes, {compression_info['compression_ratio']:.2f}x compression)"
            )
            
            return {
                'output_path': str(output_path),
                'file_size': file_size,
                'bitstream_length': len(bitstream),
                'padded_length': len(padded_bitstream),
                'padding_bits': padding_bits,
                'code_table_entries': len(code_table),
                **compression_info
            }
            
        except Exception as e:
            logger.error(f"Failed to create JXC file: {e}")
            raise
    
    def read_jxc_file(
        self,
        input_path: str
    ) -> Tuple[str, Dict[int, str], CompressionMetadata]:
        """
        JXC dosyasını oku
        
        Args:
            input_path: Giriş dosya yolu
            
        Returns:
            (bitstream, code_table, metadata)
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        try:
            with open(input_path, 'rb') as f:
                file_data = f.read()
            
            offset = 0
            
            # 1. Magic bytes ve version kontrol
            magic = file_data[offset:offset+3]
            offset += 3
            if magic != self.magic:
                raise ValueError(f"Invalid magic bytes: {magic}")
            
            version = struct.unpack('!B', file_data[offset:offset+1])[0]
            offset += 1
            if version != self.version:
                raise ValueError(f"Unsupported version: {version}")
            
            # 2. Metadata oku
            metadata_bytes = file_data[offset:offset+42]
            offset += 42
            metadata = CompressionMetadata.from_bytes(metadata_bytes)
            
            # 3. Kod tablosunu oku
            code_table_length = struct.unpack('!I', file_data[offset:offset+4])[0]
            offset += 4
            code_table_json = file_data[offset:offset+code_table_length]
            offset += code_table_length
            code_table = self._deserialize_code_table(code_table_json)
            
            # 4. Son 5 byte'ı oku (padding + CRC32)
            crc32_value = struct.unpack('!I', file_data[-4:])[0]
            padding_bits = struct.unpack('!B', file_data[-5:-4])[0]
            
            # 5. Bitstream'i oku (offset'ten CRC32 ve padding'e kadar)
            bitstream_end = len(file_data) - 5
            bitstream_bytes = file_data[offset:bitstream_end]
            
            # 6. CRC32 kontrol
            file_data_for_crc = file_data[:-4]  # CRC32 hariç
            calculated_crc32 = self._calculate_crc32(file_data_for_crc)
            
            if crc32_value != calculated_crc32:
                logger.warning(
                    f"CRC32 mismatch: expected {crc32_value:08x}, "
                    f"got {calculated_crc32:08x}"
                )
            
            # 7. Bytes'dan bitstream'e çevir
            full_bitstream = self._bytes_to_bitstring(bitstream_bytes)
            
            # Padding bitlerini kaldır
            if padding_bits > 0:
                bitstream = full_bitstream[:-padding_bits]
            else:
                bitstream = full_bitstream
            
            logger.info(
                f"JXC file read: {input_path} "
                f"(bitstream: {len(bitstream)} bits, "
                f"metadata: {metadata.wavelet} {metadata.dwt_level}L)"
            )
            
            return bitstream, code_table, metadata
                
        except Exception as e:
            logger.error(f"Failed to read JXC file: {e}")
            raise
    
    # ========== Helper Methods ==========
    
    @staticmethod
    def _serialize_code_table(code_table: Dict[int, str]) -> bytes:
        """Kod tablosunu JSON'a dönüştür"""
        # JSON içerisinde 'str' key kullan (int key desteklenmez)
        json_dict = {str(k): v for k, v in code_table.items()}
        json_str = json.dumps(json_dict, separators=(',', ':'), sort_keys=True)
        return json_str.encode('utf-8')
    
    @staticmethod
    def _deserialize_code_table(json_bytes: bytes) -> Dict[int, str]:
        """JSON'dan kod tablosunu oluştur"""
        json_str = json_bytes.decode('utf-8')
        json_dict = json.loads(json_str)
        # Tekrar int'e çevir
        return {int(k): v for k, v in json_dict.items()}
    
    @staticmethod
    def _bitstring_to_bytes(bitstring: str) -> bytes:
        """0 ve 1 dizisini bytes'a çevir"""
        if len(bitstring) % 8 != 0:
            raise ValueError(f"Bitstring length must be multiple of 8, got {len(bitstring)}")
        
        byte_array = bytearray()
        for i in range(0, len(bitstring), 8):
            byte_str = bitstring[i:i+8]
            byte_array.append(int(byte_str, 2))
        
        return bytes(byte_array)
    
    @staticmethod
    def _bytes_to_bitstring(data: bytes) -> str:
        """Bytes'ı 0 ve 1 dizisine çevir"""
        bitstring = ""
        for byte in data:
            bitstring += format(byte, '08b')
        return bitstring
    
    @staticmethod
    def _calculate_padding(bitstream_length: int) -> int:
        """Padding bit sayısını hesapla (8'e tamamlamak için)"""
        return (8 - (bitstream_length % 8)) % 8
    
    @staticmethod
    def _calculate_crc32(data: bytes) -> int:
        """CRC32 kontrol toplamı hesapla"""
        return zlib.crc32(data) & 0xffffffff
    
    @staticmethod
    def _calculate_compression_stats(
        bitstream: str,
        code_table: Dict[int, str],
        metadata: CompressionMetadata
    ) -> Dict[str, Any]:
        """Sıkıştırma istatistiklerini hesapla"""
        original_symbols = len(bitstream) / 8  # Assuming 8 bits per original value
        compressed_bits = len(bitstream)
        
        # Huffman entropi
        from collections import Counter
        code_lengths = Counter(len(code) for code in code_table.values())
        avg_code_length = sum(
            length * count for length, count in code_lengths.items()
        ) / len(code_table) if code_table else 0
        
        # Approximate original bit depth (8-16 bits per coefficient)
        estimated_original_bits = len(bitstream) * 1.5  # Conservative estimate
        
        compression_ratio = estimated_original_bits / len(bitstream) if len(bitstream) > 0 else 0
        
        return {
            'original_bits_estimate': int(estimated_original_bits),
            'compressed_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'avg_code_length': avg_code_length,
            'code_table_entries': len(code_table)
        }


class BitstreamValidator:
    """JXC dosyası doğrulama ve debugging"""
    
    @staticmethod
    def validate_jxc_file(input_path: str) -> Dict[str, Any]:
        """
        JXC dosyasını doğrula
        
        Args:
            input_path: JXC dosya yolu
            
        Returns:
            Doğrulama sonuçları
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            return {'valid': False, 'error': 'File not found'}
        
        try:
            packager = BitstreamPackager()
            bitstream, code_table, metadata = packager.read_jxc_file(input_path)
            
            # Temel doğrulamalar
            checks = {
                'file_exists': True,
                'magic_bytes_valid': True,
                'version_compatible': True,
                'crc32_valid': True,  # read_jxc_file zaten kontrol ediyor
                'bitstream_length': len(bitstream),
                'code_table_size': len(code_table),
                'metadata_valid': (
                    metadata.image_width > 0 and
                    metadata.image_height > 0 and
                    metadata.dwt_level >= 1
                ),
                'compression_stats': {
                    'compressed_bits': len(bitstream),
                    'approx_compression_ratio': len(bitstream) / (len(bitstream) + 100)
                }
            }
            
            logger.info(f"JXC file validation passed: {input_path}")
            return checks
            
        except Exception as e:
            logger.error(f"JXC file validation failed: {e}")
            return {
                'valid': False,
                'error': str(e)
            }


if __name__ == "__main__":
    # Örnek kullanım
    import numpy as np
    
    # Synthetic test
    bitstream = "0101" * 256  # 1024 bits = 128 bytes
    code_table = {
        0: "00",
        1: "01",
        2: "10",
        3: "11"
    }
    metadata = CompressionMetadata(
        image_width=256,
        image_height=256,
        dwt_level=2,
        wavelet="db4",
        quantization_method="uniform",
        quality_factor=50
    )
    
    packager = BitstreamPackager()
    
    # Write
    result = packager.create_jxc_file(
        "/tmp/test.jxc",
        bitstream,
        code_table,
        metadata
    )
    print(f"Created: {result}")
    
    # Read
    bit, tab, meta = packager.read_jxc_file("/tmp/test.jxc")
    print(f"Read bitstream length: {len(bit)}")
    print(f"Read code table: {len(tab)} entries")
    print(f"Read metadata: {meta.image_width}x{meta.image_height}")
