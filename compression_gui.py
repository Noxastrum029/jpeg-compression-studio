"""
CENG 384 — Görüntü Sıkıştırma Sistemi (GUI)

Bu modül: CustomTkinter tabanlı masaüstü arayüzü. Sol kontrol paneli, ortada orijinal /
yeniden oluşturulan önizleme, sağda DWT şeması + histogram + metrik tablosu.

Çalıştırma: python compression_gui.py
Bağımlılık: jpeg_jpeg2000_stage1.py (ImagePreprocessor, DWTTransformer)

Hoca/ekip notu: Yeni kontrol eklemek için genelde _build_ui içinde widget oluşturup
ilgili callback'te (ör. _on_param_change) veya _process içinde mantığı bağlayın.
"""

from __future__ import annotations  # İleri referanslı tip ipuçları için (Python 3.7+)

# typing: fonksiyon parametre/dönüş tipleri (IDE ve okunabilirlik için)
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from datetime import datetime
import contextlib
import re
import threading
import traceback

import cv2  # Görüntü yeniden boyutlandırma, JPEG encode/decode, BGR↔YCrCb

cv2.setUseOptimized(True)
import customtkinter as ctk  # Modern Tk arayüz bileşenleri
import numpy as np  # Matris işlemleri, PSNR/MSE
import pywt  # waverec2 — DWT ters dönüşüm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Matplotlib → Tk gömme
from matplotlib.figure import Figure  # Şema ve histogram figürleri
from matplotlib.patches import Rectangle  # DWT 2×2 kutu çizimi
from PIL import Image  # CTkImage için küçük önizleme oluşturma


def _cv2_imencode(
    ext: str,
    img: np.ndarray,
    params: Optional[List[int]] = None,
) -> Tuple[bool, np.ndarray]:
    """OpenCV imencode — Pylance/cv2 stub uyumluluğu için sarmalayıcı."""
    contiguous = np.ascontiguousarray(img)
    if params is None:
        return cv2.imencode(ext, contiguous)  # type: ignore[call-overload, arg-type]
    return cv2.imencode(ext, contiguous, params)  # type: ignore[call-overload, arg-type]


def _cv2_resize(src: np.ndarray, dsize: Tuple[int, int], *, interpolation: int) -> np.ndarray:
    """OpenCV resize — Pylance/cv2 stub uyumluluğu için sarmalayıcı."""
    return cv2.resize(  # type: ignore[call-overload, arg-type, return-value]
        np.ascontiguousarray(src, dtype=np.float64),
        dsize,
        interpolation=interpolation,
    )


def figure_canvas_redraw(canvas: Any) -> None:
    """Matplotlib TkAgg: mümkünse draw_idle (yeniden çizimleri birleştirir), değilse draw."""
    d = getattr(canvas, "draw_idle", None)
    if callable(d):
        d()
    else:
        canvas.draw()


# Aşama 1 modülü: Unicode yol destekli yükleme + DWT sınıfları
from jpeg_jpeg2000_stage1 import DWTTransformer, ImagePreprocessor
from core.pr_qmf import is_pr_qmf, wavedec2_prqmf, waverec2_prqmf, max_level_prqmf
from core.utils import ssim_fn

# Aşama 4 modülü: Huffman entropy kodlama (FR-15 ile FR-20)
from huffman_codec import huffman_compressed_size, encode_coeffs, decode_coeffs

try:
    from core.config import SCALAR_QUANTIZATION_FACTORS as _DWT_SCALAR_FACTORS
except ImportError:
    _DWT_SCALAR_FACTORS = {"LL": 1.0, "HL": 2.0, "LH": 2.0, "HH": 4.0}

# DCT 8×8 frekans bölgeleri — DWT alt bandı (LL/LH/HL/HH) DEĞİL
DCT_ZONE_KEYS: Tuple[str, ...] = ("dc_lf", "horiz_ac", "vert_ac", "diag_ac")
DCT_ZONE_SHORT: Dict[str, str] = {
    "dc_lf": "DC + LF",
    "horiz_ac": "Horiz AC",
    "vert_ac": "Vert AC",
    "diag_ac": "Diag HF AC",
}
DCT_ZONE_FULL: Dict[str, str] = {
    "dc_lf": "DC + Low Frequency Coefficients",
    "horiz_ac": "Horizontal AC Coefficients",
    "vert_ac": "Vertical AC Coefficients",
    "diag_ac": "Diagonal / High Frequency AC Coefficients",
}
DCT_ZONE_VIZ: Dict[str, str] = {
    "dc_lf": "DC + Low Frequency",
    "horiz_ac": "Horizontal AC",
    "vert_ac": "Vertical AC",
    "diag_ac": "Diagonal / High Frequency",
}
DCT_EXPLANATION_TEXT = (
    "DCT does not produce true LL/LH/HL/HH subbands. These names belong to DWT. "
    "In JPEG/DCT, the 8×8 coefficient matrix contains one DC coefficient and multiple AC coefficients. "
    "The coefficients are ordered from low to high frequency using zigzag scan for efficient entropy coding."
)


def _jpeg_zigzag_indices_8x8() -> List[Tuple[int, int]]:
    """JPEG zigzag tarama sırası (8×8 blok, düşük → yüksek frekans)."""
    n = 8
    order: List[Tuple[int, int]] = []
    for diag in range(2 * n - 1):
        coords: List[Tuple[int, int]] = []
        for i in range(n):
            j = diag - i
            if 0 <= j < n:
                coords.append((i, j))
        if diag % 2 == 1:
            coords.reverse()
        order.extend(coords)
    return order


def _jpeg_luminance_quant_matrix(quality: int = 85) -> np.ndarray:
    """JPEG parlaklık kuantalama matrisi (kalite faktörüne göre ölçekli)."""
    try:
        from stages.stage_dct_transform import DCTCompressionPipeline

        return DCTCompressionPipeline(quality=int(np.clip(quality, 1, 100))).get_quantization_matrix()
    except Exception:
        base = np.ones((8, 8), dtype=np.float32)
        scale = max(0.01, (100 - quality) / 50.0) if quality >= 50 else 50.0 / max(1, quality)
        return np.clip(base * scale, 1.0, 255.0)


def _dct_frequency_zone_masks_8x8() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """8×8 DCT düzleminde dört frekans bölgesi (DC+LF, yatay/dikey/çapraz AC)."""
    bs = 8
    dc_lf = np.zeros((bs, bs), dtype=np.float32)
    horiz_ac = np.zeros((bs, bs), dtype=np.float32)
    vert_ac = np.zeros((bs, bs), dtype=np.float32)
    diag_ac = np.zeros((bs, bs), dtype=np.float32)
    for u in range(bs):
        for v in range(bs):
            if u < 4 and v < 4:
                dc_lf[u, v] = 1.0
            elif u >= 4 and v < 4:
                horiz_ac[u, v] = 1.0
            elif u < 4 and v >= 4:
                vert_ac[u, v] = 1.0
            else:
                diag_ac[u, v] = 1.0
    return dc_lf, horiz_ac, vert_ac, diag_ac


def _dct_luma_to_u8(luma_2d: np.ndarray) -> np.ndarray:
    """DCT analizi için tek kanallı luma uint8 [0,255]."""
    g = np.asarray(luma_2d, dtype=np.float32)
    if g.ndim == 3:
        g = g[:, :, 0]
    if g.max() <= 1.5:
        g = g * 255.0
    return np.clip(g, 0, 255).astype(np.uint8)


def _dct_default_block_position(
    h0: int, w0: int, block_size: int = 8
) -> Tuple[int, int]:
    """Varsayılan örnek blok: görüntü merkezindeki 8×8 (ileride tıklama ile değiştirilebilir)."""
    bi = max(0, ((h0 // block_size) // 2) * block_size)
    bj = max(0, ((w0 // block_size) // 2) * block_size)
    bi = min(bi, max(0, h0 - block_size))
    bj = min(bj, max(0, w0 - block_size))
    return bi, bj


def _dct_sample_block_centered(
    luma_2d: np.ndarray,
    block_size: int = 8,
    block_pos: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Seçilen 8×8 bloğu döndür: ham piksel (uint8), merkezlenmiş [-128,127], (bi, bj).
    block_pos verilmezse merkez blok kullanılır.
    """
    g_u8 = _dct_luma_to_u8(luma_2d)
    h0, w0 = g_u8.shape
    if block_pos is None:
        bi, bj = _dct_default_block_position(h0, w0, block_size)
    else:
        bi, bj = block_pos
        bi = int(np.clip(bi, 0, max(0, h0 - block_size)))
        bj = int(np.clip(bj, 0, max(0, w0 - block_size)))
        bi = (bi // block_size) * block_size
        bj = (bj // block_size) * block_size
    block_u8 = g_u8[bi : bi + block_size, bj : bj + block_size]
    centered = block_u8.astype(np.float32) - 128.0
    return block_u8, centered.astype(np.float32), (bi, bj)


def _dct_image_with_block_highlight(
    luma_u8: np.ndarray,
    bi: int,
    bj: int,
    block_size: int = 8,
    max_side: int = 200,
) -> np.ndarray:
    """Orijinal görüntü üzerinde seçili 8×8 bloğu kırmızı kutu ile işaretle."""
    rgb = cv2.cvtColor(luma_u8, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(
        rgb,
        (bj, bi),
        (bj + block_size - 1, bi + block_size - 1),
        (255, 60, 40),
        2,
        lineType=cv2.LINE_8,
    )
    h, w = rgb.shape[:2]
    scale = min(max_side / max(h, 1), max_side / max(w, 1))
    if scale < 1.0:
        tw, th = max(32, int(w * scale)), max(32, int(h * scale))
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
    return rgb


def _annotate_8x8_matrix_rgb(
    values: np.ndarray,
    *,
    cell_px: int = 28,
    highlight_rc: Optional[Tuple[int, int]] = None,
    zero_gray: bool = False,
    value_fmt: Optional[str] = None,
) -> np.ndarray:
    """8×8 matris: hücre rengi + kısa sayı etiketi."""
    n = 8
    vals = np.asarray(values, dtype=np.float64)
    rgb = np.full((n * cell_px, n * cell_px, 3), 255, dtype=np.uint8)
    vmax = float(np.max(np.abs(vals))) or 1.0
    for u in range(n):
        for v in range(n):
            val = float(vals[u, v])
            y0, x0 = u * cell_px, v * cell_px
            if zero_gray and abs(val) < 0.5:
                fill = (235, 235, 235)
                text_color = (160, 160, 160)
            else:
                t = min(1.0, abs(val) / vmax)
                fill = (int(255 - 180 * t), int(240 - 120 * t), int(255 - 60 * t))
                text_color = (30, 30, 30)
            cv2.rectangle(rgb, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1), fill, -1)
            if highlight_rc == (u, v):
                cv2.rectangle(rgb, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1), (220, 40, 40), 2)
            if value_fmt == "int":
                label = f"{int(round(val))}"
            elif value_fmt == "pixel":
                label = f"{int(round(val))}"
            else:
                label = f"{val:.0f}" if abs(val) >= 10 else f"{val:.1f}"
            if len(label) > 5:
                label = label[:5]
            cv2.putText(
                rgb,
                label,
                (x0 + 2, y0 + cell_px - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            text_color,
            1,
            lineType=cv2.LINE_AA,
            )
    return rgb


def _dct_zigzag_result_rgb(quant_block: np.ndarray, width: int = 280, height: int = 120) -> np.ndarray:
    """Zigzag sonrası 1D katsayı dizisi — düşük frekans (sol) vs yüksek/sıfır (sağ)."""
    order = _jpeg_zigzag_indices_8x8()
    q = np.asarray(quant_block, dtype=np.float64)
    vals = np.array([q[u, v] for u, v in order], dtype=np.float64)
    rgb = np.full((height, width, 3), 248, dtype=np.uint8)
    n = vals.size
    bar_w = max(2, width // n)
    max_abs = float(np.max(np.abs(vals))) or 1.0
    base_y = height - 14
    for i, v in enumerate(vals):
        x0 = i * bar_w
        bar_h = int(abs(v) / max_abs * (height - 28))
        if i == 0:
            color = (220, 50, 40)
        elif i < 10:
            color = (230, 120, 50)
        elif abs(v) < 0.5:
            color = (210, 210, 210)
        else:
            color = (80, 130, 200)
        cv2.rectangle(rgb, (x0, base_y - bar_h), (x0 + bar_w - 1, base_y), color, -1)
    cv2.putText(rgb, "DC+low freq", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 50, 40), 1)
    cv2.putText(
        rgb,
        "high freq / zeros",
        (width - 108, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (80, 100, 160),
        1,
    )
    nz = int(np.count_nonzero(np.abs(vals) >= 0.5))
    cv2.putText(
        rgb,
        f"non-zero: {nz}/64",
        (4, height - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (60, 60, 60),
        1,
    )
    return rgb


JPEG_BLOCK_PIPELINE_TEXT = (
    "Selected 8×8 Block  →  DCT  →  Quantization  →  Zigzag Scan  →  JPEG Bitstream"
)


def _dct_coeff_matrix_heatmap_u8(matrix: np.ndarray, *, log_scale: bool = True) -> np.ndarray:
    """8×8 DCT/kuantize matrisini RGB ısı haritası olarak döndür."""
    m = np.asarray(matrix, dtype=np.float64)
    if log_scale:
        mag = np.log(np.abs(m) + 1.0)
    else:
        mag = np.abs(m)
    mx = float(mag.max()) or 1.0
    norm = np.clip(mag / mx, 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    rgb = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def _dct_zigzag_order_rgb(block_size: int = 8, cell_px: int = 22) -> np.ndarray:
    """8×8 zigzag tarama sırasını numaralı ızgara olarak çizer."""
    order = _jpeg_zigzag_indices_8x8()
    n = block_size
    img_h = n * cell_px
    img_w = n * cell_px
    rgb = np.full((img_h, img_w, 3), 245, dtype=np.uint8)
    for step, (u, v) in enumerate(order, start=1):
        y0, x0 = u * cell_px, v * cell_px
        y1, x1 = y0 + cell_px, x0 + cell_px
        color = (30, 120, 200) if step == 1 else (60, 60, 60)
        cv2.rectangle(rgb, (x0, y0), (x1 - 1, y1 - 1), color, 1)
        cv2.putText(
            rgb,
            str(step),
            (x0 + 4, y0 + cell_px - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 20, 20) if step > 1 else (200, 40, 40),
            1,
            lineType=cv2.LINE_AA,
        )
    return rgb


def _dct_zone_layer_images(gray: np.ndarray, block_size: int = 8) -> Tuple[List[np.ndarray], List[float]]:
    """
    Her 8×8 blokta DCT → frekans bölgelerine maske → IDCT ile uzamsal katman görüntüleri.
    Dönüş: ([dc_lf, horiz_ac, vert_ac, diag_ac] görselleştirme [0,1], enerji yüzdesi).
    """
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim == 3:
        g = g[:, :, 0]
    if g.max() > 1.5:
        g = g / 255.0
    h0, w0 = g.shape
    ph = ((h0 + block_size - 1) // block_size) * block_size
    pw = ((w0 + block_size - 1) // block_size) * block_size
    padded = np.zeros((ph, pw), dtype=np.float32)
    padded[:h0, :w0] = g

    masks = _dct_frequency_zone_masks_8x8()
    acc = [np.zeros((ph, pw), dtype=np.float32) for _ in range(4)]
    energy = [0.0, 0.0, 0.0, 0.0]

    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            blk = padded[i : i + block_size, j : j + block_size]
            d = cv2.dct(blk)
            for zi, m in enumerate(masks):
                zm = d * m
                energy[zi] += float(np.sum(zm ** 2))
                acc[zi][i : i + block_size, j : j + block_size] += cv2.idct(zm)

    total_e = sum(energy)
    if total_e < 1e-18:
        pct = [25.0, 25.0, 25.0, 25.0]
    else:
        pct = [100.0 * e / total_e for e in energy]

    out_vis: List[np.ndarray] = []
    for arr in acc:
        a = np.abs(arr[:h0, :w0])
        mx = float(a.max())
        out_vis.append(a / mx if mx > 1e-9 else np.zeros_like(a))
    return out_vis, pct


def _dct_is_nearly_flat_luma(luma_u8: np.ndarray, variance_threshold: float = 1.0) -> bool:
    """Piksel varyansı çok düşükse görüntü sabit/katı kabul edilir (AC ≈ 0 beklenir)."""
    if luma_u8.size == 0:
        return True
    return float(np.var(luma_u8.astype(np.float64))) < variance_threshold


def _dct_per_zone_coeff_samples(gray: np.ndarray, block_size: int = 8) -> List[np.ndarray]:
    """
    8×8 JPEG-style DCT histogram örnekleri: Y/luma uint8 → blok başına -128 → cv2.dct.
    Kenar dolgusu sıfır değil edge-replicate (sahte AC kenar artefaktını önler).
    """
    g_u8 = _dct_luma_to_u8(gray)
    h0, w0 = g_u8.shape
    ph = ((h0 + block_size - 1) // block_size) * block_size
    pw = ((w0 + block_size - 1) // block_size) * block_size
    if ph > h0 or pw > w0:
        padded = cv2.copyMakeBorder(
            g_u8, 0, ph - h0, 0, pw - w0, borderType=cv2.BORDER_REPLICATE
        )
    else:
        padded = g_u8

    is_flat = _dct_is_nearly_flat_luma(padded)
    masks = _dct_frequency_zone_masks_8x8()
    buckets: List[List[np.ndarray]] = [[], [], [], []]

    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            blk = padded[i : i + block_size, j : j + block_size].astype(np.float32) - 128.0
            d = cv2.dct(blk)
            for zi, m in enumerate(masks):
                vals = d[m > 0.5]
                if vals.size:
                    buckets[zi].append(vals.astype(np.float64))

    out: List[np.ndarray] = []
    for zi, b in enumerate(buckets):
        if is_flat and zi > 0:
            out.append(np.array([0.0], dtype=np.float64))
            continue
        if not b:
            out.append(np.array([0.0], dtype=np.float64))
            continue
        arr = np.concatenate(b).astype(np.float64)
        if zi > 0:
            arr = arr[np.abs(arr) > 1e-6]
            if arr.size == 0:
                arr = np.array([0.0], dtype=np.float64)
        out.append(arr)
    return out


def _dwt_single_level_four_bands(luma: np.ndarray, wavelet: str) -> Tuple[List[np.ndarray], List[float]]:
    """Tek seviye DWT: LL, HL, LH, HH — pywt (cH=LH, cV=HL, cD=HH) JPEG2000 sırası."""
    data = np.asarray(luma, dtype=np.float64)
    if data.ndim == 3:
        data = data[:, :, 0]
    if is_pr_qmf(wavelet):
        from core.pr_qmf import dwt2_prqmf
        LL, (cH, cV, cD) = dwt2_prqmf(data, wavelet)
    else:
        LL, (cH, cV, cD) = pywt.dwt2(data, wavelet, mode="periodization")
    # pywt: cH=horizontal/LH, cV=vertical/HL, cD=diagonal/HH
    bands = [LL, cV, cH, cD]
    energies = [float(np.sum(b ** 2)) for b in bands]
    s = sum(energies)
    if s < 1e-18:
        pct = [25.0, 25.0, 25.0, 25.0]
    else:
        pct = [100.0 * e / s for e in energies]
    vis: List[np.ndarray] = []
    for b in bands:
        bb = np.abs(np.asarray(b, dtype=np.float64))
        mx = float(bb.max())
        vis.append(bb / mx if mx > 1e-9 else np.zeros_like(bb))
    return vis, pct


# --- Çok seviyeli DWT mosaic (eğitimsel Subband Structure paneli) ---
_DWT_BAND_COLORS: Dict[str, str] = {
    "LL": "#E3F2FD",
    "HL": "#E0F2F1",
    "LH": "#FFF8E1",
    "HH": "#FFEBEE",
}


def _normalize_ll_band_for_display(band: np.ndarray) -> np.ndarray:
    """LL bant: min-max → [0, 1] (yalnızca görselleştirme)."""
    arr = np.asarray(band, dtype=np.float64)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mn) / (mx - mn)


def _normalize_detail_band_for_display(band: np.ndarray) -> np.ndarray:
    """Detay bant: log1p(|x|) + min-max → [0, 1] (yalnızca görselleştirme)."""
    arr = np.log1p(np.abs(np.asarray(band, dtype=np.float64)))
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mn) / (mx - mn)


def _luma_to_2d_for_dwt(luma: np.ndarray) -> np.ndarray:
    """DWT görselleştirme / analiz için tek kanallı 2B dizi."""
    arr = np.asarray(luma, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            u8 = np.clip(arr, 0, 255).astype(np.uint8)
            return cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY).astype(np.float64)
        return arr[:, :, 0]
    return arr


def _ensure_dwt_coeff_list(
    coeffs: Any,
    luma_2d: np.ndarray,
    wavelet: str = "db4",
    level: int = 2,
) -> List:
    """
    Görselleştirme için geçerli wavedec2 katsayı listesi döndür.
    dict (3 kanallı legacy) veya bozuk girişte luma üzerinden yeniden hesaplar.
    """
    if isinstance(coeffs, (list, tuple)) and len(coeffs) >= 2:
        if isinstance(coeffs[0], np.ndarray):
            return list(coeffs)
    data = _luma_to_2d_for_dwt(luma_2d)
    if is_pr_qmf(wavelet):
        max_level = max_level_prqmf(data.shape, wavelet)
        use_level = max(1, min(int(level), max_level))
        return wavedec2_prqmf(data, wavelet, level=use_level)
    else:
        max_level = pywt.dwtn_max_level(data.shape, wavelet)
        use_level = max(1, min(int(level), max_level))
        return pywt.wavedec2(data, wavelet, level=use_level, mode="periodization")


def _slice2d_to_bbox(row_slice: slice, col_slice: slice) -> Tuple[int, int, int, int]:
    y0 = 0 if row_slice.start is None else int(row_slice.start)
    y1 = int(row_slice.stop)
    x0 = 0 if col_slice.start is None else int(col_slice.start)
    x1 = int(col_slice.stop)
    return x0, y0, x1 - x0, y1 - y0


def _build_dwt_multilevel_mosaic(
    coeffs: List,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    pywt.wavedec2 katsayılarından çok seviyeli subband mosaic oluştur.

    PyWavelets coeffs_to_array yerleşimini kullanır; böylece 512×512 dışındaki
    boyutlarda da (periodization) bantlar doğru hizalanır.

    Etiket: LH=cH (ad), HL=cV (da), HH=cD (dd) — pywt yerleşimi + proje eşlemesi.
    """
    if not coeffs or len(coeffs) < 2:
        raise ValueError("Geçersiz DWT katsayı listesi")

    n_levels = len(coeffs) - 1
    norm_coeffs: List[Any] = [_normalize_ll_band_for_display(coeffs[0])]
    for cH, cV, cD in coeffs[1:]:
        norm_coeffs.append(
            (
                _normalize_detail_band_for_display(cH),
                _normalize_detail_band_for_display(cV),
                _normalize_detail_band_for_display(cD),
            )
        )

    mosaic, coeff_slices = pywt.coeffs_to_array(norm_coeffs, padding=0)
    regions: List[Dict[str, Any]] = []

    ll_sl = coeff_slices[0]
    if isinstance(ll_sl, tuple) and len(ll_sl) >= 2:
        x, y, w, h = _slice2d_to_bbox(ll_sl[0], ll_sl[1])
        regions.append(
            {
                "label": f"LL{n_levels}",
                "band": "LL",
                "level": n_levels,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )

    detail_keys = (("ad", "LH"), ("da", "HL"), ("dd", "HH"))
    for i in range(1, len(coeff_slices)):
        level_num = n_levels - i + 1
        sl_dict_raw = coeff_slices[i]
        if not isinstance(sl_dict_raw, dict):
            continue
        sl_dict = cast(Dict[str, Any], sl_dict_raw)
        for py_key, band_key in detail_keys:
            if py_key not in sl_dict:
                continue
            row_sl, col_sl = sl_dict[py_key]
            x, y, w, h = _slice2d_to_bbox(row_sl, col_sl)
            regions.append(
                {
                    "label": f"{band_key}{level_num}",
                    "band": band_key,
                    "level": level_num,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )

    return mosaic.astype(np.float64), regions


def _draw_dwt_schematic_pyramid(
    ax: Any,
    n_levels: int,
    x0: float,
    y0: float,
    size: float,
    split_idx: int,
) -> None:
    """Katsayı yokken iç içe 2×2 şema (ders slaytı mantığı)."""
    if split_idx > n_levels:
        r = Rectangle(
            (x0, y0),
            size,
            size,
            linewidth=1.5,
            edgecolor=TEXT_PRIMARY,
            facecolor=_DWT_BAND_COLORS["LL"],
            alpha=0.85,
        )
        ax.add_patch(r)
        ax.text(
            x0 + size / 2,
            y0 + size / 2,
            f"LL{n_levels}",
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            color=TEXT_PRIMARY,
        )
        return

    half = size / 2.0
    level_num = n_levels - split_idx + 1
    _draw_dwt_schematic_pyramid(ax, n_levels, x0, y0, half, split_idx + 1)

    boxes = (
        (x0 + half, y0, "HL", f"HL{level_num}"),
        (x0, y0 + half, "LH", f"LH{level_num}"),
        (x0 + half, y0 + half, "HH", f"HH{level_num}"),
    )
    for bx, by, band_key, lab in boxes:
        r = Rectangle(
            (bx, by),
            half,
            half,
            linewidth=1.2,
            edgecolor=TEXT_PRIMARY,
            facecolor=_DWT_BAND_COLORS[band_key],
            alpha=0.85,
        )
        ax.add_patch(r)
        ax.text(
            bx + half / 2,
            by + half / 2,
            lab,
            ha="center",
            va="center",
            fontsize=8 if n_levels > 1 else 9,
            weight="bold",
            color=TEXT_PRIMARY,
        )


def _luma_to_u8(luma: np.ndarray) -> np.ndarray:
    """Karşılaştırma / kayıt için tek kanal uint8 luma (0–1 veya 0–255 giriş)."""
    a = np.asarray(luma, dtype=np.float64)
    if a.ndim == 3:
        a = a[:, :, 0]
    if np.nanmax(a) <= 1.5:
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def _dct_8x8_block_mosaic(gray_u8: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Her 8×8 blokta |DCT| katsayılarını blok içinde normalize ederek gösterir (klasik blok mozaik)."""
    g = gray_u8.astype(np.float32)
    h0, w0 = g.shape
    ph = ((h0 + block_size - 1) // block_size) * block_size
    pw = ((w0 + block_size - 1) // block_size) * block_size
    padded = np.zeros((ph, pw), dtype=np.float32)
    padded[:h0, :w0] = g
    out = np.zeros_like(padded)
    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            blk = padded[i : i + block_size, j : j + block_size]
            d = cv2.dct(blk)
            ad = np.abs(d)
            mx = float(ad.max()) or 1.0
            out[i : i + block_size, j : j + block_size] = ad / mx
    return out[:h0, :w0]


def _dct_8x8_block_mosaic_rgb(gray_u8: np.ndarray, block_size: int = 8) -> np.ndarray:
    """8×8 blok mozaiği + sarı ızgara çizgileri (blok sınırları görünür)."""
    mosaic = _dct_8x8_block_mosaic(gray_u8, block_size)
    vis = (np.clip(mosaic, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    h, w = vis.shape
    grid_color = (255, 220, 0)
    for x in range(block_size, w, block_size):
        cv2.line(rgb, (x, 0), (x, h - 1), grid_color, 1, lineType=cv2.LINE_8)
    for y in range(block_size, h, block_size):
        cv2.line(rgb, (0, y), (w - 1, y), grid_color, 1, lineType=cv2.LINE_8)
    return rgb


class DWTLayerViewer(ctk.CTkToplevel):
    """
    Displays DWT sub-bands (LL, LH, HL, HH) for all levels 
    in a detailed separate scrollable window.
    """
    def __init__(self, parent, coeffs, wavelet_name):
        super().__init__(parent)
        self.title(f"DWT Decomposition Layers - {wavelet_name}")
        self.geometry("1100x850")
        self.coeffs = coeffs
        
        # Main scrollable frame to handle multi-level displays
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create Matplotlib Figure
        num_levels = len(coeffs) - 1
        # Rows = num_levels + 1 (for LL), Cols = 3 (for LH, HL, HH)
        fig = Figure(figsize=(12, 4 * (num_levels + 1)), dpi=100, facecolor='#2b2b2b')
        
        # 1. Plot the LL (Base Approximation) at the top center
        ax_ll = fig.add_subplot(num_levels + 1, 3, 2)
        ax_ll.imshow(self.coeffs[0], cmap='gray')
        ax_ll.set_title("Base Approximation (LL)", color='white', fontsize=11)
        ax_ll.axis('off')

        # 2. Plot Detail Bands (LH, HL, HH) for each level
        for i in range(1, len(self.coeffs)):
            level_label = len(self.coeffs) - i
            cH, cV, cD = self.coeffs[i]
            
            bands = [("Horizontal (LH)", cH), ("Vertical (HL)", cV), ("Diagonal (HH)", cD)]
            
            for j, (name, data) in enumerate(bands):
                ax = fig.add_subplot(num_levels + 1, 3, (i * 3) + j + 1)
                # Use 'magma' colormap to highlight high-frequency edge details
                ax.imshow(np.abs(data), cmap='magma')
                ax.set_title(f"Level {level_label}: {name}", color='white', fontsize=10)
                ax.axis('off')

        fig.tight_layout(pad=3.0)
        
        # Embed the matplotlib figure into the CustomTkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.scroll_frame)
        figure_canvas_redraw(canvas)
        canvas.get_tk_widget().pack(fill="both", expand=True)


class WaveletPreviewWindow(ctk.CTkToplevel):
    """JPEG DCT modunda, aynı hedefe yaklaşık DWT yeniden kurulumunu ayrı pencerede gösterir."""

    def __init__(self, parent, preview_rgb, actual_bpp, psnr):
        super().__init__(parent)
        self.title("Wavelet (DWT) Önizleme")
        self.geometry("500x600")
        self.minsize(400, 500)

        header = ctk.CTkLabel(
            self, text="Wavelet Önizleme (yaklaşık aynı hedef)",
            font=FONT_SECTION,
        )
        header.pack(pady=(16, 8))

        img_label = ctk.CTkLabel(self, text="")
        img_label.pack(pady=8, padx=16)
        rgb = preview_rgb
        if rgb.dtype == np.float32 or rgb.dtype == np.float64:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        elif rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        pil = Image.fromarray(rgb)
        pil.thumbnail((420, 420), Image.Resampling.LANCZOS)
        w, h = pil.size
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(w, h))
        img_label.configure(image=ctk_img, text="")
        img_label._image = ctk_img
        self._preview_image = ctk_img

        caption = ctk.CTkLabel(
            self, text=f"BPP≈{actual_bpp:.2f}, PSNR≈{psnr:.1f} dB",
            font=FONT_BODY,
        )
        caption.pack(pady=8)

        btn_close = ctk.CTkButton(self, text="Kapat", command=self.destroy)
        btn_close.pack(pady=16)

        self.transient(parent)
        self.after(20, self._bring_to_front)

    def _bring_to_front(self):
        try:
            self.lift()
            self.focus_force()
            self.attributes("-topmost", True)
            self.after(200, lambda: self.attributes("-topmost", False))
        except Exception:
            pass


class ComparisonWindow(ctk.CTkToplevel):
    """
    JPEG (DCT) vs JPEG2000 (DWT) karşılaştırma penceresi.
    Her iki yöntemin metriklerini yan yana gösterir; isteğe bağlı olarak DWT/DCT katman görselleri.
    """
    def __init__(
        self,
        parent,
        results,
        image_type,
        luma_2d: Optional[np.ndarray] = None,
        wavelet: str = "db4",
        level: int = 2,
        hardness: float = 50.0,
    ):
        super().__init__(parent)
        self.title(f"Sıkıştırma Karşılaştırması: JPEG vs JPEG2000 ({image_type})")
        self.geometry("1080x840")
        self.minsize(880, 560)
        self.configure(fg_color="#F4F7FA")

        # Alt: sabit kapat çubuğu — üst: kaydırılabilir tüm içerik (görsel sıkışmasını önler)
        bottom_bar = ctk.CTkFrame(self, fg_color="#ECEFF1", corner_radius=0, height=56)
        bottom_bar.pack(side="bottom", fill="x")
        bottom_bar.pack_propagate(False)
        ctk.CTkButton(
            bottom_bar,
            text="Kapat",
            command=self.destroy,
            width=140,
            height=36,
            fg_color="#78909c",
            hover_color="#607d8b",
            corner_radius=8,
        ).pack(pady=10)

        scroll_body = ctk.CTkScrollableFrame(
            self,
            fg_color="#F4F7FA",
            scrollbar_button_color="#90A4AE",
            scrollbar_button_hover_color="#546E7A",
        )
        scroll_body.pack(fill="both", expand=True, padx=8, pady=(10, 4))

        # Başlık
        ctk.CTkLabel(
            scroll_body,
            text="📊 JPEG (DCT) vs JPEG2000 (DWT) Analizi",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#1a237e",
        ).pack(pady=(12, 10))

        # Görüntü Türü Bilgisi
        ctk.CTkLabel(
            scroll_body,
            text=f"Algılanan Görüntü Türü: {image_type.capitalize()}",
            font=ctk.CTkFont(size=14),
            text_color="#546e7a",
        ).pack(pady=(0, 16))

        # Tablo Alanı
        self.table_frame = ctk.CTkFrame(
            scroll_body, fg_color="white", corner_radius=12, border_width=1, border_color="#CFD8DC"
        )
        self.table_frame.pack(fill="x", padx=24, pady=10)
        
        # Grid Yapılandırması
        self.table_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        headers = ["Metrik", "JPEG (DCT)", "JPEG2000 (DWT)"]
        for i, h in enumerate(headers):
            lbl = ctk.CTkLabel(
                self.table_frame, 
                text=h, 
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color="#1e88e5"
            )
            lbl.grid(row=0, column=i, padx=20, pady=15, sticky="nsew")
            
        metrics_to_show = [
            ("PSNR (Kalite - dB)", "psnr"),
            ("SSIM (Benzerlik)", "ssim"),
            ("Sıkıştırma Oranı", "compression_ratio"),
            ("BPP (Bit Per Pixel)", "bpp"),
            ("İşlem Süresi (ms)", "compression_time_ms")
        ]
        
        # Sonuçları filtrele (Örn: Q85 seviyesindeki sonuçları alalım)
        dct_res = next((r for r in results if r.method == "DCT" and r.quality == 85), None)
        dwt_res = next((r for r in results if r.method == "DWT" and r.quality == 85), None)
        
        # Eğer Q85 yoksa herhangi birini al
        if not dct_res: dct_res = next((r for r in results if r.method == "DCT"), None)
        if not dwt_res: dwt_res = next((r for r in results if r.method == "DWT"), None)
        
        for i, (name, key) in enumerate(metrics_to_show):
            # Metrik İsmi
            ctk.CTkLabel(self.table_frame, text=name, font=ctk.CTkFont(weight="bold")).grid(row=i+1, column=0, padx=20, pady=10, sticky="w")
            
            # Değerler
            val_dct = getattr(dct_res, key, 0) if dct_res else "—"
            val_dwt = getattr(dwt_res, key, 0) if dwt_res else "—"
            
            # Formatlama
            if isinstance(val_dct, float):
                if val_dct == -1.0: val_dct = "∞ (Kayıpsız)"
                else: val_dct = f"{val_dct:.2f}"
            if isinstance(val_dwt, float):
                if val_dwt == -1.0: val_dwt = "∞ (Kayıpsız)"
                else: val_dwt = f"{val_dwt:.2f}"
            
            ctk.CTkLabel(self.table_frame, text=str(val_dct)).grid(row=i+1, column=1, padx=20, pady=10)
            ctk.CTkLabel(self.table_frame, text=str(val_dwt)).grid(row=i+1, column=2, padx=20, pady=10)

        # --- Katman karşılaştırması (scroll içinde; matplotlib yüksekliği doğal kalır) ---
        if luma_2d is not None:
            layer_outer = ctk.CTkFrame(scroll_body, fg_color="transparent")
            layer_outer.pack(fill="x", padx=16, pady=(8, 14))

            layer_frame = ctk.CTkFrame(layer_outer, fg_color="white", corner_radius=14, border_width=1, border_color="#B0BEC5")
            layer_frame.pack(fill="x")

            ctk.CTkLabel(
                layer_frame,
                text="Frekans yapısı: DWT ↔ DCT",
                font=ctk.CTkFont(size=17, weight="bold"),
                text_color="#0d47a1",
            ).pack(anchor="w", padx=18, pady=(16, 4))
            ctk.CTkLabel(
                layer_frame,
                text=(
                    "Üst satır: JPEG ve DWT ile yeniden kurulmuş görüntüler (boyut ve PSNR). "
                    "Alt satır: solda 8×8 DCT blok mozaiği, sağda dalgacık katsayı haritası (log ölçek). "
                    f"Enerji tablosu tek seviye özet içindir; DWT yeniden kurulum dalgacık {wavelet}, seviye L{level} ile ana paneldeki ayarlarla uyumludur."
                ),
                font=ctk.CTkFont(size=12),
                text_color="#455a64",
                wraplength=900,
                justify="left",
            ).pack(anchor="w", padx=18, pady=(0, 12))

            try:
                _, dwt_energy_pct = _dwt_single_level_four_bands(luma_2d, wavelet)
                _, dct_energy_pct = _dct_zone_layer_images(luma_2d)
                band_short = ["LL", "HL", "LH", "HH"]
                band_long = [
                    "Yaklaşım",
                    "Yatay detay",
                    "Dikey detay",
                    "Çapraz detay",
                ]

                energy_card = ctk.CTkFrame(layer_frame, fg_color="#ECEFF1", corner_radius=10)
                energy_card.pack(fill="x", padx=14, pady=(0, 12))
                ctk.CTkLabel(
                    energy_card,
                    text="Enerji dağılımı (%) — aynı dört rol için iki yöntem",
                    font=ctk.CTkFont(size=13, weight="bold"),
                    text_color="#263238",
                ).pack(anchor="w", padx=12, pady=(10, 6))
                energy_grid = ctk.CTkFrame(energy_card, fg_color="transparent")
                energy_grid.pack(fill="x", padx=8, pady=(0, 10))
                energy_grid.grid_columnconfigure(0, weight=3, minsize=220)
                energy_grid.grid_columnconfigure(1, weight=1, minsize=100)
                energy_grid.grid_columnconfigure(2, weight=1, minsize=100)

                hdr_fg = "#37474f"
                ctk.CTkLabel(energy_grid, text="Bant adı", font=ctk.CTkFont(size=12, weight="bold"), text_color=hdr_fg).grid(
                    row=0, column=0, padx=10, pady=6, sticky="w"
                )
                ctk.CTkLabel(
                    energy_grid,
                    text="DWT enerji %",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color="#4527a0",
                ).grid(row=0, column=1, padx=10, pady=6, sticky="ew")
                ctk.CTkLabel(
                    energy_grid,
                    text="DCT bölge %",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color="#e65100",
                ).grid(row=0, column=2, padx=10, pady=6, sticky="ew")

                for r, short, long_name, de, ce in zip(
                    range(1, 5), band_short, band_long, dwt_energy_pct, dct_energy_pct
                ):
                    ctk.CTkLabel(
                        energy_grid,
                        text=f"{short} — {long_name}",
                        font=ctk.CTkFont(size=12),
                        anchor="w",
                        text_color="#263238",
                    ).grid(row=r, column=0, padx=12, pady=7, sticky="w")
                    ctk.CTkLabel(
                        energy_grid,
                        text=f"{de:.1f}",
                        font=ctk.CTkFont(size=12, weight="bold"),
                        text_color="#4527a0",
                    ).grid(row=r, column=1, padx=10, pady=7, sticky="")
                    ctk.CTkLabel(
                        energy_grid,
                        text=f"{ce:.1f}",
                        font=ctk.CTkFont(size=12, weight="bold"),
                        text_color="#e65100",
                    ).grid(row=r, column=2, padx=10, pady=7, sticky="")

                # ——— Tek büyük 2×2 panel: üst sıra yeniden kurulumlar, alt sıra DCT blokları / DWT katsayı haritası ———
                luma_u8 = _luma_to_u8(luma_2d)
                # DWT kuanta Δ (≈0.5–30.5), [0,1] katsayılarıyla uyumsuzdu → tüm bantlar 0’a yuvarlanıyordu.
                # Gri tonlamayı 0–255 float ile besleyin; yeniden kurulum ve PSNR ana pencere ile tutarlı olur.
                luma_for_dwt = luma_u8.astype(np.float64)

                jpeg_qual = 85  # üst metrik tablosu ile aynı referans (Q85)
                ok_j, enc_j = _cv2_imencode(".jpg", luma_u8, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_qual])
                jpeg_gray = cv2.imdecode(enc_j, cv2.IMREAD_GRAYSCALE) if ok_j else luma_u8.copy()
                jpeg_gray = jpeg_gray if jpeg_gray is not None else luma_u8.copy()
                jpeg_kb = float(enc_j.nbytes) / 1024.0 if ok_j else 0.0
                psnr_j = psnr_fn(luma_u8.astype(np.float64), jpeg_gray.astype(np.float64))

                # Hedef BPP'ye (hardness) ulaşmak için doğru q_step'i bul
                quant_mode = getattr(parent, "_quant_mode", "scalar")
                q_step = parent._find_best_dwt_q_step(
                    float(hardness), luma_for_dwt.size, wavelet, level, mode=quant_mode
                )
                dwt_tr = DWTTransformer(wavelet_name=wavelet, level=level)
                coeffs_raw = dwt_tr.apply_dwt2d(luma_for_dwt)
                q_coeffs = quantize_coeffs(coeffs_raw, q_step, mode=quant_mode)
                if is_pr_qmf(wavelet):
                    recon_w = waverec2_prqmf(q_coeffs, wavelet)
                else:
                    recon_w = pywt.waverec2(q_coeffs, wavelet, mode="periodization")
                recon_w = recon_w[: luma_u8.shape[0], : luma_u8.shape[1]]
                dwt_gray = np.clip(recon_w, 0, 255).astype(np.uint8)
                try:
                    from huffman_codec import huffman_compressed_size as _huff_sz

                    _orig_b, _comp_b, _sav = _huff_sz(q_coeffs)
                    dwt_kb = max(0.0, float(_comp_b)) / 1024.0
                except Exception:
                    dwt_kb = float(luma_u8.nbytes) / 1024.0
                psnr_w = psnr_fn(luma_u8.astype(np.float64), dwt_gray.astype(np.float64))

                dct_mosaic = _dct_8x8_block_mosaic(luma_u8)
                dwt_vis_bands, _ = _dwt_single_level_four_bands(luma_for_dwt, wavelet)
                dwt_band_titles = (
                    "LL (Yaklaşım)",
                    "HL (Dikey)",
                    "LH (Yatay)",
                    "HH (Çapraz)",
                )

                fig_cmp = Figure(figsize=(14.5, 12.5), dpi=105, facecolor="white")
                gs = fig_cmp.add_gridspec(2, 2, hspace=0.33, wspace=0.22, left=0.05, right=0.98, top=0.93, bottom=0.05)

                ax_ul = fig_cmp.add_subplot(gs[0, 0])
                ax_ul.imshow(jpeg_gray, cmap="gray", vmin=0, vmax=255)
                ax_ul.set_title(
                    f"JPEG (DCT) — yeniden kurulum\nKalite: {jpeg_qual}  |  Boyut: {jpeg_kb:.1f} KB  |  PSNR: {psnr_j:.2f} dB",
                    fontsize=11,
                    fontweight="bold",
                    color="#1565c0",
                )
                ax_ul.axis("off")

                ax_ur = fig_cmp.add_subplot(gs[0, 1])
                ax_ur.imshow(dwt_gray, cmap="gray", vmin=0, vmax=255)
                ax_ur.set_title(
                    f"JPEG2000-benzeri DWT — yeniden kurulum\n{wavelet}, seviye {level}, Δ≈{q_step:.1f}  |  ~{dwt_kb:.1f} KB  |  PSNR: {psnr_w:.2f} dB",
                    fontsize=11,
                    fontweight="bold",
                    color="#6a1b9a",
                )
                ax_ur.axis("off")

                ax_ll = fig_cmp.add_subplot(gs[1, 0])
                _p99_dct = max(float(np.percentile(dct_mosaic, 99.5)), 1e-6)
                ax_ll.imshow(dct_mosaic, cmap="inferno", vmin=0, vmax=_p99_dct)
                ax_ll.set_title(
                    "JPEG katmanı: 8×8 DCT blokları\n|katsayı|/max (bant içi); parlaklık 99.5% persentil ile ölçeklendi",
                    fontsize=11,
                    fontweight="bold",
                    color="#e65100",
                )
                ax_ll.axis("off")

                # Sağ alt: tek seviye DWT — pywt.dwt2 ile LL / HL / LH / HH (2×2), referans slayt düzeni
                g_lr_outer = gs[1, 1].subgridspec(2, 1, height_ratios=[0.11, 1], hspace=0.18)
                ax_dwt_hdr = fig_cmp.add_subplot(g_lr_outer[0])
                ax_dwt_hdr.axis("off")
                ax_dwt_hdr.set_title(
                    f"JPEG2000 katmanı: tek seviye DWT alt bantları — {wavelet}, seviye {level}\n"
                    "(|katsayı| bant içi max’a bölünmüş; 0–1 gri ölçek)",
                    fontsize=10,
                    fontweight="bold",
                    color="#4a148c",
                    pad=4,
                )
                g_lr = g_lr_outer[1].subgridspec(2, 2, wspace=0.38, hspace=0.42)
                cell_rc = ((0, 0), (0, 1), (1, 0), (1, 1))
                for idx, (rr, cc) in enumerate(cell_rc):
                    axb = fig_cmp.add_subplot(g_lr[rr, cc])
                    im_b = axb.imshow(dwt_vis_bands[idx], cmap="gray", vmin=0, vmax=1, aspect="equal")
                    axb.set_title(dwt_band_titles[idx], fontsize=9, fontweight="bold", color="#37474f")
                    axb.axis("off")
                    fig_cmp.colorbar(im_b, ax=axb, fraction=0.046, pad=0.04, shrink=0.78)

                fig_cmp.suptitle(
                    f"Görsel karşılaştırma (Hedef BPP ≈ {hardness:.2f})",
                    fontsize=14,
                    fontweight="bold",
                    color="#263238",
                    y=0.98,
                )

                cmp_canvas_host = ctk.CTkFrame(layer_frame, fg_color="#fafafa", corner_radius=10)
                cmp_canvas_host.pack(fill="x", padx=12, pady=(0, 16))
                c_cmp = FigureCanvasTkAgg(fig_cmp, master=cmp_canvas_host)
                figure_canvas_redraw(c_cmp)
                # expand=False: tuvali pencere yüksekliğine sıkıştırma; doğal piksel boyutu korunur
                c_cmp.get_tk_widget().pack(fill="x", expand=False, padx=6, pady=8)

            except Exception as _layer_err:
                ctk.CTkLabel(
                    layer_frame,
                    text=f"Katman görselleştirmesi hesaplanamadı: {_layer_err}",
                    text_color="#c62828",
                    font=ctk.CTkFont(size=12),
                ).pack(padx=16, pady=12)

        # Çoklu kalite eğrileri (Q50/70/85/95): tek satıra bakmak yerine trendi göster.
        chart_frame = ctk.CTkFrame(
            scroll_body, fg_color="white", corner_radius=12, border_width=1, border_color="#CFD8DC"
        )
        chart_frame.pack(fill="x", padx=24, pady=(8, 10))
        ctk.CTkLabel(
            chart_frame,
            text="Kalite Seviyelerine Göre Karşılaştırma (Q50/70/85/95)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#1e88e5",
        ).pack(pady=(10, 2))

        dct_points = sorted(
            [(r.quality, r.psnr, r.bpp) for r in results if r.method == "DCT"],
            key=lambda x: x[0],
        )
        dwt_points = sorted(
            [(r.quality, r.psnr, r.bpp) for r in results if r.method == "DWT"],
            key=lambda x: x[0],
        )

        fig = Figure(figsize=(11.8, 2.35), dpi=100)
        ax_psnr = fig.add_subplot(1, 3, 1)
        ax_bpp = fig.add_subplot(1, 3, 2)
        ax_ratio = fig.add_subplot(1, 3, 3)

        if dct_points:
            q = [x[0] for x in dct_points]
            psnr = [x[1] for x in dct_points]
            bpp = [x[2] for x in dct_points]
            ratio = [x.compression_ratio for x in sorted([r for r in results if r.method == "DCT"], key=lambda r: r.quality)]
            ax_psnr.plot(q, psnr, marker="o", label="JPEG (DCT)")
            ax_bpp.plot(q, bpp, marker="o", label="JPEG (DCT)")
            ax_ratio.plot(q, ratio, marker="o", label="JPEG (DCT)")
        if dwt_points:
            q = [x[0] for x in dwt_points]
            psnr = [x[1] for x in dwt_points]
            bpp = [x[2] for x in dwt_points]
            ratio = [x.compression_ratio for x in sorted([r for r in results if r.method == "DWT"], key=lambda r: r.quality)]
            ax_psnr.plot(q, psnr, marker="o", label="JPEG2000 (DWT)")
            ax_bpp.plot(q, bpp, marker="o", label="JPEG2000 (DWT)")
            ax_ratio.plot(q, ratio, marker="o", label="JPEG2000 (DWT)")

        ax_psnr.set_title("PSNR vs Quality", fontsize=10, fontweight="bold")
        ax_psnr.set_xlabel("Quality")
        ax_psnr.set_ylabel("PSNR (dB)")
        ax_psnr.grid(True, alpha=0.3)
        ax_psnr.legend(fontsize=8)

        ax_bpp.set_title("BPP vs Quality", fontsize=10, fontweight="bold")
        ax_bpp.set_xlabel("Quality")
        ax_bpp.set_ylabel("BPP (düşük daha iyi)")
        ax_bpp.grid(True, alpha=0.3)
        ax_bpp.legend(fontsize=8)

        ax_ratio.set_title("Compression Ratio vs Quality", fontsize=10, fontweight="bold")
        ax_ratio.set_xlabel("Quality")
        ax_ratio.set_ylabel("Compression Ratio (yüksek daha iyi)")
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.legend(fontsize=8)

        fig.tight_layout(pad=1.4)
        chart_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        figure_canvas_redraw(chart_canvas)
        chart_canvas.get_tk_widget().pack(fill="x", expand=False, padx=8, pady=(2, 10))

        # Öneri Bölümü
        recommend_frame = ctk.CTkFrame(scroll_body, fg_color="#E3F2FD", corner_radius=10)
        recommend_frame.pack(fill="x", padx=24, pady=(6, 24))
        
        rec_text = "Öneri: Karşılaştırma verisi yetersiz."
        if dct_res and dwt_res:
            dct_score = 0
            dwt_score = 0
            if dct_res.psnr > dwt_res.psnr: dct_score += 1
            else: dwt_score += 1
            if dct_res.ssim > dwt_res.ssim: dct_score += 1
            else: dwt_score += 1
            if dct_res.compression_ratio > dwt_res.compression_ratio: dct_score += 1
            else: dwt_score += 1
            if dct_res.bpp < dwt_res.bpp: dct_score += 1
            else: dwt_score += 1
            if dct_res.compression_time_ms < dwt_res.compression_time_ms: dct_score += 1
            else: dwt_score += 1

            winner = "JPEG (DCT)" if dct_score >= dwt_score else "JPEG2000 (DWT)"
            rec_text = (
                f"Öneri (Q85): {winner} daha avantajlı görünüyor. "
                f"Skor: DCT {dct_score} - DWT {dwt_score} "
                "(PSNR, SSIM, Oran, BPP, Süre kriterleri)."
            )
            
        ctk.CTkLabel(
            recommend_frame, 
            text=rec_text, 
            font=ctk.CTkFont(size=13, slant="italic"),
            text_color="#1565C0",
            wraplength=860,
            justify="left"
        ).pack(pady=10, padx=16, anchor="w")

# ---------------------------------------------------------------------
# Pipeline modülleri (stages/ entegrasyonu)
# Kural: Import'lar try/except ile sarılır; yoksa ilgili butonlar devre dışı kalır.
# ---------------------------------------------------------------------
_PIPELINE_AVAILABLE = True  # Modüller varsa True kalır
_PIPELINE_IMPORT_ERROR = ""  # Hata mesajını UI'da göstermek için saklanır
try:
    # Pipeline modülleri
    from stages.stage_lossless import LosslessCompressionPipeline
    from stages.stage3_huffman_coding import HuffmanEncoder, HuffmanDecoder
    from stages.stage2_quantization import QuantizationModule
    from stages.stage4_bitstream import BitstreamPackager, CompressionMetadata
    from stages.stage5_decoder import ImageReconstructor
    from core.rate_distortion import RateDistortionAnalyzer, OptimumBitAllocator
    from core.image_type_comparison import ImageTypeComparisonRunner
except Exception as _e:
    # Modüller eksikse uygulama çalışmaya devam etmeli (butonlar/pipeline devre dışı)
    _PIPELINE_AVAILABLE = False
    _PIPELINE_IMPORT_ERROR = str(_e)
    LosslessCompressionPipeline = None  # type: ignore[misc, assignment]

# CustomTkinter genel görünüm: koyu tema + mavi vurgu teması
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --- Renk ve boyut sabitleri (tüm arayüzde tutarlılık için tek yerden değiştirin) ---
BG_MAIN = "#1B2431"          # koyu lacivert arka plan
PANEL_BG = "#242F3F"         # panel arka planı (koyu, hafif açık)
ACCENT = "#00BFA5"           # turkuaz vurgu (ana buton rengi)
ACCENT_HOVER = "#00A896"
ACCENT_DARK = "#00796B"
SEP_COLOR = "#37475A"
TEXT_PRIMARY = "#ECEFF4"     # açık metin (koyu zemin üstünde)
TEXT_SECONDARY = "#9AA7B8"
TEXT_MUTED = "#6B7A8F"
SURFACE_NEUTRAL = "#2D3B4E"
DANGER = "#FF6B6B"
IMG_FRAME_RADIUS = 16
RADIUS_SM = 10
RADIUS_MD = 14
RADIUS_LG = 20

# --- Tipografi sabitleri (CompressionApp; ComparisonWindow ayrı ele alınır) ---
# CTkFont, Tk root oluşmadan yaratılamaz → _ensure_ui_fonts() ile doldurulur.
FONT_APP_TITLE = None
FONT_APP_SUBTITLE = None
FONT_SECTION = None
FONT_SUBSECTION = None
FONT_SUBSECTION_REG = None
FONT_LABEL = None
FONT_LABEL_SM = None
FONT_BODY = None
FONT_HINT = None
FONT_MONO = None
FONT_DECOR = None


def _ensure_ui_fonts() -> None:
    """Tk root hazır olduktan sonra tipografi sabitlerini bir kez oluştur."""
    global FONT_APP_TITLE, FONT_APP_SUBTITLE, FONT_SECTION, FONT_SUBSECTION
    global FONT_SUBSECTION_REG, FONT_LABEL, FONT_LABEL_SM, FONT_BODY
    global FONT_HINT, FONT_MONO, FONT_DECOR
    if FONT_APP_TITLE is not None:
        return
    FONT_APP_TITLE = ctk.CTkFont(family="Segoe UI", size=22, weight="bold")
    FONT_APP_SUBTITLE = ctk.CTkFont(family="Segoe UI", size=13)
    FONT_SECTION = ctk.CTkFont(size=16, weight="bold")
    FONT_SUBSECTION = ctk.CTkFont(size=14, weight="bold")
    FONT_SUBSECTION_REG = ctk.CTkFont(size=14)
    FONT_LABEL = ctk.CTkFont(size=12, weight="bold")
    FONT_LABEL_SM = ctk.CTkFont(size=10, weight="bold")
    FONT_BODY = ctk.CTkFont(size=11)
    FONT_HINT = ctk.CTkFont(size=10)
    FONT_MONO = ctk.CTkFont(family="Courier", size=10)
    FONT_DECOR = ctk.CTkFont(size=48)

# Varsayılan işlem çözünürlüğü (genişlik, yükseklik); OpenCV resize (w,h) sırası
DEFAULT_TARGET_SIZE: Tuple[int, int] = (512, 512)

# Performans: alt bant önizleme boyutu, histogram örnekleme üst sınırı, işlem debounce (ms)
BAND_THUMB_SIZE: int = 240
HISTOGRAM_MAX_SAMPLES: int = 100_000
PIPELINE_DEBOUNCE_MS: int = 650

# Makul sıkıştırma hedef aralıkları (kullanıcı girişi bu sınırlarla kısıtlanır)
BPP_MIN: float = 0.1
BPP_MAX: float = 5.0
BPP_DEFAULT: float = 0.5
CR_MIN: float = 2.0
CR_MAX: float = 50.0
CR_DEFAULT: float = 10.0
MANUAL_DELTA_MAX: float = 500.0


def compression_coeff_bounds(is_cr_mode: bool) -> Tuple[float, float]:
    """BPP veya sıkıştırma oranı moduna göre (min, max) sınırları döndür."""
    if is_cr_mode:
        return CR_MIN, CR_MAX
    return BPP_MIN, BPP_MAX


def clamp_compression_coeff(value: float, is_cr_mode: bool) -> float:
    """Değeri makul aralığa sıkıştır."""
    lo, hi = compression_coeff_bounds(is_cr_mode)
    return float(max(lo, min(hi, value)))


def format_compression_coeff(value: float, is_cr_mode: bool) -> str:
    """Moda uygun metin gösterimi."""
    if is_cr_mode:
        return f"{value:.1f}"
    return f"{value:.2f}"


def mse_fn(a: np.ndarray, b: np.ndarray) -> float:
    """İki aynı boyutlu görüntü/kanal arasında ortalama kare hata (MSE). Metrik tablosunda kullanılır."""
    # float64 ile toplama taşması riskini azalt
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr_fn(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """PSNR (dB). MSE sıfırsa görüntüler özdeş kabul edilir (yüksek sabit döndürülür)."""
    m = mse_fn(a, b)
    if m < 1e-12:
        return 99.99
    return float(10.0 * np.log10((max_val**2) / m))


def quantize_coeffs(coeffs: List, step: float, mode: str = "scalar") -> List:
    """
    pywt.wavedec2 çıktısı (liste: [cA, (cH,cV,cD), ...]) üzerinde bant başına kuantalama.

    pywt alt bant eşlemesi (JPEG2000):
      - cH = horizontal detail → LH step
      - cV = vertical detail   → HL step
      - cD = diagonal detail   → HH step

    'step' temel kuantalama adımı (Δ); skaler faktörler core.config ile uyumludur.
    mode="uniform": tüm bantlarda aynı Δ. mode="scalar" (varsayılan): bant ağırlıklı.
    """
    if step <= 0:
        step = 1e-6
    if mode == "uniform":
        delta = max(1e-6, float(step))
        out: List[Any] = [np.round(coeffs[0] / delta) * delta]
        for tup in coeffs[1:]:
            cH, cV, cD = tup
            out.append((
                np.round(cH / delta) * delta,
                np.round(cV / delta) * delta,
                np.round(cD / delta) * delta,
            ))
        return out

    ll_delta = max(0.5, step * float(_DWT_SCALAR_FACTORS.get("LL", 1.0)) / 8.0)
    lh_step = max(1e-6, step * float(_DWT_SCALAR_FACTORS.get("LH", 2.0)))
    hl_step = max(1e-6, step * float(_DWT_SCALAR_FACTORS.get("HL", 2.0)))
    hh_step = max(1e-6, step * float(_DWT_SCALAR_FACTORS.get("HH", 4.0)))

    out: List[Any] = [np.round(coeffs[0] / ll_delta) * ll_delta]

    for tup in coeffs[1:]:
        cH, cV, cD = tup
        out.append(
            (
                np.round(cH / lh_step) * lh_step,
                np.round(cV / hl_step) * hl_step,
                np.round(cD / hh_step) * hh_step,
            )
        )
    return out


def estimate_wavelet_compressed_bytes(coeffs: List) -> int:
    """
    Gerçek Huffman boyutu değil: kuantalanmış katsayıları int16 (2 byte/eleman) sayarak kabaca tahmin.
    Sıkıştırma kazancı yüzdesi bu tahmine göre hesaplanır; gerçek .bin için Aşama 3 gerekir.
    NOT: Gerçek Huffman boyutu için huffman_compressed_size() kullanılır.
    """
    total = 0
    total += coeffs[0].size * 2  # LL
    for tup in coeffs[1:]:
        for band in tup:
            total += band.size * 2
    return total


def _sep(parent: Union[ctk.CTkFrame, ctk.CTkScrollableFrame]) -> ctk.CTkFrame:
    """Sol panelde görsel bölüm ayırıcı — yüksekliği 1 px ince çizgi."""
    f = ctk.CTkFrame(parent, height=1, fg_color=SEP_COLOR)
    f.pack(fill="x", padx=12, pady=10)
    return f


class CompressionApp(ctk.CTk):
    """Ana pencere: tüm widget'lar ve durum burada tutulur."""

    def __init__(self) -> None:
        super().__init__()  # CTk kök penceresini başlat
        _ensure_ui_fonts()
        self.title("CENG 384 Proje: Görüntü Sıkıştırma Sistemi")
        self.geometry("1700x1000")  # Başlangıç boyutu (artırılmış)
        self.minsize(1200, 800)  # Küçültme alt sınırı (artırılmış)
        self.configure(fg_color=BG_MAIN)
        self.current_coeffs = None

        self._image_path: Optional[str] = None  # Son seçilen dosya yolu
        self._original_bgr: Optional[np.ndarray] = None  # OpenCV BGR tam görüntü (diskten)
        self._luma: Optional[np.ndarray] = None  # DWT/PSNR için Y kanalı, shape (h,w) veya (h,w,3), float32
        self._display_orig: Optional[np.ndarray] = None  # RGB önizleme için (thumbnail)
        self._display_recon: Optional[np.ndarray] = None  # Yeniden oluşturulan gri RGB önizleme
        self._is_color_image: bool = False  # Renkli görüntü flag'i (3-channel)
        self._quant_mode: str = "scalar"  # Kuantalama modu: "scalar" | "uniform"
        self._original_raw_bytes: int = 0  # İsteğe bağlı; şu an ham bellek boyutu
        self._last_coeffs: Optional[List] = None  # Son DWT coeffs (Katmanları Gör butonu için) (FR-7)
        self._last_coeffs = None  # Başlangıçta katsayı yok (debug/FR-7)
        self._last_encoded_image_bytes: Optional[bytes] = None  # JPEG modunda ölçülen gerçek bitstream
        self._last_encoded_image_ext: Optional[str] = None
        self._image_refs: Dict[str, Any] = {}  # CTkImage referansları (GC koruması)
        self._dct_block_pos: Optional[Tuple[int, int]] = None  # Örnek 8×8 blok (bi, bj); None → merkez
        self._last_jpeg_quality: int = 85
        self._run_history: List[Dict[str, Any]] = []  # Son işlem kayıtları (en yeni en üstte gösterilir)
        self._max_history_items: int = 10
        self._history_visible: bool = True

        self._process_job_id: int = 0
        self._process_lock = threading.Lock()
        self._snap_orig_bytes_override: Optional[int] = None

        self._build_ui()  # Tüm arayüzü kur
        self._on_param_change()  # Başlangıç yöntemine göre katsayı etiketleri / ipuçları
        self._set_status("Hazır — Görüntü Yükle ile başlayın.")
        self._reset_metrics_display()  # Tablo hücrelerini — yap
        self._set_placeholders_empty()  # Görüntü yok metinleri
        self._refresh_history_view()

    def _build_ui(self) -> None:
        """Pencere yerleşimi: üst başlık → gövde (grid: sol | orta | sağ) → alt durum çubuğu."""

        # --- Üst başlık alanı ---
        header = ctk.CTkFrame(self, fg_color=ACCENT_DARK)
        header.pack(fill="x", padx=24, pady=(18, 8))

        ctk.CTkLabel(
            header,
            text="CENG 384 Proje: Görüntü Sıkıştırma Sistemi",
            font=FONT_APP_TITLE,
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="JPEG ve JPEG2000 (DWT) karşılaştırması — kontrol paneli ve anlık analiz",
            font=FONT_APP_SUBTITLE,
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w", pady=(4, 0))

        # --- Gövde: 3 sütun grid (0=sol kontrol, 1=orta scrollable analiz, 2=sağ history) ---
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=(0, 0))
        body.grid_columnconfigure(1, weight=1)  # Orta sütun genişleyebilir
        body.grid_rowconfigure(0, weight=1)

        # --- Sol: Kontrol paneli (kaydırılabilir içerik) ---
        self.left_panel = ctk.CTkFrame(
            body,
            width=308,
            corner_radius=RADIUS_LG,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=SEP_COLOR,
        )
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.left_panel.grid_propagate(False)  # width=308 sabit kalsın

        self.left_scroll = ctk.CTkScrollableFrame(
            self.left_panel,
            fg_color="transparent",
            width=280,
        )
        self.left_scroll.pack(fill="both", expand=True, padx=6, pady=(8, 8))
        self.left_col = self.left_scroll  # Sol panel konteyner alias'ı (istenen buton snippet'i için)

        ctk.CTkLabel(
            self.left_scroll,
            text="Kontrol Paneli",
            font=FONT_SECTION,
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w", padx=8, pady=(4, 2))

        # NOT: Orijinal görüntü boyutu korunur — yeniden boyutlandırma yapılmaz
        _sep(self.left_scroll)

        # Dosya seçici — tkinter filedialog _on_load_image içinde
        btn_load = ctk.CTkButton(
            self.left_scroll,
            text="Görüntü Yükle",
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self._on_load_image,
        )
        btn_load.pack(fill="x", padx=8, pady=(0, 8))

        # Temizle butonu — tüm işlemleri sıfırla
        btn_clear = ctk.CTkButton(
            self.left_scroll,
            text="Temizle",
            height=36,
            font=ctk.CTkFont(size=13),
            fg_color="transparent",
            border_width=1,
            border_color=ACCENT,
            text_color=ACCENT,
            hover_color=SURFACE_NEUTRAL,
            command=self._on_clear_image,
        )
        btn_clear.pack(fill="x", padx=8, pady=(0, 8))

        # Yöntem: _process içinde üç dal (JPEG / DWT / karşılaştır)
        ctk.CTkLabel(
            self.left_scroll,
            text="Yöntem Seçimi",
            font=FONT_LABEL,
        ).pack(anchor="w", padx=8)
        self.method_var = ctk.StringVar(value="JPEG2000 (DWT)")
        self._method_var = self.method_var  # Türkçe: karşılaştır butonunda aynı değişkeni kullanmak için alias
        self.method_menu = ctk.CTkOptionMenu(
            self.left_scroll,
            values=[
                "JPEG (DCT)",
                "JPEG2000 (DWT)",
                "Lossless (Kayıpsız - Tıbbi)",  # Kayıpsız zlib tabanlı pipeline (parmak izi/medikal) (FR-?)
            ],
            variable=self.method_var,
            command=self._on_method_change,
            width=250,
        )
        self.method_menu.pack(padx=8, pady=(6, 8))

        _sep(self.left_scroll)

        # DWT parametreleri — JPEG (DCT) modunda gizlenir
        self.dwt_controls_frame = ctk.CTkFrame(self.left_scroll, fg_color="transparent")
        ctk.CTkLabel(
            self.dwt_controls_frame,
            text="Dalgacık Tipi",
            font=FONT_LABEL,
        ).pack(anchor="w", padx=8)
        self.wavelet_var = ctk.StringVar(value="haar")
        self.wavelet_menu = ctk.CTkOptionMenu(
            self.dwt_controls_frame,
            values=["haar", "db2", "db4", "db6", "db8", "db12", "db32", "bior4.4", "prqmf4", "prqmf6"],
            variable=self.wavelet_var,
            command=self._on_wavelet_changed,
            width=250,
        )
        self.wavelet_menu.pack(padx=8, pady=(6, 0))
        self.wavelet_hint = ctk.CTkLabel(
            self.dwt_controls_frame,
            text="",
            font=FONT_HINT,
            text_color=ACCENT,
            wraplength=260,
            justify="left",
        )
        self.wavelet_hint.pack(anchor="w", padx=24, pady=(2, 4))
        _sep(self.dwt_controls_frame)
        self.level_title = ctk.CTkLabel(
            self.dwt_controls_frame,
            text="Dekompozisyon Seviyesi (1–3)",
            font=FONT_LABEL,
        )
        self.level_title.pack(anchor="w", padx=8, pady=(0, 0))
        self.level_var = ctk.IntVar(value=2)
        self.level_slider = ctk.CTkSegmentedButton(
            self.dwt_controls_frame,
            values=["1", "2", "3"],
            command=lambda _value: self._on_param_change(),
        )
        self.level_slider.set("2")
        self.level_slider.pack(fill="x", padx=8, pady=6)
        self.level_label = ctk.CTkLabel(self.dwt_controls_frame, text="Seviye: 2")
        self.level_label.pack(anchor="w", padx=8)
        self.level_hint = ctk.CTkLabel(
            self.dwt_controls_frame,
            text="DWT modunda aktif.",
            font=FONT_BODY,
            text_color=ACCENT,
            wraplength=260,
            justify="left",
        )
        self.level_hint.pack(anchor="w", padx=8, pady=(2, 0))

        _sep(self.dwt_controls_frame)
        ctk.CTkLabel(
            self.dwt_controls_frame,
            text="Kuantalama Modu",
            font=FONT_LABEL,
        ).pack(anchor="w", padx=8)
        self.quant_mode_seg = ctk.CTkSegmentedButton(
            self.dwt_controls_frame,
            values=["Scalar (Varsayılan)", "Uniform"],
            command=self._on_quant_mode_change,
        )
        self.quant_mode_seg.set("Scalar (Varsayılan)")
        self.quant_mode_seg.pack(fill="x", padx=8, pady=(6, 6))

        self.manual_step_toggle = ctk.CTkSwitch(
            self.dwt_controls_frame,
            text="Manuel adım kullan",
            command=self._on_manual_step_toggle,
        )
        self.manual_step_toggle.pack(anchor="w", padx=8, pady=(2, 4))

        self.manual_step_entry = ctk.CTkEntry(
            self.dwt_controls_frame,
            placeholder_text="Δ değeri",
            width=120,
        )
        self.manual_step_entry.configure(state="disabled")
        self.manual_step_entry.pack(anchor="w", padx=8, pady=(0, 2))

        self.manual_step_hint = ctk.CTkLabel(
            self.dwt_controls_frame,
            text="Manuel adım aktifse hedef BPP/CR yok sayılır. Δ üst sınırı: 500.",
            font=FONT_BODY,
            text_color=TEXT_SECONDARY,
            wraplength=260,
            justify="left",
        )
        self.manual_step_hint.pack(anchor="w", padx=8, pady=(2, 0))

        self.dwt_controls_frame.pack(fill="x", padx=0, pady=0)

        _sep(self.left_scroll)

        # --- Sıkıştırma Modu Seçimi (BPP / Compression Ratio) ---
        ctk.CTkLabel(
            self.left_scroll,
            text="Sıkıştırma Modu",
            font=FONT_LABEL,
        ).pack(anchor="w", padx=8, pady=(0, 2))
        self.comp_mode_var = ctk.StringVar(value="BPP")
        self.comp_mode_seg = ctk.CTkSegmentedButton(
            self.left_scroll,
            values=["BPP", "Compression Ratio"],
            variable=self.comp_mode_var,
            command=self._on_comp_mode_change,
        )
        self.comp_mode_seg.pack(fill="x", padx=8, pady=(0, 6))

        # Sıkıştırma katsayısı — BPP veya CR değeri
        self.hardness_title = ctk.CTkLabel(
            self.left_scroll,
            text="Hedef BPP (Bits Per Pixel)",
            font=FONT_LABEL,
        )
        self.hardness_title.pack(anchor="w", padx=8, pady=(0, 0))
        
        self.compression_control_frame = ctk.CTkFrame(self.left_scroll, fg_color="transparent")
        self.compression_control_frame.pack(fill="x", padx=8, pady=6)
        
        coeff_row = ctk.CTkFrame(self.compression_control_frame, fg_color="transparent")
        coeff_row.pack(fill="x")
        ctk.CTkLabel(coeff_row, text="Değer:").pack(side="left", padx=(0, 8))
        self.compression_coeff_entry = ctk.CTkEntry(
            coeff_row,
            width=90,
            placeholder_text=f"{BPP_MIN:g} – {BPP_MAX:g}",
        )
        self.compression_coeff_entry.insert(0, format_compression_coeff(BPP_DEFAULT, False))
        self.compression_coeff_entry.pack(side="left")
        self.compression_coeff_entry.bind("<KeyRelease>", lambda _e: self._on_coeff_entry_key())
        self.compression_coeff_entry.bind("<FocusOut>", lambda _e: self._normalize_compression_coeff_entry())
        self.compression_coeff_entry.bind("<Return>", lambda _e: self._normalize_compression_coeff_entry())
        self._register_coeff_entry_validation()

        self.coeff_range_label = ctk.CTkLabel(
            self.compression_control_frame,
            text=self._coeff_range_text(False),
            font=FONT_HINT,
            text_color=TEXT_MUTED,
        )
        self.coeff_range_label.pack(anchor="w", pady=(2, 0))
        
        self.quant_label = ctk.CTkLabel(self.compression_control_frame, text="—")
        self.quant_label.pack(anchor="w", pady=(6, 0))
        
        # Mod açıklama ipucu
        self.quant_hint = ctk.CTkLabel(
            self.compression_control_frame,
            text="",
            font=FONT_HINT,
            text_color=ACCENT,
            justify="left",
        )
        self.quant_hint.pack(anchor="w", pady=(2, 8))

        _sep(self.left_scroll)

        # Manuel yenileme — slider zaten canlı işliyor; yine de zorunlu tıklama için
        ctk.CTkButton(  # Türkçe: sıkıştırılmış/yeniden oluşturulmuş görseli kaydetme butonu
            self.left_col,  # Türkçe: sol panel konteyneri (Uygula/Yenile butonunun üstü)
            text="💾 Sıkıştırılmışı İndir",  # Türkçe: buton metni
            command=self._save_compressed_image,  # Türkçe: kaydetme işlemini başlat
            fg_color="transparent",
            border_width=1,
            border_color=ACCENT,
            text_color=ACCENT,
            hover_color=SURFACE_NEUTRAL,
        ).pack(fill="x", padx=16, pady=(8, 0))  # Türkçe: sol panelde yerleşim
        btn_apply = ctk.CTkButton(
            self.left_scroll,
            text="Uygula / Yenile",
            height=38,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self._on_apply_clicked,
        )
        btn_apply.pack(fill="x", padx=8, pady=(16, 12))

        self.btn_wavelet_preview = ctk.CTkButton(
            self.left_scroll,
            text="🌊 Wavelet Önizle",
            command=self._on_wavelet_preview_clicked,
            fg_color="transparent",
            border_width=1,
            border_color=ACCENT,
            text_color=ACCENT,
        )
        self.btn_wavelet_preview.pack(fill="x", padx=8, pady=(0, 12))
        self.btn_wavelet_preview.pack_forget()

        ctk.CTkButton(
            self.left_col,
            text="📊 Metrik Tablosu",
            command=self._show_metrics,
            fg_color="transparent",
            border_width=1,
            border_color=ACCENT,
            text_color=ACCENT,
            hover_color=SURFACE_NEUTRAL,
        ).pack(fill="x", padx=16, pady=(8, 0))

        # R-D Analizi butonu
        ctk.CTkButton(
            self.left_col,
            text="📈 R-D Analizi",
            command=lambda: [
                setattr(self, '_manual_trigger', True),
                self._run_rd_analysis()
            ],
            fg_color="transparent",
            border_width=1,
            border_color=ACCENT,
            text_color=ACCENT,
            hover_color=SURFACE_NEUTRAL,
        ).pack(fill="x", padx=16, pady=(8, 0))

        # Görüntü türü testi butonu
        ctk.CTkButton(
            self.left_col,
            text="🔬 Tür Testi",
            command=lambda: [
                setattr(self, '_manual_trigger', True),
                self._run_image_type_test()
            ],
            fg_color="transparent",
            border_width=1,
            border_color=ACCENT,
            text_color=ACCENT,
            hover_color=SURFACE_NEUTRAL,
        ).pack(fill="x", padx=16, pady=(8, 0))

        # Katmanları görselleştir butonu
        ctk.CTkButton(
            self.left_col,
            text="👁 Katmanları Gör",
            command=self._show_layers,
            fg_color="transparent",
            border_width=1,
            border_color=ACCENT,
            text_color=ACCENT,
            hover_color=SURFACE_NEUTRAL,
        ).pack(fill="x", padx=16, pady=(8, 0))

        # JPEG vs JPEG2000 Karşılaştır butonu
        ctk.CTkButton(
            self.left_col,
            text="🖼 JPEG vs JPEG2000",
            command=self._show_jpeg_vs_jpeg2000_comparison,
            fg_color="transparent",
            border_width=1,
            border_color=ACCENT,
            text_color=ACCENT,
            hover_color=SURFACE_NEUTRAL,
        ).pack(fill="x", padx=16, pady=(8, 16))

        # --- Orta: Scrollable alan (görüntüler + şema + grafikler + bantlar alt alta) ---
        self.center_scroll = ctk.CTkScrollableFrame(
            body,
            fg_color="transparent"
        )
        self.center_scroll.grid(row=0, column=1, sticky="nsew", padx=0)
        try:
            self.center_scroll._parent_canvas.bind(
                "<MouseWheel>", self._on_center_scroll_event, add="+"
            )
        except Exception:
            pass

        # Orta içinde container (pack layoutu)
        center = ctk.CTkFrame(self.center_scroll, fg_color="transparent")
        center.pack(fill="x", padx=0)

        # --- Görüntüler bölümü ---
        img_frame = ctk.CTkFrame(
            center,
            corner_radius=RADIUS_LG,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=SEP_COLOR,
        )
        img_frame.pack(fill="x", padx=0, pady=(0, 12))
        img_frame.grid_columnconfigure(0, weight=1)
        img_frame.grid_columnconfigure(1, weight=0)
        img_frame.grid_columnconfigure(2, weight=1)

        self.orig_frame = ctk.CTkFrame(
            img_frame,
            corner_radius=IMG_FRAME_RADIUS,
            fg_color=SURFACE_NEUTRAL,
            border_width=1,
            border_color=SEP_COLOR,
        )
        self.orig_frame.grid(row=0, column=0, padx=16, pady=20, sticky="n")
        ctk.CTkLabel(self.orig_frame, text="Orijinal Görüntü", font=FONT_LABEL).pack(pady=(12, 6))
        # CTkLabel: görüntü yokken metin; yüklendikten sonra _set_image_label ile PIL/CTkImage
        self.lbl_orig = ctk.CTkLabel(
            self.orig_frame,
            text="Henüz görüntü yok\n\nGörüntü Yükle’ye tıklayın",
            width=380,
            height=320,
            fg_color=BG_MAIN,
            text_color=TEXT_MUTED,
            font=FONT_SUBSECTION_REG,
            justify="center",
        )
        self.lbl_orig.pack(padx=16, pady=(0, 4))
        # İşlem çözünürlüğü bilgisi (piksel sayısı aynı; sıkıştırma bit cinsinden ayrı metrikte)
        self.lbl_orig_caption = ctk.CTkLabel(
            self.orig_frame,
            text="Çözünürlük: —",
            font=FONT_BODY,
            text_color=TEXT_SECONDARY,
        )
        self.lbl_orig_caption.pack(pady=(0, 12))

        arrow = ctk.CTkLabel(img_frame, text="➜", font=FONT_DECOR, text_color=ACCENT)
        arrow.grid(row=0, column=1, pady=120, padx=30)

        self.recon_frame = ctk.CTkFrame(
            img_frame,
            corner_radius=IMG_FRAME_RADIUS,
            fg_color=SURFACE_NEUTRAL,
            border_width=1,
            border_color=SEP_COLOR,
        )
        self.recon_frame.grid(row=0, column=2, padx=16, pady=20, sticky="n")
        ctk.CTkLabel(
            self.recon_frame,
            text="Sıkıştırılmış / Yeniden Oluşturulmuş",
            font=FONT_LABEL,
        ).pack(pady=(12, 6))
        self.lbl_recon = ctk.CTkLabel(
            self.recon_frame,
            text="Önce görüntü yükleyin\n\nSonra Uygula ile işleyin",
            width=380,
            height=320,
            fg_color=BG_MAIN,
            text_color=TEXT_MUTED,
            font=FONT_SUBSECTION_REG,
            justify="center",
        )
        self.lbl_recon.pack(padx=16, pady=(0, 4))
        self.lbl_recon_caption = ctk.CTkLabel(
            self.recon_frame,
            text="Çözünürlük: —",
            font=FONT_BODY,
            text_color=TEXT_SECONDARY,
        )
        self.lbl_recon_caption.pack(pady=(0, 12))
        self.btn_recon_zoom = ctk.CTkButton(
            self.recon_frame,
            text="🔍 Zoom (Bozunma İncele)",
            height=30,
            command=self._open_recon_zoom_view,
            fg_color="transparent",
            border_width=1,
            border_color=ACCENT,
            text_color=ACCENT,
            hover_color=SURFACE_NEUTRAL,
        )
        self.btn_recon_zoom.pack(padx=16, pady=(0, 12), anchor="e")

        # --- Sağ: İşlem Geçmişi (dar panel) ---
        history_panel = ctk.CTkFrame(
            body,
            width=250,
            corner_radius=RADIUS_LG,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=SEP_COLOR,
        )
        history_panel.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        history_panel.grid_propagate(False)

        # İşlem geçmişi: sağ panel
        self.history_frame = ctk.CTkFrame(
            history_panel,
            fg_color=SURFACE_NEUTRAL,
            corner_radius=RADIUS_MD,
            border_width=1,
            border_color=SEP_COLOR,
        )
        self.history_frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.history_header = ctk.CTkFrame(self.history_frame, fg_color="transparent")
        self.history_header.pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkLabel(
            self.history_header,
            text="İşlem Geçmişi (Son 10)",
            font=FONT_SUBSECTION,
            text_color=TEXT_PRIMARY,
        ).pack(side="left", padx=(4, 0), pady=(2, 2))

        self.btn_toggle_history = ctk.CTkButton(
            self.history_header,
            text="Gizle",
            width=70,
            height=28,
            command=self._toggle_history_visibility,
            fg_color="transparent",
            border_width=1,
            border_color=ACCENT,
            text_color=ACCENT,
            hover_color=SURFACE_NEUTRAL,
        )
        self.btn_toggle_history.pack(side="right", padx=(4, 0), pady=(0, 0))

        self.history_scroll = ctk.CTkScrollableFrame(
            self.history_frame,
            fg_color="transparent",
            height=220,
        )
        self.history_scroll.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.btn_clear_history = ctk.CTkButton(
            self.history_frame,
            text="Geçmişi Temizle",
            height=30,
            command=self._clear_history,
            fg_color="transparent",
            border_width=1,
            border_color=ACCENT,
            text_color=ACCENT,
            hover_color=SURFACE_NEUTRAL,
        )
        self.btn_clear_history.pack(anchor="e", padx=8, pady=(0, 8))

        # --- Orta: Analiz panelleri (şema + grafikler + bantlar) - center'e pack edilir ---
        self.schema_title_label = ctk.CTkLabel(
            center,
            text="2D DWT Subband Structure",
            font=FONT_LABEL,
        )
        self.schema_title_label.pack(anchor="w", padx=16, pady=(14, 2))

        self.dct_explanation_label = ctk.CTkLabel(
            center,
            text="",
            font=FONT_BODY,
            text_color=TEXT_SECONDARY,
            wraplength=720,
            justify="left",
        )
        self.dct_explanation_label.pack(anchor="w", padx=16, pady=(0, 4))

        # Sabit LL/HL/LY/HH şeması — başlangıçta boş bilgilerle
        self.fig_schema = Figure(figsize=(11.0, 3.5), dpi=100)
        self.fig_schema.patch.set_facecolor(PANEL_BG)
        initial_info = {
            "method": "—",
            "level_delta": "—",
            "huffman": "—",
            "compression": "—"
        }
        self._draw_static_dwt_schema(info_dict=initial_info, level=2)
        self.schema_frame = ctk.CTkFrame(center, fg_color="transparent")
        self.schema_frame.pack(fill="x", padx=16, pady=(0, 6))
        self.canvas_schema = FigureCanvasTkAgg(self.fig_schema, master=self.schema_frame)
        self.canvas_schema.get_tk_widget().pack(fill="x", padx=0, pady=0)
        
        # Dinamik bilgiler label'ı schema figüründe gösterilecek

        # Metrik widget'ları
        self.metric_val_psnr = ctk.CTkLabel(self, text="—")
        self.metric_val_mse = ctk.CTkLabel(self, text="—")
        self.metric_val_cr = ctk.CTkLabel(self, text="—")
        self.metric_val_huffman = ctk.CTkLabel(self, text="—")
        self.metric_quality = ctk.CTkLabel(self, text="Kalite: —")

        self.metric_note = ctk.CTkLabel(
            center,
            text="",
            font=FONT_BODY,
            text_color=TEXT_SECONDARY,
            wraplength=370,
        )
        self.metric_note.pack(anchor="w", padx=16, pady=(0, 10))

        # Analiz önizlemesi — pack sırası _update_dct_center_layout ile yönteme göre ayarlanır
        self.band_frame = ctk.CTkFrame(
            center,
            fg_color=PANEL_BG,
            corner_radius=RADIUS_MD,
            border_width=1,
            border_color=SEP_COLOR,
        )

        self._hist_block_title = ctk.CTkLabel(
            center,
            text="Anlık grafikler (katsayı dağılımı)",
            font=FONT_LABEL,
        )
        self._hist_block_sub = ctk.CTkLabel(
            center,
            text="Yönteme göre DWT veya DCT katsayıları",
            font=FONT_BODY,
            text_color=TEXT_MUTED,
        )
        self.fig = Figure(figsize=(5.5, 5.0), dpi=100)
        self.fig.patch.set_facecolor(PANEL_BG)
        self.ax_ll = self.fig.add_subplot(2, 2, 1)
        self.ax_hl = self.fig.add_subplot(2, 2, 2)
        self.ax_lh = self.fig.add_subplot(2, 2, 3)
        self.ax_hh = self.fig.add_subplot(2, 2, 4)
        for ax, t in zip(
            (self.ax_ll, self.ax_hl, self.ax_lh, self.ax_hh),
            ("LL", "HL", "LH", "HH"),
        ):
            ax.set_title(t, fontsize=9)
            ax.tick_params(labelsize=7)
        self.fig.subplots_adjust(hspace=0.5, wspace=0.3)
        self.hist_frame = ctk.CTkFrame(center, fg_color="transparent")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.hist_frame)
        self.canvas.get_tk_widget().pack(fill="x", padx=0, pady=0)

        self.band_panel_heading = ctk.CTkLabel(
            self.band_frame,
            text="Analiz önizlemesi",
            font=FONT_SUBSECTION,
            text_color=TEXT_PRIMARY,
        )
        self.band_panel_heading.pack(anchor="w", padx=12, pady=(10, 4))

        # --- DCT satırı (klasik JPEG/DCT akışı: blok → DCT → kuantalama → zigzag) ---
        self.band_row_dct = ctk.CTkFrame(self.band_frame, fg_color="transparent")
        self._lbl_band_row_dct = ctk.CTkLabel(
            self.band_row_dct,
            text="JPEG / DCT Coefficient Analysis",
            font=FONT_LABEL,
            text_color=ACCENT_HOVER,
        )
        self._lbl_band_row_dct.pack(anchor="w", padx=12, pady=(4, 2))

        # --- 1. JPEG Pipeline (collapsible, kompakt) ---
        self._dct_pipeline_open = True
        self.dct_pipeline_section = ctk.CTkFrame(self.band_row_dct, fg_color=SURFACE_NEUTRAL, corner_radius=RADIUS_SM)
        self.dct_pipeline_section.pack(fill="x", padx=12, pady=(0, 8))
        self.btn_pipeline_toggle = ctk.CTkButton(
            self.dct_pipeline_section,
            text="JPEG Compression Pipeline ▼",
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="transparent",
            text_color=ACCENT,
            hover_color=BG_MAIN,
            anchor="w",
            command=self._toggle_dct_pipeline,
        )
        self.btn_pipeline_toggle.pack(fill="x", padx=4, pady=(4, 0))
        self.dct_pipeline_body = ctk.CTkFrame(self.dct_pipeline_section, fg_color="transparent")
        self.dct_pipeline_body.pack(fill="x", padx=4, pady=(0, 6))
        self.fig_dct_pipeline = Figure(figsize=(10.0, 1.15), dpi=100)
        self.fig_dct_pipeline.patch.set_facecolor(PANEL_BG)
        pl_frame = ctk.CTkFrame(self.dct_pipeline_body, fg_color="transparent")
        pl_frame.pack(fill="x")
        self.canvas_dct_pipeline = FigureCanvasTkAgg(self.fig_dct_pipeline, master=pl_frame)
        self.canvas_dct_pipeline.get_tk_widget().pack(fill="x", padx=0, pady=(0, 2))
        self.dct_pipeline_info_label = ctk.CTkLabel(
            self.dct_pipeline_body,
            text="—",
            font=FONT_HINT,
            text_color=TEXT_SECONDARY,
            wraplength=680,
            justify="left",
        )
        self.dct_pipeline_info_label.pack(anchor="w", padx=6, pady=(0, 2))
        self._draw_dct_pipeline_compact()

        # --- 2. Example 8×8 Block Analysis (ana odak) ---
        self.dct_example_frame = ctk.CTkFrame(self.band_row_dct, fg_color=BG_MAIN, corner_radius=RADIUS_SM)
        self.dct_example_frame.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkLabel(
            self.dct_example_frame,
            text="Example 8×8 Block Analysis",
            font=FONT_LABEL,
            text_color=ACCENT_HOVER,
        ).pack(anchor="w", padx=8, pady=(10, 4))

        dct_sel_row = ctk.CTkFrame(self.dct_example_frame, fg_color="transparent")
        dct_sel_row.pack(fill="x", padx=8, pady=(0, 6))
        self.dct_block_selector_label = ctk.CTkLabel(dct_sel_row, text="—")
        self.dct_block_selector_label.pack(side="left", padx=(0, 10))
        sel_info = ctk.CTkFrame(dct_sel_row, fg_color="transparent")
        sel_info.pack(side="left", fill="y")
        ctk.CTkLabel(
            sel_info,
            text="Selected 8×8 Block on Image",
            font=FONT_LABEL_SM,
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w")
        self.dct_block_pos_label = ctk.CTkLabel(
            sel_info,
            text="Block position: —",
            font=FONT_HINT,
            text_color=TEXT_MUTED,
        )
        self.dct_block_pos_label.pack(anchor="w", pady=(2, 0))
        ctk.CTkLabel(
            sel_info,
            text="(varsayılan: merkez blok; ileride tıklama ile seçilebilir)",
            font=ctk.CTkFont(size=9),
            text_color=TEXT_MUTED,
        ).pack(anchor="w")
        self.btn_random_block = ctk.CTkButton(
            sel_info,
            text="🎲 Rastgele Blok Seç",
            command=self._on_random_block_clicked,
            fg_color="transparent",
            border_width=1,
            border_color=ACCENT,
            text_color=ACCENT,
        )
        self.btn_random_block.pack(anchor="w", pady=(6, 0))

        dct_mat_grid = ctk.CTkFrame(self.dct_example_frame, fg_color="transparent")
        dct_mat_grid.pack(fill="both", expand=True, padx=8, pady=(0, 10))
        for c in (0, 1):
            dct_mat_grid.grid_columnconfigure(c, weight=1, uniform="dct2col")
        for r in (0, 1):
            dct_mat_grid.grid_rowconfigure(r, weight=1, uniform="dct2row")

        def _dct_analysis_cell(row: int, col: int) -> ctk.CTkFrame:
            cell = ctk.CTkFrame(
                dct_mat_grid,
                fg_color=PANEL_BG,
                corner_radius=RADIUS_SM,
                border_width=1,
                border_color=SEP_COLOR,
            )
            cell.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            inner = ctk.CTkFrame(cell, fg_color="transparent")
            inner.pack(fill="both", expand=True, padx=10, pady=10)
            return inner

        cell_orig = _dct_analysis_cell(0, 0)
        self.dct_pixel_block_label = ctk.CTkLabel(cell_orig, text="—")
        self.dct_pixel_block_label.pack(pady=(4, 6))
        ctk.CTkLabel(
            cell_orig,
            text="Original 8×8 Pixel Block",
            font=FONT_LABEL_SM,
            text_color=ACCENT_HOVER,
        ).pack(pady=(0, 4))

        cell_dct = _dct_analysis_cell(0, 1)
        self.dct_coeff_matrix_label = ctk.CTkLabel(cell_dct, text="—")
        self.dct_coeff_matrix_label.pack(pady=(4, 6))
        ctk.CTkLabel(
            cell_dct,
            text="DCT Coefficient Matrix",
            font=FONT_LABEL_SM,
            text_color=ACCENT_HOVER,
        ).pack(pady=(0, 2))
        ctk.CTkLabel(
            cell_dct,
            text="(DC = top-left, red)",
            font=ctk.CTkFont(size=8),
            text_color=TEXT_MUTED,
        ).pack(pady=(0, 4))

        cell_quant = _dct_analysis_cell(1, 0)
        self.dct_quant_matrix_label = ctk.CTkLabel(cell_quant, text="—")
        self.dct_quant_matrix_label.pack(pady=(4, 6))
        ctk.CTkLabel(
            cell_quant,
            text="Quantized DCT Matrix",
            font=FONT_LABEL_SM,
            text_color=ACCENT_HOVER,
        ).pack(pady=(0, 2))
        ctk.CTkLabel(
            cell_quant,
            text="(gray = zeroed)",
            font=ctk.CTkFont(size=8),
            text_color=TEXT_MUTED,
        ).pack(pady=(0, 4))

        cell_zigzag = _dct_analysis_cell(1, 1)
        self.dct_zigzag_label = ctk.CTkLabel(cell_zigzag, text="—")
        self.dct_zigzag_label.pack(pady=(4, 6))
        ctk.CTkLabel(
            cell_zigzag,
            text="Zigzag Scan Result (1D)",
            font=FONT_LABEL_SM,
            text_color=ACCENT_HOVER,
        ).pack(pady=(0, 4))
        self.dct_zigzag_coeffs_label = ctk.CTkLabel(
            cell_zigzag,
            text="First 16 Zigzag Coefficients\n—",
            font=FONT_MONO,
            text_color=TEXT_PRIMARY,
            justify="left",
            wraplength=320,
        )
        self.dct_zigzag_coeffs_label.pack(anchor="w", pady=(0, 4))

        # --- Metrikler (kompakt) ---
        self.dct_metrics_frame = ctk.CTkFrame(self.band_row_dct, fg_color=SURFACE_NEUTRAL, corner_radius=RADIUS_SM)
        self.dct_metrics_frame.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(
            self.dct_metrics_frame,
            text="Metrikler",
            font=FONT_LABEL_SM,
            text_color=ACCENT_HOVER,
        ).pack(anchor="w", padx=8, pady=(6, 2))
        dct_met_row = ctk.CTkFrame(self.dct_metrics_frame, fg_color="transparent")
        dct_met_row.pack(fill="x", padx=8, pady=(0, 8))
        self.dct_metric_psnr = ctk.CTkLabel(dct_met_row, text="PSNR: —", font=FONT_BODY)
        self.dct_metric_psnr.pack(side="left", padx=(0, 14))
        self.dct_metric_mse = ctk.CTkLabel(dct_met_row, text="MSE: —", font=FONT_BODY)
        self.dct_metric_mse.pack(side="left", padx=(0, 14))
        self.dct_metric_ssim = ctk.CTkLabel(dct_met_row, text="SSIM: —", font=FONT_BODY)
        self.dct_metric_ssim.pack(side="left", padx=(0, 14))
        self.dct_metric_cr = ctk.CTkLabel(dct_met_row, text="Compression: —", font=FONT_BODY)
        self.dct_metric_cr.pack(side="left", padx=(0, 14))
        self.dct_metric_file = ctk.CTkLabel(dct_met_row, text="File Size: —", font=FONT_BODY)
        self.dct_metric_file.pack(side="left", padx=(0, 14))

        # --- 3. Frequency Region Visualization ---
        self.dct_freq_section = ctk.CTkFrame(self.band_row_dct, fg_color="transparent")
        self.dct_freq_section.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(
            self.dct_freq_section,
            text="Frequency Region Visualization",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w", padx=4, pady=(0, 4))
        ctk.CTkLabel(
            self.dct_freq_section,
            text="DC + AC bölgelerinin görüntü üzerindeki uzamsal etkisi",
            font=ctk.CTkFont(size=9),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=4, pady=(0, 4))

        self.band_grid_dct = ctk.CTkFrame(self.dct_freq_section, fg_color="transparent")
        self.band_grid_dct.pack(fill="x", padx=0, pady=(0, 4))
        for c in (0, 1):
            self.band_grid_dct.grid_columnconfigure(c, weight=1)
        for r in (0, 1):
            self.band_grid_dct.grid_rowconfigure(r, weight=1)

        self.dct_band_labels: Dict[str, ctk.CTkLabel] = {}
        self.dct_band_title_labels: Dict[str, ctk.CTkLabel] = {}

        # --- DWT satırı ---
        self.band_row_dwt = ctk.CTkFrame(self.band_frame, fg_color="transparent")
        self._lbl_band_row_dwt = ctk.CTkLabel(
            self.band_row_dwt,
            text="2D DWT Subband Structure (JPEG2000)",
            font=FONT_LABEL,
            text_color=ACCENT,
        )
        self._lbl_band_row_dwt.pack(anchor="w", padx=12, pady=(4, 2))
        self.band_grid = ctk.CTkFrame(self.band_row_dwt, fg_color="transparent")  # FR-7
        self.band_grid.pack(fill="x", padx=8, pady=(0, 10))  # FR-7
        self.band_grid.grid_columnconfigure(0, weight=1)
        self.band_grid.grid_columnconfigure(1, weight=1)
        self.band_grid.grid_rowconfigure(0, weight=1)
        self.band_grid.grid_rowconfigure(1, weight=1)

        self.band_labels: Dict[str, ctk.CTkLabel] = {}  # DWT thumbnail (FR-7)
        self.dwt_band_title_labels: Dict[str, ctk.CTkLabel] = {}
        self.dwt_band_cells: Dict[str, ctk.CTkFrame] = {}

        def _make_band_cell(
            row: int,
            col: int,
            name: str,
            grid: ctk.CTkFrame,
            title_store: Dict[str, ctk.CTkLabel],
        ) -> Tuple[ctk.CTkLabel, ctk.CTkFrame]:
            cell = ctk.CTkFrame(grid, fg_color=BG_MAIN, corner_radius=RADIUS_SM)  # FR-7
            cell.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")  # FR-7
            tit = ctk.CTkLabel(
                cell,
                text=name,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=TEXT_SECONDARY,
            )
            tit.pack()
            title_store[name] = tit
            lbl = ctk.CTkLabel(cell, text="—")  # FR-7
            lbl.pack(padx=4, pady=4)  # FR-7
            return lbl, cell

        _dct_cell_layout = (
            (0, 0, "dc_lf", DCT_ZONE_VIZ["dc_lf"]),
            (0, 1, "horiz_ac", DCT_ZONE_VIZ["horiz_ac"]),
            (1, 0, "vert_ac", DCT_ZONE_VIZ["vert_ac"]),
            (1, 1, "diag_ac", DCT_ZONE_VIZ["diag_ac"]),
        )
        for row, col, key, title in _dct_cell_layout:
            lbl, _cell = _make_band_cell(
                row, col, title, self.band_grid_dct, self.dct_band_title_labels
            )
            self.dct_band_labels[key] = lbl

        for row, col, name in ((0, 0, "LL"), (0, 1, "HL"), (1, 0, "LH"), (1, 1, "HH")):
            lbl, cell = _make_band_cell(row, col, name, self.band_grid, self.dwt_band_title_labels)
            self.band_labels[name] = lbl
            self.dwt_band_cells[name] = cell

        self.dwt_mosaic_frame = ctk.CTkFrame(self.band_grid, fg_color=BG_MAIN, corner_radius=RADIUS_SM)
        self.dwt_mosaic_frame.grid(row=0, column=0, rowspan=2, columnspan=2, padx=4, pady=4, sticky="nsew")
        self.dwt_mosaic_title = ctk.CTkLabel(
            self.dwt_mosaic_frame,
            text="Multiresolution decomposition mosaic",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_SECONDARY,
        )
        self.dwt_mosaic_title.pack(pady=(4, 0))
        self.dwt_mosaic_label = ctk.CTkLabel(self.dwt_mosaic_frame, text="—")
        self.dwt_mosaic_label.pack(padx=6, pady=6)
        self._set_dwt_preview_layout("mosaic")

        body.grid_columnconfigure(1, weight=1)

        # --- Alt durum çubuğu (pack sırası: önce body expand; status en altta kalsın diye sonradan pack edilebilir;
        # burada body üstte pack edildiği için status_frame sonra pack bottom ile eklenir) ---
        self.status_frame = ctk.CTkFrame(self, fg_color=ACCENT_DARK, corner_radius=0, height=28)
        self.status_frame.pack(fill="x", side="bottom")
        self.status_frame.pack_propagate(False)
        self._status_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=FONT_BODY,
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        self._status_label.pack(fill="x", padx=16, pady=4)

        self._configure_band_rows_visibility(self.method_var.get())
        self._update_histogram_section_caption(self.method_var.get())
        self._update_analysis_section_title(self.method_var.get())
        self._preserve_scroll_position(self._update_dct_center_layout, self.method_var.get())
        self._update_dwt_controls_visibility(self.method_var.get())

    def _on_center_scroll_event(self, event=None):
        """Scroll bittikten kısa süre sonra TEK SEFERLİK temiz yeniden
        çizim yapar (ara kareler yerine)."""
        if hasattr(self, "_scroll_redraw_job") and self._scroll_redraw_job:
            try:
                self.after_cancel(self._scroll_redraw_job)
            except Exception:
                pass
        self._scroll_redraw_job = self.after(80, self._finalize_scroll_redraw)

    def _finalize_scroll_redraw(self):
        self._scroll_redraw_job = None
        try:
            self.center_scroll._parent_canvas.update_idletasks()
        except Exception:
            pass

    def _luma_y_2d(self) -> np.ndarray:
        """DCT/DWT analizi ve PSNR için tek kanallı luma (float)."""
        if self._luma is None:
            return np.zeros((8, 8), dtype=np.float32)
        lu = self._luma
        if isinstance(lu, np.ndarray) and lu.ndim == 3:
            return lu[:, :, 0].astype(np.float32)
        return np.asarray(lu, dtype=np.float32)

    def _configure_band_rows_visibility(self, method: str) -> None:
        """DCT ve DWT analiz satırlarını yönteme göre göster/gizle."""
        if not hasattr(self, "band_row_dct"):
            return
        self.band_row_dct.pack_forget()
        self.band_row_dwt.pack_forget()
        if method == "JPEG (DCT)":
            self.band_panel_heading.configure(text="JPEG / DCT Analysis")
            self.band_row_dct.pack(fill="x", padx=0, pady=0)
        elif method == "JPEG2000 (DWT)":
            self.band_panel_heading.configure(text="2D DWT Subband Structure (JPEG2000)")
            self.band_row_dwt.pack(fill="x", padx=0, pady=0)
        elif method == "JPEG vs JPEG2000 (karşılaştır)":
            self.band_panel_heading.configure(
                text="Karşılaştırma — üst: DCT katsayı analizi, alt: DWT alt bantları"
            )
            self.band_row_dct.pack(fill="x", padx=0, pady=(0, 2))
            self.band_row_dwt.pack(fill="x", padx=0, pady=(2, 0))
        elif method == "Lossless (Kayıpsız - Tıbbi)":
            self.band_panel_heading.configure(text="Kayıpsız analiz — DWT referans")
            self.band_row_dwt.pack(fill="x", padx=0, pady=0)
        else:
            self.band_panel_heading.configure(text="Analiz önizlemesi")
            self.band_row_dwt.pack(fill="x", padx=0, pady=0)

    def _update_dwt_controls_visibility(self, method: str) -> None:
        """Yalnızca JPEG2000 (DWT) modunda dalgacık/seviye/kuantalama
        kontrollerini göster; DCT ve Lossless'te gizle."""
        if not hasattr(self, "dwt_controls_frame"):
            return
        if "JPEG2000" in method:
            self.dwt_controls_frame.pack(fill="x", padx=0, pady=0)
        else:
            self.dwt_controls_frame.pack_forget()
        if hasattr(self, "btn_wavelet_preview"):
            if method == "JPEG (DCT)":
                self.btn_wavelet_preview.pack(fill="x", padx=8, pady=(0, 12))
            else:
                self.btn_wavelet_preview.pack_forget()
        if hasattr(self, "btn_random_block"):
            if method == "JPEG (DCT)":
                self.btn_random_block.pack(anchor="w", pady=(6, 0))
            else:
                self.btn_random_block.pack_forget()

    def _update_dct_center_layout(self, method: str) -> None:
        """DCT modunda üst şemayı gizle; histogramı analiz panelinden sonra göster."""
        if not hasattr(self, "hist_frame"):
            return
        for w in (
            self.schema_title_label,
            self.dct_explanation_label,
            self.schema_frame,
            self.band_frame,
            self._hist_block_title,
            self._hist_block_sub,
            self.hist_frame,
        ):
            w.pack_forget()

        if method == "JPEG (DCT)":
            self.band_frame.pack(fill="x", padx=16, pady=(8, 0))
            self._hist_block_title.pack(anchor="w", padx=16, pady=(12, 2))
            self._hist_block_sub.pack(anchor="w", padx=16, pady=(0, 4))
            self.hist_frame.pack(fill="x", padx=16, pady=(0, 12))
        else:
            self.schema_title_label.pack(anchor="w", padx=16, pady=(14, 2))
            if "karşılaştır" in method:
                self.dct_explanation_label.pack(anchor="w", padx=16, pady=(0, 4))
            self.schema_frame.pack(fill="x", padx=16, pady=(0, 6))
            self._hist_block_title.pack(anchor="w", padx=16, pady=(4, 2))
            self._hist_block_sub.pack(anchor="w", padx=16, pady=(0, 4))
            self.hist_frame.pack(fill="x", padx=16, pady=(0, 8))
            self.band_frame.pack(fill="x", padx=16, pady=(0, 0))

    def _toggle_dct_pipeline(self) -> None:
        """JPEG pipeline bölümünü aç/kapat."""
        if not hasattr(self, "dct_pipeline_body"):
            return
        self._dct_pipeline_open = not self._dct_pipeline_open
        if self._dct_pipeline_open:
            self.dct_pipeline_body.pack(fill="x", padx=4, pady=(0, 6))
            self.btn_pipeline_toggle.configure(text="JPEG Compression Pipeline ▼")
        else:
            self.dct_pipeline_body.pack_forget()
            self.btn_pipeline_toggle.configure(text="JPEG Compression Pipeline ▶")

    def _draw_dct_pipeline_compact(self, info_dict: Optional[Dict[str, str]] = None) -> None:
        """Kompakt yatay JPEG akış şeması (DCT modu pipeline paneli)."""
        if not hasattr(self, "fig_dct_pipeline"):
            return
        self.fig_dct_pipeline.clear()
        ax = self.fig_dct_pipeline.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.2)
        ax.axis("off")
        self.fig_dct_pipeline.patch.set_facecolor(PANEL_BG)
        self.fig_dct_pipeline.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
        steps = [
            (0.05, 0.25, 1.35, 0.7, "RGB→YUV"),
            (1.55, 0.25, 1.35, 0.7, "8×8 Blocks"),
            (3.05, 0.25, 1.35, 0.7, "DCT"),
            (4.55, 0.25, 1.35, 0.7, "Quantize"),
            (6.05, 0.25, 1.35, 0.7, "Zigzag"),
            (7.55, 0.25, 1.85, 0.7, "Entropy / Bitstream"),
        ]
        for i, (x, y, w, h, lab) in enumerate(steps):
            ax.add_patch(Rectangle(
                (x, y), w, h, linewidth=1.0, edgecolor=TEXT_PRIMARY,
                facecolor="#BBDEFB" if i < 3 else "#FFE0B2" if i < 5 else "#C8E6C9",
                alpha=0.9,
            ))
            ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=8, weight="bold")
            if i < len(steps) - 1:
                nx = steps[i + 1][0]
                ax.annotate("", xy=(nx, y + h / 2), xytext=(x + w + 0.02, y + h / 2),
                            arrowprops=dict(arrowstyle="->", color="#546e7a", lw=1.2))
        figure_canvas_redraw(self.canvas_dct_pipeline)
        if info_dict and hasattr(self, "dct_pipeline_info_label"):
            parts = [
                f"Method: {info_dict.get('method', '—')}",
                f"Quality: {info_dict.get('level_delta', '—')}",
                f"Output: {info_dict.get('export_file', info_dict.get('huffman', '—'))}",
                f"Compression: {info_dict.get('compression', '—')}",
            ]
            self.dct_pipeline_info_label.configure(text="  |  ".join(parts))

    def _update_analysis_section_title(self, method: str) -> None:
        """Şema başlığı ve DCT açıklamasını seçili yönteme göre güncelle."""
        if not hasattr(self, "schema_title_label"):
            return
        if method == "JPEG (DCT)":
            self.schema_title_label.configure(text="JPEG / DCT Analysis")
            self.dct_explanation_label.configure(text="")
        elif method == "JPEG2000 (DWT)":
            self.schema_title_label.configure(text="2D DWT Subband Structure")
            self.dct_explanation_label.configure(text="")
        elif "karşılaştır" in method:
            self.schema_title_label.configure(text="DCT vs DWT — Analiz Panelleri")
            self.dct_explanation_label.configure(text=DCT_EXPLANATION_TEXT)
        elif method == "Lossless (Kayıpsız - Tıbbi)":
            self.schema_title_label.configure(text="2D DWT Subband Structure")
            self.dct_explanation_label.configure(text="")
        else:
            self.schema_title_label.configure(text="2D DWT Subband Structure")
            self.dct_explanation_label.configure(text="")

    def _update_histogram_section_caption(self, method: str) -> None:
        """Histogram başlığını DWT / DCT / karşılaştırma moduna göre güncelle."""
        if not hasattr(self, "_hist_block_title"):
            return
        if method == "JPEG (DCT)":
            self._hist_block_title.configure(text="4. DCT Coefficient Magnitude Distribution")
            self._hist_block_sub.configure(
                text="DC+LF / Horizontal / Vertical / Diagonal-HF AC Coefficients — tüm görüntü blokları"
            )
        elif method == "JPEG2000 (DWT)":
            self._hist_block_title.configure(text="Anlık grafikler — DWT katsayı dağılımı")
            self._hist_block_sub.configure(text="LL + en ince seviye HL, LH, HH")
        elif "karşılaştır" in method:
            self._hist_block_title.configure(text="Anlık grafikler — DWT katsayı dağılımı")
            self._hist_block_sub.configure(
                text="Karşılaştırma: histogram DWT; DCT katsayı analizi üst panelde"
            )
        elif method == "Lossless (Kayıpsız - Tıbbi)":
            self._hist_block_title.configure(text="Anlık grafikler — DWT (referans ayrışma)")
            self._hist_block_sub.configure(text="Kayıpsız işlemde bilgi için DWT analizi")
        else:
            self._hist_block_title.configure(text="Anlık grafikler (katsayı dağılımı)")
            self._hist_block_sub.configure(text="Yönteme göre DWT veya DCT katsayıları")

    def _set_dct_zone_cell_titles(self, *, for_compare: bool) -> None:
        for key in DCT_ZONE_KEYS:
            if key in self.dct_band_title_labels:
                txt = DCT_ZONE_VIZ[key]
                suf = "\n[DCT]" if for_compare else ""
                self.dct_band_title_labels[key].configure(text=txt + suf)

    def _reset_dwt_cell_titles_short(self) -> None:
        for k in ("LL", "HL", "LH", "HH"):
            if k in self.dwt_band_title_labels:
                self.dwt_band_title_labels[k].configure(text=k)

    def _set_dwt_cell_titles_compare(self) -> None:
        for k in ("LL", "HL", "LH", "HH"):
            if k in self.dwt_band_title_labels:
                self.dwt_band_title_labels[k].configure(text=f"{k}\n[DWT]")

    def _on_random_block_clicked(self) -> None:
        if self.method_var.get() != "JPEG (DCT)":
            return
        if self._luma is None:
            return
        try:
            luma_2d = self._luma_y_2d()
            h, w = luma_2d.shape[:2]
            max_i = max(0, h - 8)
            max_j = max(0, w - 8)
            bi = int(np.random.randint(0, max_i // 8 + 1)) * 8
            bj = int(np.random.randint(0, max_j // 8 + 1)) * 8
            self._dct_block_pos = (bi, bj)
            self._draw_example_8x8_block_analysis(
                luma_2d, jpeg_quality=self._last_jpeg_quality
            )
        except Exception as e:
            print(f"[Rastgele Blok] Hata: {e}")

    def _draw_example_8x8_block_analysis(
        self, luma_2d: np.ndarray, jpeg_quality: int = 85
    ) -> None:
        """Tek 8×8 örnek blok üzerinde klasik JPEG akışını adım adım görselleştir."""
        from PIL import Image as PilImage

        block_u8, centered, (bi, bj) = _dct_sample_block_centered(
            luma_2d, block_pos=self._dct_block_pos
        )
        self._dct_block_pos = (bi, bj)

        dct_block = cv2.dct(centered)
        q_mat = _jpeg_luminance_quant_matrix(jpeg_quality)
        quant_block = np.round(dct_block / q_mat).astype(np.int16)

        luma_u8 = _dct_luma_to_u8(luma_2d)
        sel_rgb = _dct_image_with_block_highlight(luma_u8, bi, bj, max_side=220)
        sh, sw = sel_rgb.shape[:2]
        sel_img = PilImage.fromarray(sel_rgb, mode="RGB")
        sel_ctk = ctk.CTkImage(light_image=sel_img, dark_image=sel_img, size=(sw, sh))
        self.dct_block_selector_label.configure(image=sel_ctk, text="")
        self._image_refs["dct_block_selector"] = sel_ctk
        self.dct_block_pos_label.configure(
            text=f"Block position: row {bi}–{bi + 7}, col {bj}–{bj + 7}  |  centered to [-128,127]"
        )

        order = _jpeg_zigzag_indices_8x8()
        zigzag_1d = [int(quant_block[u, v]) for u, v in order]
        first16 = zigzag_1d[:16]
        zline1 = ", ".join(str(x) for x in first16[:8])
        zline2 = ", ".join(str(x) for x in first16[8:16])
        if hasattr(self, "dct_zigzag_coeffs_label"):
            self.dct_zigzag_coeffs_label.configure(
                text=f"First 16 Zigzag Coefficients\n[{zline1},\n {zline2}]"
            )

        # 2×2 kartlar: eşit kutu boyutu; orijinal blok biraz daha büyük ölçeklenir
        panel_display: Dict[str, Tuple[int, int]] = {
            "dct_pixel_block_label": (272, 272),
            "dct_coeff_matrix_label": (288, 288),
            "dct_quant_matrix_label": (288, 288),
            "dct_zigzag_label": (288, 200),
        }
        viz_specs = (
            ("dct_pixel_block_label", _annotate_8x8_matrix_rgb(block_u8, value_fmt="pixel"), "dct_pixel_block"),
            (
                "dct_coeff_matrix_label",
                _annotate_8x8_matrix_rgb(dct_block, highlight_rc=(0, 0)),
                "dct_coeff_mat",
            ),
            (
                "dct_quant_matrix_label",
                _annotate_8x8_matrix_rgb(quant_block.astype(np.float64), zero_gray=True, value_fmt="int"),
                "dct_quant_mat",
            ),
            (
                "dct_zigzag_label",
                _dct_zigzag_result_rgb(quant_block, width=320, height=140),
                "dct_zigzag_result",
            ),
        )
        for label_attr, rgb, ref_key in viz_specs:
            tw, th = panel_display[label_attr]
            img = PilImage.fromarray(rgb, mode="RGB").resize((tw, th), Image.Resampling.NEAREST)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(tw, th))
            getattr(self, label_attr).configure(image=ctk_img, text="")
            self._image_refs[ref_key] = ctk_img

    def _update_dct_metrics_labels(
        self,
        *,
        psnr: float,
        mse: float,
        ssim: float,
        cr_pct: float,
        file_kb: float,
        ratio: float,
    ) -> None:
        """DCT analiz panelindeki metrik satırını güncelle."""
        if not hasattr(self, "dct_metric_psnr"):
            return
        self.dct_metric_psnr.configure(text=f"PSNR: {psnr:.2f} dB")
        self.dct_metric_mse.configure(text=f"MSE: {mse:.4f}")
        if hasattr(self, "dct_metric_ssim"):
            self.dct_metric_ssim.configure(text=f"SSIM: {ssim:.3f}")
        self.dct_metric_cr.configure(text=f"Compression: {ratio:.1f}:1 (↓{cr_pct:.0f}%)")
        self.dct_metric_file.configure(text=f"File Size: {file_kb:.1f} KB")

    def _draw_dct_zone_thumbnails(self, luma_2d: np.ndarray) -> None:
        """8×8 DCT frekans bölgelerinin uzamsal katmanlarını küçük resim olarak göster."""
        from PIL import Image as PilImage

        layers, _pct = _dct_zone_layer_images(luma_2d)
        ts = BAND_THUMB_SIZE
        for key, arr in zip(DCT_ZONE_KEYS, layers):
            vis = np.clip(np.asarray(arr, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
            resample = Image.Resampling.NEAREST if key != "dc_lf" else Image.Resampling.BILINEAR
            img = PilImage.fromarray(vis, mode="L").resize((ts, ts), resample)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(ts, ts))
            self.dct_band_labels[key].configure(image=ctk_img, text="")
            self._image_refs[f"dct_band_{key}"] = ctk_img

    def _draw_dct_zone_histograms(self, luma_2d: np.ndarray, level: int) -> None:
        """DCT frekans bölgelerine göre katsayı büyüklüğü histogramları (2×2) — aktif Y/luma."""
        if not hasattr(self, "fig"):
            return
        if self._luma is None:
            self.fig.clf()
            figure_canvas_redraw(self.canvas)
            return
        try:
            # Her çizimde güncel orijinal luma (sıkıştırılmış/recon veya eski cache değil)
            luma_active = self._luma_y_2d()
            sample_sets = _dct_per_zone_coeff_samples(luma_active)

            self.fig.clf()
            axes = self.fig.subplots(2, 2)
            flat_axes = np.asarray(axes).flatten()
            titles = [DCT_ZONE_FULL[k] for k in DCT_ZONE_KEYS]
            colors = ["blue", "red", "green", "purple"]
            flat_note = _dct_is_nearly_flat_luma(_dct_luma_to_u8(luma_active))

            for zi, (ax, data, title, color) in enumerate(
                zip(flat_axes, sample_sets, titles, colors)
            ):
                band = np.abs(np.asarray(data, dtype=np.float64).ravel())
                n = band.size
                if n > HISTOGRAM_MAX_SAMPLES:
                    step = max(1, (n + HISTOGRAM_MAX_SAMPLES - 1) // HISTOGRAM_MAX_SAMPLES)
                    band = band[::step]
                if flat_note and zi > 0:
                    band = np.array([0.0], dtype=np.float64)
                    ax.hist(band, bins=[-0.5, 0.5], color=color, alpha=0.7, edgecolor="black")
                    ax.set_xlim(-1.0, 1.0)
                elif band.size == 1 and band[0] == 0.0:
                    ax.hist(band, bins=[-0.5, 0.5], color=color, alpha=0.7, edgecolor="black")
                    ax.set_xlim(-1.0, 1.0)
                else:
                    ax.hist(band, bins=56, color=color, alpha=0.7, edgecolor="black")
                ax.set_title(title, fontsize=9, fontweight="bold")
                ax.set_xlabel("|Katsayı|", fontsize=8)
                ax.set_ylabel("Frekans", fontsize=8)
                ax.grid(True, alpha=0.3)

            note = " (sabit görüntü — AC ≈ 0)" if flat_note else ""
            self.fig.suptitle(
                f"DCT Coefficient Magnitude Distribution (Y/luma){note}",
                fontsize=11,
                fontweight="bold",
            )
            self.fig.tight_layout()
            figure_canvas_redraw(self.canvas)
        except Exception as e:
            self._report_error("DCT histogram hatası", e, code="ERR-DCTHIST-01")

    def render_dct_analysis_panel(
        self,
        luma_2d: np.ndarray,
        level: int,
        *,
        jpeg_quality: int = 85,
        psnr: float = 0.0,
        mse: float = 0.0,
        ssim: float = 0.0,
        cr_pct: float = 0.0,
        file_kb: float = 0.0,
        ratio: float = 0.0,
        for_compare: bool = False,
    ) -> None:
        """JPEG/DCT modu: pipeline → örnek blok → frekans bölgeleri → enerji histogramı."""
        self._draw_example_8x8_block_analysis(luma_2d, jpeg_quality=jpeg_quality)
        self._update_dct_metrics_labels(
            psnr=psnr, mse=mse, ssim=ssim, cr_pct=cr_pct, file_kb=file_kb, ratio=ratio
        )
        self._draw_dct_zone_thumbnails(luma_2d)
        self._set_dct_zone_cell_titles(for_compare=for_compare)
        self._draw_dct_zone_histograms(luma_2d, level)

    def _preserve_scroll_position(self, redraw_fn, *args, **kwargs):
        """Scroll pozisyonunu koruyarak bir yeniden-çizim fonksiyonunu çalıştırır."""
        try:
            canvas = self.center_scroll._parent_canvas
            current_pos = canvas.yview()[0]
        except Exception:
            current_pos = None
            canvas = None
        redraw_fn(*args, **kwargs)
        if current_pos is not None and canvas is not None:
            self.after_idle(lambda pos=current_pos, c=canvas: c.yview_moveto(pos))

    def render_dwt_subband_panel(
        self,
        coeffs: Any,
        luma_2d: np.ndarray,
        level: int,
        info_dict: Optional[Dict[str, str]] = None,
        *,
        wavelet: str = "db4",
        for_compare: bool = False,
    ) -> None:
        """JPEG2000/DWT modu: çok seviyeli alt bant mosaic, histogram ve önizleme."""
        coeff_list = _ensure_dwt_coeff_list(coeffs, luma_2d, wavelet=wavelet, level=level)
        viz_level = len(coeff_list) - 1

        self.fig_schema.clear()
        self._draw_static_dwt_schema(
            info_dict=info_dict or {},
            coeffs=coeff_list,
            level=viz_level,
            luma_2d=luma_2d,
            wavelet=wavelet,
        )
        figure_canvas_redraw(self.canvas_schema)
        self._set_dwt_preview_layout("mosaic")
        self._draw_histograms(coeff_list, viz_level)
        self._draw_band_images(coeff_list, viz_level)
        if for_compare:
            self.dwt_mosaic_title.configure(
                text=f"Multiresolution mosaic — L{viz_level} [DWT karşılaştırma]"
            )

    def _draw_dwt_info_table_on_axes(
        self, ax: Any, info_dict: Optional[Dict[str, str]]
    ) -> None:
        """DWT şema paneli — sağ taraftaki bilgi tablosu."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        if not info_dict:
            return

        table_data = [
            ("Yöntem (Method)", info_dict.get("method", "—")),
            ("Seviye (Level)", info_dict.get("level_delta", "—")),
            ("Huffman (katsayı)", info_dict.get("huffman", "—")),
        ]
        if info_dict.get("export_file"):
            table_data.append(("Çıktı dosyası", info_dict["export_file"]))
        table_data.append(("Sıkıştırma (Compression)", info_dict.get("compression", "—")))
        if "ssim" in info_dict:
            table_data.append(("SSIM", info_dict["ssim"]))

        n_rows = len(table_data)
        row_h = 0.88 / max(n_rows, 1)
        col1_w = 0.42
        for i, (label, value) in enumerate(table_data):
            y = 0.94 - (i + 1) * row_h
            ax.add_patch(
                Rectangle(
                    (0.02, y),
                    col1_w,
                    row_h * 0.92,
                    linewidth=1.0,
                    edgecolor="#999999",
                    facecolor=SURFACE_NEUTRAL,
                    alpha=0.9,
                    transform=ax.transAxes,
                    clip_on=True,
                )
            )
            ax.text(
                0.04,
                y + row_h * 0.46,
                label,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=9,
                weight="bold",
                color=TEXT_PRIMARY,
                clip_on=True,
            )
            ax.add_patch(
                Rectangle(
                    (0.02 + col1_w, y),
                    0.96 - col1_w,
                    row_h * 0.92,
                    linewidth=1.0,
                    edgecolor="#999999",
                    facecolor=SURFACE_NEUTRAL,
                    alpha=0.95,
                    transform=ax.transAxes,
                    clip_on=True,
                )
            )
            value_str = str(value)
            if len(value_str) > 30:
                value_str = value_str[:27] + "..."
            ax.text(
                0.04 + col1_w,
                y + row_h * 0.46,
                value_str,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=9,
                color=TEXT_PRIMARY,
                clip_on=True,
            )

    def _resize_schema_figure_for_mosaic(self, mosaic_h: int, mosaic_w: int, n_rows: int = 5) -> None:
        """Mosaic en-boy oranına göre fig_schema boyutunu ayarla."""
        aspect = mosaic_w / max(mosaic_h, 1)
        fig_h = float(np.clip(3.2 + 0.55 * (mosaic_h / 256.0), 3.2, 6.8))
        fig_w = float(np.clip(9.5 + 0.35 * aspect, 9.5, 13.0))
        extra_table_height = max(0, (n_rows - 3)) * 0.35
        fig_h = fig_h + extra_table_height
        self.fig_schema.set_size_inches(fig_w, fig_h)

    def _set_dwt_preview_layout(self, mode: str) -> None:
        """DWT alt bant önizlemesi: 'mosaic' (çok seviyeli) veya 'cells' (lossless 2×2)."""
        if not hasattr(self, "dwt_mosaic_frame"):
            return
        positions = {"LL": (0, 0), "HL": (0, 1), "LH": (1, 0), "HH": (1, 1)}
        if mode == "mosaic":
            self.dwt_mosaic_frame.grid(row=0, column=0, rowspan=2, columnspan=2, padx=4, pady=4, sticky="nsew")
            for cell in getattr(self, "dwt_band_cells", {}).values():
                cell.grid_remove()
        else:
            self.dwt_mosaic_frame.grid_remove()
            for name, cell in getattr(self, "dwt_band_cells", {}).items():
                r, c = positions[name]
                cell.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")

    def _draw_static_dwt_schema(
        self,
        info_dict: Optional[Dict[str, str]] = None,
        coeffs: Optional[List] = None,
        level: int = 2,
        luma_2d: Optional[np.ndarray] = None,
        wavelet: str = "db4",
    ) -> None:
        """Çok seviyeli DWT mosaic (sol) ve bilgi tablosu (sağ). Katsayı yoksa iç içe şema."""
        self.fig_schema.patch.set_facecolor(PANEL_BG)
        n_levels = max(1, int(level))
        # _draw_dwt_info_table_on_axes ile aynı satır sayısı (resize tablo oluşmadan önce)
        if info_dict:
            n_table_rows = (
                4
                + (1 if info_dict.get("export_file") else 0)
                + (1 if "ssim" in info_dict else 0)
            )
        else:
            n_table_rows = 5

        coeff_list: Optional[List] = None
        if coeffs is not None:
            try:
                if luma_2d is not None:
                    coeff_list = _ensure_dwt_coeff_list(coeffs, luma_2d, wavelet=wavelet, level=level)
                elif isinstance(coeffs, (list, tuple)) and len(coeffs) >= 2:
                    coeff_list = list(coeffs)
            except Exception:
                coeff_list = None

        if coeff_list is not None and len(coeff_list) >= 2:
            try:
                mosaic, regions = _build_dwt_multilevel_mosaic(coeff_list)
            except Exception:
                mosaic, regions = None, []
            if mosaic is not None:
                mh, mw = mosaic.shape
                n_levels = len(coeff_list) - 1
                self._resize_schema_figure_for_mosaic(mh, mw, n_rows=n_table_rows)
                self.fig_schema.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.06, wspace=0.18)

                gs = self.fig_schema.add_gridspec(1, 2, width_ratios=[1.4, 1.0])
                ax_m = self.fig_schema.add_subplot(gs[0, 0])
                ax_info = self.fig_schema.add_subplot(gs[0, 1])

                ax_m.imshow(mosaic, cmap="gray", interpolation="nearest", aspect="equal")
                ax_m.set_xlim(-0.5, mw - 0.5)
                ax_m.set_ylim(mh - 0.5, -0.5)

                for reg in regions:
                    rx, ry, rw, rh = reg["x"], reg["y"], reg["w"], reg["h"]
                    band_key = reg["band"]
                    edge = {"LL": ACCENT_HOVER, "HL": "#00695c", "LH": "#f57f17", "HH": DANGER}.get(
                        band_key, TEXT_PRIMARY
                    )
                    ax_m.add_patch(
                        Rectangle(
                            (rx - 0.5, ry - 0.5),
                            rw,
                            rh,
                            fill=False,
                            edgecolor=edge,
                            linewidth=max(0.6, min(1.8, mw / 256.0)),
                        )
                    )
                    fs = max(5.5, min(11.0, min(rw, rh) / max(mw / (3.5 * n_levels), 1.0)))
                    ax_m.text(
                        rx + rw / 2,
                        ry + rh / 2,
                        reg["label"],
                        ha="center",
                        va="center",
                        fontsize=fs,
                        color="#FFEB3B",
                        weight="bold",
                    )

                ax_m.set_title(
                    f"Multiresolution DWT subband mosaic (L{n_levels})",
                    fontsize=10,
                    weight="bold",
                    pad=6,
                )
                ax_m.axis("off")
                self._draw_dwt_info_table_on_axes(ax_info, info_dict)
                return

        # Katsayı yok: ders slaytı tarzı iç içe şema
        extra_table_height = max(0, (n_table_rows - 3)) * 0.35
        self.fig_schema.set_size_inches(11.0, max(3.5, 2.2 + 0.55 * n_levels) + extra_table_height)
        self.fig_schema.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        gs = self.fig_schema.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
        ax_s = self.fig_schema.add_subplot(gs[0, 0])
        ax_info = self.fig_schema.add_subplot(gs[0, 1])

        ax_s.set_xlim(0, 3.2)
        ax_s.set_ylim(0, 3.2)
        ax_s.set_aspect("equal")
        ax_s.axis("off")
        pyramid_size = min(2.85, 1.35 + 0.55 * n_levels)
        y0 = (3.2 - pyramid_size) / 2.0
        _draw_dwt_schematic_pyramid(ax_s, n_levels, 0.15, y0, pyramid_size, 1)
        ax_s.set_title(
            f"DWT subband pyramid (L{n_levels}) — görüntü yükleyin",
            fontsize=10,
            weight="bold",
        )
        self._draw_dwt_info_table_on_axes(ax_info, info_dict)

    def _draw_static_dct_schema(self, info_dict: Optional[Dict[str, str]] = None) -> None:
        """JPEG/DCT akış şeması: 8×8 blok → DCT → kuantalama → zigzag (DWT alt bandı yok)."""
        ax = self.fig_schema.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.axis("off")
        self.fig_schema.patch.set_facecolor(PANEL_BG)
        self.fig_schema.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        flow_steps = [
            (0.15, 1.55, 1.15, 0.85, "RGB→YUV\n(luma)", SURFACE_NEUTRAL),
            (1.45, 1.55, 1.15, 0.85, "8×8\nBlocks", PANEL_BG),
            (2.75, 1.55, 1.15, 0.85, "DCT\n[-128,127]", SURFACE_NEUTRAL),
            (0.15, 0.35, 1.15, 0.85, "Quantize\n(loss)", PANEL_BG),
            (1.45, 0.35, 1.15, 0.85, "Zigzag\nScan", SURFACE_NEUTRAL),
            (2.75, 0.35, 1.15, 0.85, "Entropy\nCoding", PANEL_BG),
        ]
        for x, y, w, h, lab, color in flow_steps:
            r = Rectangle((x, y), w, h, linewidth=1.5, edgecolor=TEXT_PRIMARY, facecolor=color, alpha=0.9)
            ax.add_patch(r)
            ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=9, weight="bold", color=TEXT_PRIMARY)

        zone_boxes = [
            (0.2, 2.45, 0.55, 0.4, "DC", ACCENT_HOVER),
            (0.85, 2.45, 0.55, 0.4, "LF", "#1976d2"),
            (1.5, 2.45, 0.55, 0.4, "Horiz\nAC", "#e65100"),
            (2.15, 2.45, 0.55, 0.4, "Vert\nAC", "#2e7d32"),
            (2.8, 2.45, 0.55, 0.4, "Diag\nAC", "#6a1b9a"),
        ]
        for x, y, w, h, lab, fc in zone_boxes:
            r = Rectangle((x, y), w, h, linewidth=1.0, edgecolor="#455a64", facecolor=fc, alpha=0.25)
            ax.add_patch(r)
            ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=7, weight="bold", color=fc)

        if info_dict:
            table_data = [
                ("Yöntem (Method)", info_dict.get("method", "—")),
                ("JPEG Kalite", info_dict.get("level_delta", "—")),
                ("Çıktı / Bitstream", info_dict.get("huffman", "—")),
            ]
            if info_dict.get("export_file"):
                table_data.append(("Dosya Boyutu", info_dict["export_file"]))
            table_data.append(("Sıkıştırma (Compression)", info_dict.get("compression", "—")))

            tbl_x, tbl_y, tbl_w = 3.8, 0.3, 5.8
            row_h, col1_w = 0.6, 2.8
            col2_w = tbl_w - col1_w
            for i, (label, value) in enumerate(table_data):
                y = tbl_y + (len(table_data) - 1 - i) * row_h
                ax.add_patch(Rectangle((tbl_x, y), col1_w, row_h, linewidth=1.0, edgecolor="#999", facecolor=SURFACE_NEUTRAL, alpha=0.9, clip_on=True))
                ax.text(tbl_x + 0.15, y + row_h / 2, label, ha="left", va="center", fontsize=9.5, weight="bold", color=TEXT_PRIMARY, clip_on=True)
                ax.add_patch(Rectangle((tbl_x + col1_w, y), col2_w, row_h, linewidth=1.0, edgecolor=SEP_COLOR, facecolor=SURFACE_NEUTRAL, alpha=0.95, clip_on=True))
                ax.text(tbl_x + col1_w + 0.15, y + row_h / 2, value, ha="left", va="center", fontsize=9.5, color=TEXT_PRIMARY, clip_on=True)

    def _set_status(self, message: str) -> None:
        """Durum çubuğu metnini güncelle — tüm kullanıcı geri bildirimi buraya yönlendirilebilir."""
        self._status_label.configure(text=message)

    def _report_error(
        self,
        context: str,
        error: Exception,
        *,
        code: Optional[str] = None,
        show_dialog: bool = False,
        dialog_title: str = "Hata",
        threaded: bool = False,
    ) -> None:
        """Hata mesajını tek noktadan raporla: status, opsiyonel dialog ve traceback."""
        prefix = f"[{code}] " if code else ""
        message = f"{prefix}{context}: {error}"
        if threaded:
            self.after(0, lambda: self._set_status(message))
            if show_dialog:
                import tkinter.messagebox as mb
                self.after(0, lambda: mb.showerror(dialog_title, message))
        else:
            self._set_status(message)
            if show_dialog:
                import tkinter.messagebox as mb
                mb.showerror(dialog_title, message)
        traceback.print_exc()

    def _short_path(self, path: str, max_len: int = 48) -> str:
        """Uzun Windows yollarını durum satırında kısalt."""
        if len(path) <= max_len:
            return path
        return "…" + path[-(max_len - 1) :]

    def _update_resolution_captions(self) -> None:
        """Çözünürlük bilgisini göster: orijinal ve yeniden oluşturulan."""
        if self._original_bgr is not None:
            h, w = self._original_bgr.shape[:2]
            cap_orig = f"Çözünürlük: {w} × {h} (Orijinal)"
        else:
            cap_orig = "Çözünürlük: —"

        if self._luma is not None:
            h, w = self._luma.shape[:2]
            cap_recon = f"Çözünürlük: {w} × {h} (Yeniden Oluş.)"
        else:
            cap_recon = "Çözünürlük: —"

        self.lbl_orig_caption.configure(text=cap_orig)
        self.lbl_recon_caption.configure(text=cap_recon)

    def _on_resolution_selected(self, value: str) -> None:
        """Segmented buton değişince hedef boyutu ayarla; görüntü varsa yeniden yükle ve işle."""
        if "1024" in value:
            self._target_size = (1024, 1024)
        else:
            self._target_size = (512, 512)
        self._update_resolution_captions()
        if self._image_path is not None:
            try:
                self._reload_luma()  # Aynı dosyadan yeni boyutta Y kanalı
                self._set_status(f"Çözünürlük değiştirildi: {self._target_size[0]}×{self._target_size[1]} — {self._short_path(self._image_path)}")
                self._process()
            except Exception as e:
                self._report_error("Çözünürlük güncelleme hatası", e, code="ERR-RESIZE-01")
        else:
            self._set_status(f"Hedef çözünürlük: {self._target_size[0]}×{self._target_size[1]} — önce görüntü yükleyin.")

    def _reload_luma(self) -> None:
        """Diskteki _image_path için: BGR yükle → Renkli görüntüler 3B, Grayscale 2B; sol önizleme üret."""
        if self._image_path is None:
            return
        pre = ImagePreprocessor(self._image_path)
        pre.load_image()
        original_bgr = pre.original_bgr
        if original_bgr is None:
            return
        self._original_bgr = original_bgr
        self._original_raw_bytes = int(original_bgr.nbytes)
        
        # Görüntü türünü detect et: Renkli (3B) veya Grayscale (2B)
        if len(self._original_bgr.shape) == 3 and self._original_bgr.shape[2] == 3:
            # Renkli görüntü: BGR 3-channel kalsın
            pre.convert_to_luma(force_grayscale=False)
            self._is_color_image = True
        else:
            # Grayscale görüntü: Y channel olarak işle
            pre.convert_to_luma(force_grayscale=True)
            self._is_color_image = False
            
        self._luma = pre.luma_channel
        self._display_orig = self._bgr_to_display(self._original_bgr)  # Orijinal boyut
        self._set_image_label(self.lbl_orig, self._display_orig)
        self._update_resolution_captions()  # Çözünürlük bilgisini güncelle
        
        # Eski sıkıştırılmış görüntüyü temizle
        self._display_recon = None
        self.lbl_recon.configure(text="Önce görüntü yükleyin\n\nSonra Uygula ile işleyin", image="")

    def _reset_metrics_display(self) -> None:
        """Görüntü yokken veya hata durumunda metrik hücrelerini sıfırla."""
        self.metric_val_psnr.configure(text="—")
        self.metric_val_mse.configure(text="—")
        self.metric_val_cr.configure(text="—")
        self.metric_val_huffman.configure(text="—")
        self.metric_quality.configure(text="Kalite: —")
        # DWT alt bant görüntülerini temizle (FR-7: dönüşüm görselleştirme paneli)
        if hasattr(self, "band_labels"):  # FR-7: ilk kurulum tamamlanmadan çağrılabilir
            for lbl in self.band_labels.values():  # FR-7
                lbl.configure(image="", text="—")  # FR-7
        if hasattr(self, "dwt_mosaic_label"):
            self.dwt_mosaic_label.configure(image="", text="—")
        if hasattr(self, "dwt_mosaic_title"):
            self.dwt_mosaic_title.configure(text="Multiresolution decomposition mosaic")
        if hasattr(self, "dct_band_labels"):
            for lbl in self.dct_band_labels.values():
                lbl.configure(image="", text="—")
        for attr in (
            "dct_block_selector_label",
            "dct_pixel_block_label",
            "dct_coeff_matrix_label",
            "dct_quant_matrix_label",
            "dct_zigzag_label",
        ):
            if hasattr(self, attr):
                getattr(self, attr).configure(image="", text="—")
        if hasattr(self, "dct_block_pos_label"):
            self.dct_block_pos_label.configure(text="Block position: —")
        if hasattr(self, "dct_zigzag_coeffs_label"):
            self.dct_zigzag_coeffs_label.configure(text="First 16 Zigzag Coefficients\n—")
        if hasattr(self, "dct_pipeline_info_label"):
            self.dct_pipeline_info_label.configure(text="—")
        if hasattr(self, "dct_metric_psnr"):
            self.dct_metric_psnr.configure(text="PSNR: —")
            self.dct_metric_mse.configure(text="MSE: —")
            if hasattr(self, "dct_metric_ssim"):
                self.dct_metric_ssim.configure(text="SSIM: —")
            self.dct_metric_cr.configure(text="Compression: —")
            self.dct_metric_file.configure(text="File Size: —")
        self._dct_block_pos = None

    @staticmethod
    def _quality_text_from_psnr(psnr: float) -> str:
        if psnr >= 35.0:
            return "Kalite: iyi"
        if psnr >= 28.0:
            return "Kalite: orta"
        return "Kalite: düşük"

    @staticmethod
    def _history_method_label(method: str) -> str:
        """Uzun yöntem adlarını kartta kısa/okunur hale getir."""
        if method == "JPEG (DCT)":
            return "JPEG"
        if method == "JPEG2000 (DWT)":
            return "JPEG2000"
        if method == "JPEG vs JPEG2000 (karşılaştır)":
            return "Karşılaştır"
        if method == "Lossless (Kayıpsız - Tıbbi)":
            return "Lossless"
        return method

    def _add_history_entry(
        self,
        method: str,
        level: int,
        history_target_text: str,
        psnr: float,
        mse: float,
        ssim: float,
        huffman_text: str,
        filename: str = "—",
    ) -> None:
        """Yeni işlem sonucunu geçmiş listesine ekle ve görünümü yenile."""
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "method": self._history_method_label(method),
            "level": level,
            "history_target_text": history_target_text,
            "psnr": round(psnr, 2),
            "mse": round(mse, 4),
            "ssim": round(ssim, 3),
            "huffman": huffman_text,
            "filename": filename,
        }
        self._run_history.append(entry)
        self._run_history = self._run_history[-self._max_history_items :]
        self._refresh_history_view()

    def _refresh_history_view(self) -> None:
        """Geçmiş kartlarını en yeni üstte olacak şekilde yeniden çiz."""
        if not hasattr(self, "history_scroll"):
            return

        for child in self.history_scroll.winfo_children():
            child.destroy()

        if not self._run_history:
            ctk.CTkLabel(
                self.history_scroll,
                text="Henüz işlem kaydı yok.",
                text_color=TEXT_MUTED,
                font=FONT_BODY,
            ).pack(anchor="w", padx=6, pady=6)
            return

        for entry in reversed(self._run_history):
            card = ctk.CTkFrame(self.history_scroll, fg_color=BG_MAIN, corner_radius=RADIUS_SM)
            card.pack(fill="x", padx=2, pady=4)

            ctk.CTkLabel(
                card,
                text=f"{entry['time']}  |  {entry['method']}",
                font=FONT_LABEL,
                text_color=TEXT_PRIMARY,
                anchor="w",
            ).pack(fill="x", padx=8, pady=(6, 0))

            ctk.CTkLabel(
                card,
                text=f"📄 {entry['filename']}",
                font=FONT_HINT,
                text_color=ACCENT_HOVER,
                anchor="w",
            ).pack(fill="x", padx=8, pady=(2, 0))

            text_target = entry.get("history_target_text", f"Hedef BPP: {entry.get('compression_coeff', 0):.2f}")
            ctk.CTkLabel(
                card,
                text=f"Seviye: L{entry['level']}   {text_target}",
                font=FONT_BODY,
                text_color=TEXT_SECONDARY,
                anchor="w",
            ).pack(fill="x", padx=8, pady=(0, 0))

            ctk.CTkLabel(
                card,
                text=(
                    f"PSNR: {entry['psnr']:.2f} dB  MSE: {entry['mse']:.4f}  "
                    f"SSIM: {entry.get('ssim', 0.0):.3f}"
                ),
                font=FONT_BODY,
                text_color=TEXT_PRIMARY,
                anchor="w",
            ).pack(fill="x", padx=8, pady=(0, 0))

            ctk.CTkLabel(
                card,
                text=f"Huffman: {entry['huffman']}",
                font=FONT_BODY,
                text_color=TEXT_PRIMARY,
                anchor="w",
            ).pack(fill="x", padx=8, pady=(0, 6))

    def _clear_history(self) -> None:
        """Kullanıcı isteğiyle işlem geçmişini temizle."""
        self._run_history.clear()
        self._refresh_history_view()

    def _toggle_history_visibility(self) -> None:
        """Geçmiş kart alanını aç/kapat."""
        if not hasattr(self, "history_scroll"):
            return

        self._history_visible = not self._history_visible
        if self._history_visible:
            self.history_scroll.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self.btn_clear_history.pack(anchor="e", padx=8, pady=(0, 8))
            self.btn_toggle_history.configure(text="Gizle")
            self.history_frame.configure(height=0)
        else:
            self.history_scroll.pack_forget()
            self.btn_clear_history.pack_forget()
            self.btn_toggle_history.configure(text="Göster")
            # Panel gerçekten katlansın (boşluk bırakmasın)
            self.history_frame.configure(height=52)

    def _draw_band_images(self, coeffs: list, level: int) -> None:
        """
        Çok seviyeli DWT mosaic önizlemesi (yalnızca görselleştirme; katsayılar değişmez).
        """
        from PIL import Image as PilImage

        self._set_dwt_preview_layout("mosaic")
        try:
            mosaic, regions = _build_dwt_multilevel_mosaic(coeffs)
        except Exception as e:
            self._report_error("DWT mosaic önizleme hatası", e, code="ERR-DWT-MOS")
            return

        arr_u8 = (np.clip(mosaic, 0.0, 1.0) * 255.0).astype(np.uint8)
        mh, mw = arr_u8.shape
        max_side = min(BAND_THUMB_SIZE * 2, 420)
        scale = max_side / max(mh, mw, 1)
        tw = max(1, int(mw * scale))
        th = max(1, int(mh * scale))
        img = PilImage.fromarray(arr_u8, mode="L").resize((tw, th), Image.Resampling.BILINEAR)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(tw, th))
        self.dwt_mosaic_label.configure(image=ctk_img, text="")
        n_sub = len(regions)
        self.dwt_mosaic_title.configure(
            text=f"Multiresolution mosaic — L{level} ({n_sub} alt bant)"
        )
        self._image_refs["dwt_mosaic"] = ctk_img

    def _set_placeholders_empty(self) -> None:
        """Başlangıçta görüntü etiketlerinde metin göster (image= kaldır)."""
        self.lbl_orig.configure(
            image="",
            text="Henüz görüntü yok\n\nGörüntü Yükle’ye tıklayın",
            text_color=TEXT_MUTED,
        )
        self.lbl_recon.configure(
            image="",
            text="Önce görüntü yükleyin\n\nSonra Uygula ile işleyin",
            text_color=TEXT_MUTED,
        )

    @staticmethod
    def _coeff_range_text(is_cr_mode: bool) -> str:
        lo, hi = compression_coeff_bounds(is_cr_mode)
        if is_cr_mode:
            return f"İzin verilen aralık: {lo:g} – {hi:g} (örn. 10 = 10:1)"
        return f"İzin verilen aralık: {lo:g} – {hi:g} BPP"

    def _register_coeff_entry_validation(self) -> None:
        """Yalnızca sayısal giriş ve üst sınır aşımını tuş anında engelle."""
        inner = getattr(self.compression_coeff_entry, "_entry", self.compression_coeff_entry)
        vcmd = (self.register(self._validate_coeff_input), "%P")
        inner.configure(validate="key", validatecommand=vcmd)

    def _validate_coeff_input(self, proposed: str) -> bool:
        if proposed == "":
            return True
        normalized = proposed.strip().replace(",", ".")
        if normalized == ".":
            return True
        if not re.fullmatch(r"\d*\.?\d*", normalized):
            return False
        try:
            value = float(normalized)
        except ValueError:
            return False
        _, hi = compression_coeff_bounds(self._is_cr_mode())
        return value <= hi

    def _get_compression_coefficient(self) -> float:
        """Metin kutusundaki hedef değeri alır; geçersiz/boş veya aralık dışı → sınırlanmış varsayılan."""
        is_cr = self._is_cr_mode()
        default = CR_DEFAULT if is_cr else BPP_DEFAULT
        try:
            t = self.compression_coeff_entry.get().strip().replace(",", ".")
            if not t:
                return default
            v = float(t)
        except ValueError:
            return default
        return clamp_compression_coeff(v, is_cr)

    def _normalize_compression_coeff_entry(self) -> None:
        """Odak kaybedince veya Enter ile kutuyu makul aralıktaki değere sıkıştır."""
        is_cr = self._is_cr_mode()
        raw = self.compression_coeff_entry.get().strip().replace(",", ".")
        default = CR_DEFAULT if is_cr else BPP_DEFAULT
        try:
            v = float(raw) if raw else default
        except ValueError:
            v = default
        clamped = clamp_compression_coeff(v, is_cr)
        self.compression_coeff_entry.delete(0, "end")
        self.compression_coeff_entry.insert(0, format_compression_coeff(clamped, is_cr))
        if hasattr(self, "coeff_range_label"):
            self.coeff_range_label.configure(text=self._coeff_range_text(is_cr))
        self._update_quant_hint_labels(clamped, self.method_var.get())

    def _schedule_process_debounced(self) -> None:
        """Görüntü yüklüyken sıkıştırmayı tekilleştir (debounce — tuş spam'i önler)."""
        if self._luma is None:
            return
        if hasattr(self, "_debounce_id"):
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(PIPELINE_DEBOUNCE_MS, self._on_apply_clicked)

    def _is_cr_mode(self) -> bool:
        """True if Compression Ratio mode is selected."""
        return getattr(self, 'comp_mode_var', None) is not None and self.comp_mode_var.get() == "Compression Ratio"

    def _on_comp_mode_change(self, _value: str = "") -> None:
        """Sıkıştırma modu değişince etiket, placeholder ve aralık kısıtını güncelle."""
        is_cr = self._is_cr_mode()
        lo, hi = compression_coeff_bounds(is_cr)
        if is_cr:
            self.hardness_title.configure(text="Hedef Sıkıştırma Oranı (örn. 10 = 10:1)")
            cur = self._get_compression_coefficient()
            if cur < CR_MIN or cur > CR_MAX:
                cur = CR_DEFAULT
        else:
            self.hardness_title.configure(text="Hedef BPP (Bits Per Pixel)")
            cur = self._get_compression_coefficient()
            if cur < BPP_MIN or cur > BPP_MAX:
                cur = BPP_DEFAULT
        cur = clamp_compression_coeff(cur, is_cr)
        self.compression_coeff_entry.configure(placeholder_text=f"{lo:g} – {hi:g}")
        self.compression_coeff_entry.delete(0, "end")
        self.compression_coeff_entry.insert(0, format_compression_coeff(cur, is_cr))
        if hasattr(self, "coeff_range_label"):
            self.coeff_range_label.configure(text=self._coeff_range_text(is_cr))
        self._register_coeff_entry_validation()
        self._on_param_change()

    def _update_quant_hint_labels(self, q: float, method: str) -> None:
        """Sadece katsayıya bağlı etiket/ipucu (dalga/seviye ile ilgili blokları çalıştırmaz)."""
        if self._is_cr_mode():
            self.quant_label.configure(text=f"Hedef Oran: {q:.1f}:1  (dosya boyutu ÷ {q:.1f})")
            if method == "Lossless (Kayıpsız - Tıbbi)":
                self.quant_hint.configure(text="Kayıpsız modda oran hedefi dikkate alınmaz.")
            elif method == "JPEG (DCT)":
                self.quant_hint.configure(text="JPEG kalitesi hedef boyuta ulaşmak için binary search ile ayarlanır.")
            elif method == "JPEG2000 (DWT)":
                self.quant_hint.configure(text="Kuantalama adımı hedef boyuta binary search ile ayarlanır.")
            else:
                self.quant_hint.configure(text="Her iki yöntem de hedef boyuta binary search ile ayarlanır.")
        else:
            self.quant_label.configure(text=f"Hedef BPP: {q:.2f} bits/pixel")
            if method == "JPEG (DCT)":
                self.quant_hint.configure(text="JPEG kalitesi hedefe ulaşmak için otomatik ayarlanır.")
            elif method == "JPEG2000 (DWT)":
                self.quant_hint.configure(text="Kuantalama adımı hedefe ulaşmak için otomatik ayarlanır.")
            elif method == "Lossless (Kayıpsız - Tıbbi)":
                self.quant_hint.configure(text="Kayıpsız modda BPP hedefi dikkate alınmaz, maksimum sıkıştırma yapılır.")
            else:
                self.quant_hint.configure(text="Karşılaştırma: İki yöntem de aynı hedef BPP'ye ulaşmaya çalışır.")

    def _on_coeff_entry_key(self, _event=None) -> None:
        """Katsayı yazılırken dalgacık ipuçlarını yeniden çizmeden etiket güncelle + debounce."""
        q = self._get_compression_coefficient()
        self._update_quant_hint_labels(q, self.method_var.get())
        self._schedule_process_debounced()

    def _on_method_change(self, _choice: Optional[str] = None) -> None:
        """Yönteme göre DWT kontrolleri ve ipuçları güncelle."""
        m = self.method_var.get()
        
        use_dwt = "JPEG2000" in m or "karşılaştır" in m or "Lossless" in m
        state = "normal" if use_dwt else "disabled"
        self.level_slider.configure(state=state)
        self.wavelet_menu.configure(state=state)

        self._update_dwt_controls_visibility(m)
        self._preserve_scroll_position(self._update_dct_center_layout, m)
        
        if self._is_cr_mode():
            self.hardness_title.configure(text="Hedef Sıkıştırma Oranı (örn. 10 = 10:1)")
        else:
            self.hardness_title.configure(text="Hedef BPP (Bits Per Pixel)")

        self._configure_band_rows_visibility(m)
        self._update_histogram_section_caption(m)
        self._update_analysis_section_title(m)
        self._on_param_change()

    def _on_param_change(self) -> None:
        """Sıkıştırma katsayısı / seviye / dalgacık değişince etiketleri güncelle; görüntü hazırsa işlemi yenile."""
        lv = int(self.level_slider.get())
        self.level_var.set(lv)
        self.level_label.configure(text=f"Seviye: {lv}")
        q = self._get_compression_coefficient()
        
        # Sertlik etiketini moda göre değiştirelim
        method = self.method_var.get()
        
        # Dalgacık ve Seviye ipuçlarını güncelle (DWT modunda)
        if "JPEG2000" in method or "karşılaştır" in method:
            # Dalgacık Tipi İpucu - Detaylı
            wv = self.wavelet_var.get()
            wavelet_hints = {
                "haar": "⚡ Haar\n  Hız: Çok Hızlı\n  Kalite: Düşük\n  Detay: Az (Keskin Geçiş)",
                "db2": "📊 db2 (Daubechies 2)\n  Hız: Hızlı\n  Kalite: Düşük\n  Detay: Sınırlı",
                "db4": "🌊 db4 (Daubechies 4) ⭐ ÖNERİLEN\n  Hız: Dengeli\n  Kalite: İyi\n  Detay: Pürüzsüz",
                "db6": "🎯 db6 (Daubechies 6)\n  Hız: Yavaş\n  Kalite: Yüksek\n  Detay: Ince Detaylar",
                "db8": "🔬 db8 (Daubechies 8)\n  Hız: Çok Yavaş\n  Kalite: Çok Yüksek\n  Detay: Kapsamlı Analiz",
                "db12": "💠 db12 (Daubechies 12)\n  Hız: Orta\n  Kalite: Yüksek\n  Detay: Çok İnce",
                "db32": "🧬 db32 (Daubechies 32)\n  Hız: Aşırı Yavaş\n  Kalite: Olağanüstü\n  Detay: Maksimum Ayrışma",
                "bior4.4": "📐 bior4.4 (Biorthogonal)\n  Hız: Orta\n  Kalite: JPEG2000 Uyumlu\n  Detay: Simetrik",
                "prqmf4": "🔗 PR-QMF (4-Tap)\n  Hız: Orta\n  Kalite: İyi\n  Detay: Perfect Reconstruction Filtre Bankası",
                "prqmf6": "🔗 PR-QMF (6-Tap)\n  Hız: Orta\n  Kalite: İyi\n  Detay: Daha Uzun Filtre, Daha Keskin Ayrışma",
            }
            hint_text = wavelet_hints.get(wv, "Dalgacık tipi seçildi.")
            self.wavelet_hint.configure(text=hint_text)
            
            # Seviye İpucu
            if lv == 1:
                self.level_hint.configure(text="📉 1 Seviye: Az sıkıştırma.\n Detaylar daha çok korunur.")
            elif lv == 2:
                self.level_hint.configure(text="⚖️ 2 Seviye: Dengeli.\n Standart sıkıştırma oranı.")
            else:
                self.level_hint.configure(text="🚀 3 Seviye: Yüksek sıkıştırma.\n Dosya boyutu en küçüğe iner.")
        else:
            # DCT veya Lossless modunda bu hintleri temizle veya özel metin yaz
            if method == "JPEG (DCT)":
                self.wavelet_hint.configure(text="DCT modunda dalgacık kullanılmaz.")
                self.level_hint.configure(text="DCT'de 8x8 bloklar sabittir.")
            elif method == "Lossless (Kayıpsız - Tıbbi)":
                self.wavelet_hint.configure(text="Kayıpsız modda dalgacık yok.")
                self.level_hint.configure(text="Doğrudan piksel sıkıştırma.")
            else:
                self.wavelet_hint.configure(text="")
                # Level hint zaten _on_method_change'de desc_text olarak setleniyor

        self._update_quant_hint_labels(q, method)
        self._schedule_process_debounced()

    def _on_quant_mode_change(self, choice: str = "") -> None:
        """Kuantalama modu değişimi — snap'e aktarılır ve görüntü yeniden işlenir."""
        self._quant_mode = "uniform" if choice == "Uniform" else "scalar"
        self._schedule_process_debounced()

    def _on_manual_step_toggle(self) -> None:
        """Manuel adım anahtarı: entry'yi aç/kapat."""
        if self.manual_step_toggle.get():
            self.manual_step_entry.configure(state="normal")
        else:
            self.manual_step_entry.configure(state="disabled")

    def _on_wavelet_changed(self, _choice: str = "") -> None:
        """Dalgacık seçildiğinde hint'i güncelle ve görüntüyü hemen yeniden işle."""
        wv = self.wavelet_var.get()
        wavelet_hints = {
            "haar": "⚡ Haar\n  Hız: Çok Hızlı\n  Kalite: Düşük\n  Detay: Az (Keskin Geçiş)",
            "db2": "📊 db2 (Daubechies 2)\n  Hız: Hızlı\n  Kalite: Düşük\n  Detay: Sınırlı",
            "db4": "🌊 db4 (Daubechies 4) ⭐ ÖNERİLEN\n  Hız: Dengeli\n  Kalite: İyi\n  Detay: Pürüzsüz",
            "db6": "🎯 db6 (Daubechies 6)\n  Hız: Yavaş\n  Kalite: Yüksek\n  Detay: Ince Detaylar",
            "db8": "🔬 db8 (Daubechies 8)\n  Hız: Çok Yavaş\n  Kalite: Çok Yüksek\n  Detay: Kapsamlı Analiz",
            "db12": "💠 db12 (Daubechies 12)\n  Hız: Orta\n  Kalite: Yüksek\n  Detay: Çok İnce",
            "db32": "🧬 db32 (Daubechies 32)\n  Hız: Aşırı Yavaş\n  Kalite: Olağanüstü\n  Detay: Maksimum Ayrışma",
            "bior4.4": "📐 bior4.4 (Biorthogonal)\n  Hız: Orta\n  Kalite: JPEG2000 Uyumlu\n  Detay: Simetrik",
            "prqmf4": "🔗 PR-QMF (4-Tap)\n  Hız: Orta\n  Kalite: İyi\n  Detay: Perfect Reconstruction Filtre Bankası",
            "prqmf6": "🔗 PR-QMF (6-Tap)\n  Hız: Orta\n  Kalite: İyi\n  Detay: Daha Uzun Filtre, Daha Keskin Ayrışma",
        }
        hint_text = wavelet_hints.get(wv, "Dalgacık tipi seçildi.")
        self.wavelet_hint.configure(text=hint_text)
        self._update_quant_hint_labels(self._get_compression_coefficient(), self.method_var.get())

        if self._luma is not None:
            if hasattr(self, "_debounce_id"):
                try:
                    self.after_cancel(self._debounce_id)
                except Exception:
                    pass
            self._start_process_async(status_prefix=f"Dalgacık: {wv}")

    def _snapshot_process_inputs(self) -> Dict[str, Any]:
        """UI thread: parametreleri ve görüntü kopyalarını dondur (worker için)."""
        coeff_val = self._get_compression_coefficient()
        cr_mode = self._is_cr_mode()
        num_pixels = int(self._luma.shape[0] * self._luma.shape[1])  # type: ignore[union-attr]
        orig_bytes = self._get_original_file_bytes()
        if cr_mode:
            target_ratio = coeff_val
            target_size_bytes = max(1, int(orig_bytes / target_ratio)) if orig_bytes else 1
            target_bpp = (target_size_bytes * 8) / max(1, num_pixels)
            hardness = target_ratio
        else:
            target_bpp = coeff_val
            target_ratio = 0.0
            target_size_bytes = 0
            hardness = target_bpp
        manual_step_value = None
        try:
            raw_step = self.manual_step_entry.get().strip()
            if raw_step:
                manual_step_value = float(raw_step)
                manual_step_value = min(manual_step_value, MANUAL_DELTA_MAX)
        except (TypeError, ValueError):
            manual_step_value = None
        return {
            "method": self.method_var.get(),
            "coeff_val": coeff_val,
            "cr_mode": cr_mode,
            "level": int(self.level_slider.get()),
            "wavelet": self.wavelet_var.get(),
            "num_pixels": num_pixels,
            "orig_file_bytes": orig_bytes,
            "target_ratio": target_ratio,
            "target_size_bytes": target_size_bytes,
            "target_bpp": target_bpp,
            "hardness": hardness,
            "luma": np.array(self._luma, copy=True),
            "original_bgr": np.array(self._original_bgr, copy=True),
            "is_color_image": bool(self._is_color_image),
            "image_path": self._image_path,
            "quant_mode": self._quant_mode,
            "manual_step_enabled": bool(self.manual_step_toggle.get()),
            "manual_step_value": manual_step_value,
        }

    @contextlib.contextmanager
    def _use_process_snap(self, snap: Dict[str, Any]):
        """Worker thread: hesaplama sırasında snap verisini geçici olarak bağla."""
        prev = (
            self._luma,
            self._original_bgr,
            self._image_path,
            self._is_color_image,
            self._snap_orig_bytes_override,
        )
        self._luma = snap["luma"]
        self._original_bgr = snap["original_bgr"]
        self._image_path = snap["image_path"]
        self._is_color_image = snap["is_color_image"]
        self._snap_orig_bytes_override = int(snap["orig_file_bytes"])
        try:
            yield
        finally:
            (
                self._luma,
                self._original_bgr,
                self._image_path,
                self._is_color_image,
                self._snap_orig_bytes_override,
            ) = prev

    def _start_process_async(self, status_prefix: str = "İşleniyor") -> None:
        """Sıkıştırmayı arka planda çalıştır; UI donmasın."""
        if self._luma is None or self._original_bgr is None:
            self._reset_metrics_display()
            return

        method = self.method_var.get()
        self._configure_band_rows_visibility(method)
        self._update_histogram_section_caption(method)

        self._process_job_id += 1
        job_id = self._process_job_id
        snap = self._snapshot_process_inputs()
        self._set_status(f"{status_prefix}… (hesaplama arka planda)")
        self.update_idletasks()

        def _worker() -> None:
            with self._process_lock:
                if job_id != self._process_job_id:
                    return
                try:
                    with self._use_process_snap(snap):
                        result = self._process_compute_body(snap)
                    result["ok"] = True
                except Exception as exc:
                    result = {
                        "ok": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
            self.after(0, lambda: self._process_on_complete(result, snap, job_id))

        threading.Thread(target=_worker, daemon=True, name="compression-worker").start()

    def _process_on_complete(self, result: Dict[str, Any], snap: Dict[str, Any], job_id: int) -> None:
        """Ana thread: worker sonucunu arayüze uygula."""
        if job_id != self._process_job_id:
            return
        if not result.get("ok"):
            err = result.get("error", "Bilinmeyen hata")
            print(result.get("traceback", err))
            self._set_status(f"Hata: {err}")
            self.metric_note.configure(text=str(err)[:200])
            return
        if result.get("error"):
            self._set_status(result["error"])
            if result.get("lossless_import"):
                self.metric_note.configure(text=f"Lossless import hata: {_PIPELINE_IMPORT_ERROR}")
            return
        self._process_apply_ui(result, snap)

    def _on_apply_clicked(self) -> None:
        """Parametrelerle yeniden işle (arka plan thread)."""
        self._start_process_async()

    def _on_wavelet_preview_clicked(self) -> None:
        if self.method_var.get() != "JPEG (DCT)":
            return
        if self._luma is None:
            return
        try:
            num_pixels = self._luma.shape[0] * self._luma.shape[1]
            # JPEG'te dalgacık UI gizli, sabit haar + level=2 kullan
            wavelet = "haar"
            level = 2
            # JPEG'in hedeflediği bpp'yi kullan (aynı hedef, "yaklaşık")
            target_bpp = self._get_compression_coefficient() if not self._is_cr_mode() else 1.0
            quant_mode = getattr(self, "_quant_mode", "scalar")
            if self._is_color_image:
                q_step = self._find_best_dwt_q_step_color_bpp(
                    target_bpp, num_pixels, wavelet, level, mode=quant_mode
                )
            else:
                q_step = self._find_best_dwt_q_step(
                    target_bpp, num_pixels, wavelet, level,
                    channel_multiplier=1, mode=quant_mode,
                )
            eval_result = self._evaluate_dwt_jpeg2000_step(
                q_step, wavelet, level, return_details=True, mode=quant_mode
            )
            recon = eval_result.get("recon_luma")
            if recon is None:
                return
            psnr = eval_result.get("psnr", 0.0)
            compressed_bytes = eval_result.get("compressed_bytes", 0)
            actual_bpp = (compressed_bytes * 8) / max(1, num_pixels)
            if isinstance(recon, np.ndarray) and recon.ndim == 3:
                preview = cv2.cvtColor(np.clip(recon, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            else:
                preview = recon
            self._wavelet_preview_win = WaveletPreviewWindow(self, preview, actual_bpp, psnr)
        except Exception as e:
            print(f"[Wavelet Önizle] Hata: {e}")

    def _save_compressed_image(self) -> None:
        """Sıkıştırılmış/yeniden oluşturulmuş görüntüyü diske kaydeder."""
        if not hasattr(self, '_recon_luma') or self._recon_luma is None:
            import tkinter.messagebox as mb
            mb.showwarning("Uyarı", "Önce bir görüntü yükleyin ve Uygula butonuna basın.")
            return
        
        try:
            import tkinter.filedialog as fd
            from PIL import Image
            import os
            
            # Çıktı daima JPEG olsun
            ext = ".jpg"
            types = [("JPEG", "*.jpg;*.jpeg"), ("Tüm dosyalar", "*.*")]
            
            path = fd.asksaveasfilename(
                defaultextension=ext,
                filetypes=types,
                title="Sıkıştırılmış Görseli Kaydet",
                parent=self.winfo_toplevel(),
            )
            
            if not path:
                return

            out_ext = os.path.splitext(path)[1].lower()
            if out_ext not in [".jpg", ".jpeg", ".jfif"]:
                path = f"{os.path.splitext(path)[0]}.jpg"
                out_ext = ".jpg"

            # Sıkıştırılmış byte varsa mümkün olduğunca aynen yaz.
            # Uzantı uyuşmazlığında bile JPEG/JPEG2000 akışını koruyarak "yeniden encode" kaynaklı
            # dosya şişmesini engeller.
            last_bytes = getattr(self, "_last_encoded_image_bytes", None)
            last_ext = getattr(self, "_last_encoded_image_ext", None)

            if last_bytes and last_ext:
                if last_ext == ".jpg":
                    with open(path, "wb") as f:
                        f.write(last_bytes)
                    self._set_status(
                        f"✓ Kaydedildi (sıkıştırılmış JPEG akışı): {path} ({len(last_bytes)/1024:.1f} KB)"
                    )
                    return
            
            # Sıkıştırılmış byte yoksa veya format uymuyor: PIL ile kaydet
            # Optimize sıkıştırma ile kaydet
            img_data = np.clip(self._recon_luma, 0, 255).astype(np.uint8)
            # Renkli (BGR) ve gri çıktıları doğru modda kaydet — tek kanal L ile RGB karışmasını önler
            if img_data.ndim == 3 and img_data.shape[2] >= 3:
                rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb, mode="RGB")
            elif img_data.ndim == 3:
                pil_img = Image.fromarray(img_data[:, :, 0], mode="L")
            else:
                pil_img = Image.fromarray(img_data, mode="L")
            
            # Çıktı daima JPEG encode edilir
            save_kwargs = {"quality": 75, "optimize": True}
            pil_img.save(path, **save_kwargs)
            
            self._set_status(f"✓ Kaydedildi: {path}")
                
        except Exception as e:
            self._report_error("Kayıt işleminde hata", e, code="ERR-SAVE-01", show_dialog=True)

    def _open_recon_zoom_view(self) -> None:
        """Sıkıştırılmış görüntüyü yumuşatmasız yakınlaştırma ile ayrı pencerede göster."""
        if not hasattr(self, "_recon_luma") or self._recon_luma is None:
            import tkinter.messagebox as mb
            mb.showwarning("Uyarı", "Önce görüntüyü sıkıştırın (Uygula / Yenile).")
            return

        import tkinter as tk
        from PIL import ImageTk

        data = np.clip(self._recon_luma, 0, 255).astype(np.uint8)
        if data.ndim == 3 and data.shape[2] >= 3:
            rgb = cv2.cvtColor(data[:, :, :3], cv2.COLOR_BGR2RGB)
        elif data.ndim == 3:
            gray = data[:, :, 0]
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

        zoom_win = tk.Toplevel(self)
        zoom_win.title("Sıkıştırılmış Görüntü — Zoom İnceleme")
        zoom_win.geometry("980x760")

        toolbar = ctk.CTkFrame(zoom_win, fg_color="transparent")
        toolbar.pack(fill="x", padx=10, pady=(10, 6))

        ctk.CTkLabel(toolbar, text="Zoom:", font=FONT_LABEL).pack(side="left", padx=(0, 8))
        zoom_var = tk.StringVar(value="2x")

        canvas_wrap = tk.Frame(zoom_win, bg="#e9eef2")
        canvas_wrap.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        canvas = tk.Canvas(canvas_wrap, bg="#111318", highlightthickness=0)
        hbar = tk.Scrollbar(canvas_wrap, orient="horizontal", command=canvas.xview)
        vbar = tk.Scrollbar(canvas_wrap, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        canvas_wrap.grid_rowconfigure(0, weight=1)
        canvas_wrap.grid_columnconfigure(0, weight=1)
        zoom_photo_ref: List[Any] = [None]

        def _render_zoom(scale_text: str) -> None:
            scale = int(scale_text.replace("x", ""))
            h, w = rgb.shape[:2]
            pil = Image.fromarray(rgb)
            if scale > 1:
                pil = pil.resize((w * scale, h * scale), Image.Resampling.NEAREST)

            photo = ImageTk.PhotoImage(pil)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            zoom_photo_ref[0] = photo
            canvas.configure(scrollregion=(0, 0, pil.size[0], pil.size[1]))

        zoom_menu = ctk.CTkOptionMenu(
            toolbar,
            values=["1x", "2x", "4x", "8x"],
            variable=zoom_var,
            command=_render_zoom,
            width=90,
        )
        zoom_menu.pack(side="left")

        ctk.CTkLabel(
            toolbar,
            text="NEAREST ölçekleme aktif: blok bozunmaları yumuşatılmaz.",
            text_color=TEXT_SECONDARY,
            font=FONT_BODY,
        ).pack(side="left", padx=(12, 0))

        _render_zoom(zoom_var.get())

    def _on_load_image(self) -> None:
        """Dosya seç → _reload_luma → metrik ve histogram güncelle."""
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Görüntü seçin",
            filetypes=[
                ("Görüntü", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.jfif"),
                ("Tüm dosyalar", "*.*"),
            ],
        )
        if not path:
            return
        self._image_path = path
        self._set_status(f"Yükleniyor… {self._short_path(path)}")
        self.update_idletasks()
        try:
            self._reload_luma()
            self._detect_and_show_type()  # Otomatik tür algılama
            self._reset_metrics_display()  # Eski metrikleri temizle
        except Exception as e:
            self.metric_note.configure(text=f"Yükleme hatası: {e}")
            self._report_error("Yükleme hatası", e, code="ERR-LOAD-01")
            return

        self._set_status(f"Görüntü yüklendi: {self._short_path(path)}")
        # Görüntü yüklenince doğrudan _process çağır
        self._on_apply_clicked()

    def _on_clear_image(self) -> None:
        """Tüm görüntü ve işlemleri temizle — başlangıç durumuna dön."""
        # Görüntü verilerini sıfırla
        self._image_path = None
        self._original_bgr = None
        self._luma = None
        self._display_orig = None
        self._display_recon = None
        self._last_coeffs = None
        self.current_coeffs = None
        
        # UI'ı placeholder'a döndür
        self.lbl_orig.configure(text="Henüz görüntü yok\n\nGörüntü Yükle'ye tıklayın")
        self.lbl_recon.configure(text="Önce görüntü yükleyin\n\nSonra Uygula ile işleyin")
        self._update_resolution_captions()  # Çözünürlük bilgisini sıfırla
        
        # Grafikleri temizle (önceki histogram kalıntısı kalmasın)
        if hasattr(self, "fig"):
            self.fig.clf()
            figure_canvas_redraw(self.canvas)

        # Metrikleri sıfırla (DWT + DCT önizleme hücreleri dahil)
        self._reset_metrics_display()
        self._reset_dwt_cell_titles_short()
        for key in DCT_ZONE_KEYS:
            if key in self.dct_band_title_labels:
                self.dct_band_title_labels[key].configure(text=DCT_ZONE_VIZ[key])
        self._set_placeholders_empty()
        
        empty_info = {
            "method": "—",
            "level_delta": "—",
            "huffman": "—",
            "compression": "—"
        }
        if self.method_var.get() == "JPEG (DCT)":
            self._draw_dct_pipeline_compact(info_dict=empty_info)
        else:
            self.fig_schema.clear()
            self._draw_static_dwt_schema(info_dict=empty_info, level=int(self.level_var.get()))
            figure_canvas_redraw(self.canvas_schema)
        
        # Status
        self._set_status("Temizlendi — Yeni görüntü yüklemek için Görüntü Yükle'ye tıklayın.")

    def _bgr_to_display(self, bgr: np.ndarray) -> np.ndarray:
        """Önizleme için: BGR → RGB uint8 dizi (ekranda gösterim). Orijinal boyut korunur."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _set_image_label(self, label: ctk.CTkLabel, rgb: np.ndarray) -> None:
        """CTkImage oluştur; referansı label._image üzerinde tut (çöp toplanmasın)."""
        # Float ise uint8'e dönüştür
        if rgb.dtype == np.float32 or rgb.dtype == np.float64:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        pil = Image.fromarray(rgb)
        # DCT modunda sağ panelde yumuşatma/interpolasyon kullanma.
        # Tüm görüntüyü koru; sadece NEAREST ile kutuya sığdır.
        if label is getattr(self, "lbl_recon", None) and getattr(self, "_is_dct_mode", False):
            pil.thumbnail((360, 320), Image.Resampling.NEAREST)
        else:
            pil.thumbnail((360, 320), Image.Resampling.LANCZOS)  # Kutuya sığdır
        w, h = pil.size
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(w, h))
        label.configure(image=ctk_img, text="", text_color=("gray10", "gray90"))
        label._image = ctk_img  # Zorunlu referans (CustomTkinter dokümantasyonu)

    def _find_best_jpeg_quality(self, target_bpp: float, num_pixels: int) -> int:
        # JPEG kalite-bpp eğrisi düşük kalite bölgesinde çok doğrusal değildir.
        # Tam tarama ile en yakın bpp hedefini seçerek ani kalite sıçramalarını azalt.
        best_q = 50
        best_diff = float("inf")
        best_bpp = 0.0
        min_bpp = float("inf")

        orig_bgr = self._require_original_bgr()

        for q in range(1, 101):
            ok, enc = _cv2_imencode(".jpg", orig_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                continue

            bpp = (int(enc.nbytes) * 8) / max(1, num_pixels)
            if bpp < min_bpp:
                min_bpp = bpp
            diff = abs(bpp - target_bpp)

            # Eşit farkta daha yüksek kaliteyi tercih et (daha az bozulma).
            if diff < best_diff or (abs(diff - best_diff) < 1e-12 and q > best_q):
                best_diff = diff
                best_q = q
                best_bpp = bpp

        self._last_jpeg_target_bpp = float(target_bpp)
        self._last_jpeg_actual_bpp = float(best_bpp)
        self._last_jpeg_min_bpp = float(min_bpp if min_bpp != float("inf") else 0.0)

        if target_bpp < self._last_jpeg_min_bpp:
            print(
                f"[WARN] JPEG BPP floor reached: target={target_bpp:.3f}, "
                f"min_achievable={self._last_jpeg_min_bpp:.3f}"
            )
        print(f"[DEBUG] JPEG BPP mode: target={target_bpp:.3f}, selected_q={best_q}, actual_bpp={best_bpp:.3f}")
        return best_q

    def _find_best_dwt_q_step(
        self,
        target_bpp: float,
        num_pixels: int,
        wavelet: str,
        level: int,
        channel_multiplier: int = 1,
        mode: str = "scalar",
    ) -> float:
        """
        Hedef BPP'yi karşılayan kuantalama adımını (Δ) ikili arama ile bulur.

        Algoritma:
          target_bytes = num_pixels * target_bpp / 8   (tek kanal için)
          İkili arama: Δ ↑ → sıkıştırma ↑ (daha az byte)
          huffman_compressed_size() ile her adımda gerçek boyut ölçülür.
          Hedef boyutun ±5% içindeyse erken durur.
        """
        from huffman_codec import huffman_compressed_size as _huff_sz

        # Hedef toplam byte değerini, aranan kanal akışına göre normalize et.
        # Örn: renkli JPEG2000-DWT modunda arama tek kanal üzerinde yapılırken
        # nihai çıktı 3 kanal (B,G,R) olduğundan kanal başına hedefe düşürülür.
        target_bytes = (target_bpp * num_pixels) / 8.0
        channel_multiplier = max(1, int(channel_multiplier))
        target_bytes = target_bytes / channel_multiplier
        target_bytes = target_bytes / CompressionApp._dwt_level_aggressiveness(level)

        dwt = DWTTransformer(wavelet_name=wavelet, level=level)
        luma = self._require_luma()
        if luma.ndim == 3:
            luma_1ch = luma[:, :, 0]
        else:
            luma_1ch = luma
        coeffs = dwt.apply_dwt2d(luma_1ch.astype(np.float64))

        low, high = 0.5, 10000.0
        best_step = 10.0
        best_diff = float('inf')

        for _ in range(20):
            mid = (low + high) / 2.0
            q_c = quantize_coeffs(coeffs, mid, mode=mode)
            _, comp_bytes, _ = _huff_sz(q_c)

            diff = abs(comp_bytes - target_bytes)
            if diff < best_diff:
                best_diff = diff
                best_step = mid

            # ±5% tolerans içindeyse aramayı durdur
            if diff <= 0.05 * target_bytes:
                return mid

            if comp_bytes > target_bytes:
                low = mid   # Fazla byte → adımı büyüt (daha agresif kuantalama)
            else:
                high = mid  # Az byte → adımı küçült (daha az kuantalama)

        return best_step

    def _find_best_dwt_q_step_color_bpp(
        self,
        target_bpp: float,
        num_pixels: int,
        wavelet: str,
        level: int,
        mode: str = "scalar",
    ) -> float:
        """Renkli görsel için BPP hedefini TOPLAM Y+Cr+Cb Huffman
        baytına göre arar (CR modundaki mantıkla aynı disiplin,
        tek-kanal proxy YOK, channel_multiplier YOK)."""
        target_bytes = (target_bpp * num_pixels) / 8.0
        low, high = 0.5, 10000.0
        best_step = high
        best_diff = float("inf")
        for _ in range(20):
            mid = (low + high) / 2.0
            eval_result = self._evaluate_dwt_jpeg2000_step(
                mid, wavelet, level, return_details=False, mode=mode
            )
            comp_bytes = eval_result["compressed_bytes"]
            diff = abs(comp_bytes - target_bytes)
            if diff < best_diff:
                best_diff = diff
                best_step = mid
            if comp_bytes > target_bytes:
                low = mid
            else:
                high = mid
            if diff <= 0.05 * target_bytes:
                return mid
        return best_step

    # ------------------------------------------------------------------
    # Compression Ratio mode helpers
    # ------------------------------------------------------------------

    def _find_best_jpeg_quality_for_size(self, img: np.ndarray, target_bytes: int) -> Tuple[int, int]:
        """Hedef boyuta en yakın (aşmayan en yüksek kalite) JPEG quality döndür."""
        if target_bytes <= 0:
            return 85, 0
        best_q = 1
        best_actual = 0
        smallest_q = 1
        smallest_bytes = float("inf")

        for q in range(1, 101):
            ok, enc = _cv2_imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                continue
            actual_bytes = int(enc.nbytes)
            if actual_bytes < smallest_bytes:
                smallest_bytes = actual_bytes
                smallest_q = q
            if actual_bytes <= target_bytes:
                best_q = q
                best_actual = actual_bytes

        if best_actual == 0:
            return smallest_q, int(smallest_bytes)
        return best_q, best_actual

    def _encode_recon_jpeg(
        self, recon_luma: np.ndarray, target_bytes: Optional[int] = None
    ) -> Tuple[Optional[bytes], int]:
        """Yeniden oluşturulmuş görüntüyü hedef boyuta uygun JPEG olarak kodla."""
        recon_uint8 = np.clip(recon_luma, 0, 255).astype(np.uint8)
        if target_bytes is not None and target_bytes > 0:
            quality, _ = self._find_best_jpeg_quality_for_size(recon_uint8, int(target_bytes))
        else:
            quality = 85
        ok, enc = _cv2_imencode(".jpg", recon_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None, quality
        return enc.tobytes(), quality

    @staticmethod
    def _decode_jpeg_bytes(jpeg_bytes: bytes, color: bool) -> Optional[np.ndarray]:
        """JPEG byte akışını BGR veya gri float32 görüntüye çöz."""
        if not jpeg_bytes:
            return None
        flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
        dec = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), flag)
        if dec is None:
            return None
        return dec.astype(np.float32)

    def _evaluate_dwt_huffman_bytes(
        self,
        step: float,
        wavelet: str,
        level: int,
        coeffs: Any = None,
        comp_ratio: float = 5.0,
        return_details: bool = False,
        mode: str = "scalar",
    ) -> Tuple[int, Dict[str, Any]]:
        """DWT adımı için Huffman katsayı boyutunu ölç (CR araması bununla yapılır)."""
        dbg = self._evaluate_dwt_jpeg2000_step(
            step,
            wavelet,
            level,
            coeffs=coeffs,
            return_details=return_details,
            comp_ratio=comp_ratio,
            mode=mode,
        )
        return int(dbg["compressed_bytes"]), dbg

    def _evaluate_dwt_export_bytes(
        self,
        step: float,
        wavelet: str,
        level: int,
        target_bytes: int,
        coeffs: Any = None,
        comp_ratio: float = 5.0,
        mode: str = "scalar",
    ) -> Tuple[int, Dict[str, Any]]:
        """DWT adımı için gerçek çıktı dosya boyutunu (JPEG) ölç."""
        dbg = self._evaluate_dwt_jpeg2000_step(
            step,
            wavelet,
            level,
            coeffs=coeffs,
            return_details=True,
            comp_ratio=comp_ratio,
            mode=mode,
        )
        recon = dbg.get("recon_luma")
        if recon is None:
            export_bytes = int(dbg["compressed_bytes"])
        else:
            enc_bytes, _ = self._encode_recon_jpeg(recon, target_bytes=target_bytes)
            export_bytes = len(enc_bytes) if enc_bytes else int(dbg["compressed_bytes"])
        dbg["export_bytes"] = export_bytes
        return export_bytes, dbg

    def _get_export_file_bytes(self) -> int:
        """İndirilebilir çıktı dosyasının byte boyutu (JPEG/JP2/zlib)."""
        encoded = self._last_encoded_image_bytes
        return len(encoded) if encoded else 0

    def _require_luma(self) -> np.ndarray:
        """Yüklü luma dizisini döndür; yoksa RuntimeError."""
        if self._luma is None:
            raise RuntimeError("Görüntü yüklenmedi (_luma).")
        return self._luma

    def _get_resized_color_bgr(self) -> Optional[np.ndarray]:
        """self._original_bgr'yi, self._luma ile aynı hedef boyuta
        resize edilmiş BGR 3-kanal olarak döndürür. Renkli görsel
        yoksa None döner."""
        if not self._is_color_image or self._original_bgr is None:
            return None
        target_h, target_w = self._luma.shape[:2]
        return cv2.resize(
            self._original_bgr, (target_w, target_h),
            interpolation=cv2.INTER_AREA
        )

    def _require_original_bgr(self) -> np.ndarray:
        """Yüklü BGR dizisini döndür; yoksa RuntimeError."""
        if self._original_bgr is None:
            raise RuntimeError("Görüntü yüklenmedi (_original_bgr).")
        return self._original_bgr

    def _get_original_file_bytes(self) -> int:
        """Return the raw uncompressed size (or file size) of the loaded image."""
        override = getattr(self, "_snap_orig_bytes_override", None)
        if override is not None:
            return int(override)
        import os
        if getattr(self, '_image_path', None) and os.path.exists(str(self._image_path)):
            return os.path.getsize(str(self._image_path))
        if self._original_bgr is not None:
            return int(self._original_bgr.nbytes)
        return 0

    def _find_best_jpeg_quality_by_ratio(self, comp_ratio: float) -> int:
        """Return the highest JPEG quality whose encoded size is <= target size."""
        orig_bytes = self._get_original_file_bytes()
        if orig_bytes == 0 or comp_ratio <= 0 or self._original_bgr is None:
            return 50
        target_bytes = max(1, int(orig_bytes / comp_ratio))
        quality, best_actual = self._find_best_jpeg_quality_for_size(self._original_bgr, target_bytes)
        actual_ratio = orig_bytes / best_actual if best_actual > 0 else 0
        print(f"\n--- DEBUG (JPEG CR MODE) ---")
        print(f"original_size: {orig_bytes} bytes ({orig_bytes/1024:.2f} KB)")
        print(f"target_size:   {target_bytes} bytes ({target_bytes/1024:.2f} KB) for ratio {comp_ratio}:1")
        print(f"actual_size:   {best_actual} bytes ({best_actual/1024:.2f} KB)")
        print(f"actual_ratio:  {actual_ratio:.2f}:1")
        print(f"best quality:  {quality}")
        print(f"----------------------------\n")
        return quality

    @staticmethod
    def _dwt_delta_bounds_for_ratio(comp_ratio: float) -> Tuple[float, float]:
        """Bound DWT quantization to avoid visually destructive CR targets."""
        r = max(1.0, float(comp_ratio))
        if r <= 2.0:
            return 4.0, 24.0
        if r <= 5.0:
            t = (r - 2.0) / 3.0
            return 6.0 + 10.0 * t, 24.0 + 28.0 * t
        if r <= 10.0:
            t = (r - 5.0) / 5.0
            return 16.0 + 18.0 * t, 52.0 + 40.0 * t
        upper = min(180.0, 92.0 + (r - 10.0) * 2.4)
        lower = min(140.0, max(24.0, upper - 60.0))
        return lower, upper

    @staticmethod
    def _dwt_level_aggressiveness(level: int) -> float:
        """Return a monotonic level weight so deeper DWT levels quantize more aggressively."""
        level = max(1, int(level))
        return 1.0 + 0.12 * (level - 1)

    @staticmethod
    def _dwt_chroma_step_scale(comp_ratio: float) -> float:
        """Use modest chroma quantization, similar in spirit to image codecs."""
        return min(2.0, 1.5 + 0.1 * max(1.0, float(comp_ratio)))

    def _evaluate_dwt_jpeg2000_step(
        self,
        step: float,
        wavelet: str,
        level: int,
        coeffs: Any = None,
        return_details: bool = False,
        comp_ratio: float = 5.0,
        mode: str = "scalar",
    ) -> Dict[str, Any]:
        """Evaluate one JPEG2000-style DWT step using the same path as final rendering."""
        from huffman_codec import huffman_compressed_size

        luma = self._require_luma()
        dwt = DWTTransformer(wavelet_name=wavelet, level=level)
        is_color = self._is_color_image
        chroma_scale = CompressionApp._dwt_chroma_step_scale(comp_ratio)
        total_orig = 0
        total_huff = 0

        color_bgr = self._get_resized_color_bgr() if is_color else None
        if color_bgr is None:
            is_color = False

        if is_color:
            source_u8 = np.clip(color_bgr, 0, 255).astype(np.uint8)
            ycrcb = cv2.cvtColor(source_u8, cv2.COLOR_BGR2YCrCb).astype(np.float64)
            h, w = ycrcb.shape[:2]
            planes = [
                ("Y", ycrcb[:, :, 0], step, (w, h)),
                ("Cr", _cv2_resize(ycrcb[:, :, 1], (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA), step * chroma_scale, (w, h)),
                ("Cb", _cv2_resize(ycrcb[:, :, 2], (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA), step * chroma_scale, (w, h)),
            ]

            q_coeffs_dict: Dict[str, List] = {}
            recon_planes: List[np.ndarray] = []
            for name, plane, plane_step, full_size in planes:
                plane_coeffs = dwt.apply_dwt2d(plane)
                q_coeffs = quantize_coeffs(plane_coeffs, plane_step, mode=mode)
                q_coeffs_dict[name] = q_coeffs
                orig_ch, huff_ch, _ = huffman_compressed_size(q_coeffs)
                total_orig += orig_ch
                total_huff += huff_ch

                if return_details:
                    if is_pr_qmf(wavelet):
                        recon = waverec2_prqmf(q_coeffs, wavelet)
                    else:
                        recon = pywt.waverec2(q_coeffs, wavelet, mode="periodization")
                    recon = recon[: plane.shape[0], : plane.shape[1]]
                    if name != "Y":
                        recon = _cv2_resize(recon, full_size, interpolation=cv2.INTER_LINEAR)
                    recon_planes.append(recon)

            recon_luma = None
            psnr = 0.0
            if return_details:
                recon_ycrcb = np.stack(recon_planes, axis=2)
                recon_luma = cv2.cvtColor(np.clip(recon_ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR).astype(np.float32)
                psnr = psnr_fn(color_bgr.astype(np.float32), recon_luma)

            return {
                "compressed_bytes": int(total_huff),
                "original_coeff_bytes": int(total_orig),
                "psnr": psnr,
                "recon_luma": recon_luma,
                "coeffs_for_plot": q_coeffs_dict.get("Y"),
                "last_coeffs": q_coeffs_dict.get("Y"),
            }

        if coeffs is None:
            coeffs = dwt.apply_dwt2d(luma.astype(np.float64))
        q_coeffs = quantize_coeffs(coeffs, step, mode=mode)
        total_orig, total_huff, _ = huffman_compressed_size(q_coeffs)
        recon_luma = None
        psnr = 0.0
        if return_details:
            if is_pr_qmf(wavelet):
                recon_luma = waverec2_prqmf(q_coeffs, wavelet)
            else:
                recon_luma = pywt.waverec2(q_coeffs, wavelet, mode="periodization")
            recon_luma = recon_luma[: luma.shape[0], : luma.shape[1]]
            psnr = psnr_fn(luma.astype(np.float32), recon_luma.astype(np.float32))
        return {
            "compressed_bytes": int(total_huff),
            "original_coeff_bytes": int(total_orig),
            "psnr": psnr,
            "recon_luma": recon_luma,
            "coeffs_for_plot": q_coeffs,
            "last_coeffs": q_coeffs,
        }

    def _find_best_dwt_q_step_by_ratio(
        self, comp_ratio: float, wavelet: str, level: int, mode: str = "scalar"
    ) -> float:
        """
        Huffman katsayı boyutunu hedefleyerek DWT kuantalama adımını (Δ) bul.

        CR araması JPEG dosya boyutuyla değil Huffman ile yapılır; aksi halde DWT
        neredeyse kuantalama yapmadan yalnızca JPEG ile hedefe ulaşır ve görüntü bozulmaz.
        """
        orig_bytes = self._get_original_file_bytes()
        if orig_bytes == 0 or comp_ratio <= 0:
            return 10.0
        target_bytes = max(1, int(orig_bytes / comp_ratio))

        dwt = DWTTransformer(wavelet_name=wavelet, level=level)
        luma = self._require_luma()
        coeffs = None
        if not self._is_color_image:
            coeffs = dwt.apply_dwt2d(luma.astype(np.float64))

        low, high = CompressionApp._dwt_delta_bounds_for_ratio(comp_ratio)
        min_delta, max_delta = low, high
        best_step = high
        best_huffman, _ = self._evaluate_dwt_huffman_bytes(
            high, wavelet, level, coeffs=coeffs, comp_ratio=comp_ratio, mode=mode
        )
        target_met = best_huffman <= target_bytes

        for _ in range(50):
            mid = (low + high) / 2.0
            huffman_bytes, _ = self._evaluate_dwt_huffman_bytes(
                mid, wavelet, level, coeffs=coeffs, comp_ratio=comp_ratio, mode=mode
            )

            if huffman_bytes <= target_bytes:
                best_step = mid
                best_huffman = huffman_bytes
                target_met = True
                high = mid
            else:
                low = mid

        if not target_met:
            best_step = high
            best_huffman, dbg = self._evaluate_dwt_huffman_bytes(
                best_step, wavelet, level, coeffs=coeffs, comp_ratio=comp_ratio, return_details=True, mode=mode
            )
        else:
            best_huffman, dbg = self._evaluate_dwt_huffman_bytes(
                best_step, wavelet, level, coeffs=coeffs, comp_ratio=comp_ratio, return_details=True, mode=mode
            )

        best_export_bytes, _ = self._evaluate_dwt_export_bytes(
            best_step, wavelet, level, target_bytes, coeffs=coeffs, comp_ratio=comp_ratio, mode=mode
        )
        actual_ratio = orig_bytes / best_export_bytes if best_export_bytes > 0 else 0
        print(f"\n--- DEBUG (DWT CR MODE) ---")
        print(f"wavelet:       {wavelet}")
        print(f"ratio:         {comp_ratio:.2f}:1")
        print(f"delta_bounds:  {min_delta:.2f} .. {max_delta:.2f}")
        print(f"original_size: {orig_bytes} bytes ({orig_bytes/1024:.2f} KB)")
        print(f"target_size:   {target_bytes} bytes ({target_bytes/1024:.2f} KB) for ratio {comp_ratio}:1")
        print(f"huffman_size:  {best_huffman} bytes ({best_huffman/1024:.2f} KB)")
        print(f"export_size:   {best_export_bytes} bytes ({best_export_bytes/1024:.2f} KB)")
        print(f"actual_ratio:  {actual_ratio:.2f}:1")
        print(f"delta:         {best_step:.2f}")
        print(f"psnr:          {dbg.get('psnr', 0.0):.2f} dB")
        print(f"huffman_met:   {best_huffman <= target_bytes}")
        print(f"---------------------------\n")
        return best_step

    def _process(self) -> None:
        """Geriye dönük: async işlem başlat."""
        self._start_process_async()

    def _process_compute_body(self, snap: Dict[str, Any]) -> Dict[str, Any]:
        """
        Worker thread: sıkıştırma hesabı (UI yok).
        snap + _use_process_snap ile self._luma / _original_bgr bağlı olmalı.
        """
        method = snap["method"]
        coeff_val = snap["coeff_val"]
        cr_mode = snap["cr_mode"]
        num_pixels = snap["num_pixels"]
        target_ratio = snap["target_ratio"]
        target_size_bytes = snap["target_size_bytes"]
        target_bpp = snap["target_bpp"]
        hardness = snap["hardness"]
        orig_bytes = snap["orig_file_bytes"]
        level = snap["level"]
        wavelet = snap["wavelet"]
        luma = self._require_luma()
        orig_bgr = self._require_original_bgr()
        is_color_luma = self._is_color_image

        result: Dict[str, Any] = {
            "method": method,
            "cr_mode": cr_mode,
            "level": level,
            "wavelet": wavelet,
            "target_ratio": target_ratio,
            "target_size_bytes": target_size_bytes,
            "target_bpp": target_bpp,
            "is_dct_mode": False,
            "q_step": 0.0,
            "note": "",
            "coeffs_for_plot": None,
            "last_coeffs": None,
            "current_coeffs": None,
            "compressed_bytes": 0,
            "last_encoded_image_bytes": None,
            "last_encoded_image_ext": None,
            "lossless_clear_bands": False,
            "lossless_metric_huffman": None,
        }

        recon_luma: Optional[np.ndarray] = None
        coeffs_for_plot: Optional[List] = None
        compressed_bytes: int = 0
        note = ""
        q_step = 0.0

        if method == "Lossless (Kayıpsız - Tıbbi)":
            if not _PIPELINE_AVAILABLE or LosslessCompressionPipeline is None:
                return {
                    **result,
                    "error": "Lossless mod için stages/core modülleri gerekli.",
                    "lossless_import": True,
                }

            zlib_level = int(max(1, min(9, 1 + (hardness / 100.0) * 8)))
            lossless = LosslessCompressionPipeline(compression_level=zlib_level)
            compressed, _meta = lossless.compress_image_data(luma.astype(np.uint8))
            recon_luma = luma.copy()
            compressed_bytes = len(compressed)
            original_bytes = luma.astype(np.uint8).nbytes
            lossless_ratio = 100.0 * (1.0 - (compressed_bytes / original_bytes))
            note = (
                f"Lossless (zlib L{zlib_level}) — {compressed_bytes/1024:.1f} KB / "
                f"{original_bytes/1024:.1f} KB (↓%{lossless_ratio:.0f}). PSNR=∞, MSE=0 (100% kayıpsız)"
            )
            dwt = DWTTransformer(wavelet_name=wavelet, level=level)
            luma_2d_dwt = _luma_to_2d_for_dwt(luma)
            coeffs_for_plot = dwt.apply_dwt2d(luma_2d_dwt)
            result.update({
                "last_encoded_image_bytes": compressed,
                "last_encoded_image_ext": ".bin",
                "lossless_clear_bands": True,
                "lossless_metric_huffman": f"zlib L{zlib_level}",
                "current_coeffs": coeffs_for_plot,
                "last_coeffs": None,
            })

        elif method == "JPEG (DCT)":
            if cr_mode:
                jpeg_quality = self._find_best_jpeg_quality_by_ratio(target_ratio)
            else:
                jpeg_quality = self._find_best_jpeg_quality(target_bpp, num_pixels)
            ok, enc = _cv2_imencode(
                ".jpg",
                orig_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            if not ok:
                return {**result, "error": "JPEG kodlama başarısız."}
            compressed_bytes = int(enc.nbytes)
            decode_flag = cv2.IMREAD_COLOR if self._is_color_image else cv2.IMREAD_GRAYSCALE
            dec = cv2.imdecode(enc, decode_flag)
            if dec is None:
                return {**result, "error": "JPEG decode başarısız."}
            if self._is_color_image:
                recon_luma = dec.astype(np.float32)
            elif dec.ndim == 2:
                recon_luma = dec.astype(np.float32)
            else:
                recon_luma = self._bgr_to_luma(dec)
            note = (
                f"JPEG kalite={jpeg_quality}. Önizleme: 8×8 DCT katsayı analizi (Y/luma kanalı)."
            )
            result.update({
                "is_dct_mode": True,
                "jpeg_quality": jpeg_quality,
                "last_encoded_image_bytes": enc.tobytes(),
                "last_encoded_image_ext": ".jpg",
                "coeffs_for_plot": None,
            })

        elif method == "JPEG2000 (DWT)":
            export_target_bytes = int(target_size_bytes) if cr_mode else int((target_bpp * num_pixels) / 8)
            manual_enabled = snap.get("manual_step_enabled", False)
            manual_value = snap.get("manual_step_value")
            quant_mode = snap.get("quant_mode", "scalar")

            if manual_enabled and manual_value is not None and manual_value > 0:
                q_step = float(manual_value)
            else:
                if cr_mode:
                    q_step = self._find_best_dwt_q_step_by_ratio(
                        target_ratio, wavelet, level, mode=quant_mode
                    )
                else:
                    if is_color_luma:
                        q_step = self._find_best_dwt_q_step_color_bpp(
                            target_bpp, num_pixels, wavelet, level, mode=quant_mode,
                        )
                    else:
                        q_step = self._find_best_dwt_q_step(
                            target_bpp, num_pixels, wavelet, level,
                            channel_multiplier=1, mode=quant_mode,
                        )
            eval_result = self._evaluate_dwt_jpeg2000_step(
                q_step,
                wavelet,
                level,
                return_details=True,
                comp_ratio=target_ratio if cr_mode else max(1.0, orig_bytes / max(1, export_target_bytes)),
                mode=quant_mode,
            )
            recon_luma = eval_result["recon_luma"]
            if recon_luma is None:
                return {**result, "error": "DWT yeniden oluşturma başarısız."}
            coeffs_for_plot = eval_result["coeffs_for_plot"]
            compressed_bytes = eval_result["compressed_bytes"]
            total_orig = eval_result["original_coeff_bytes"]
            saving = 100.0 * (1.0 - compressed_bytes / max(total_orig, 1))
            is_color = is_color_luma
            huffman_info = (
                f"Huffman(YCrCb 4:2:0): {compressed_bytes/1024:.1f} KB / Ham: {total_orig/1024:.1f} KB (↓%{saving:.0f})"
                if is_color
                else f"Huffman(per-band): {compressed_bytes/1024:.1f} KB / Ham: {total_orig/1024:.1f} KB (↓%{saving:.0f})"
            )
            note = f"DWT ({wavelet}), seviye {level}, Δ≈{q_step:.2f}. {huffman_info}"
            enc_bytes, jpeg_q = self._encode_recon_jpeg(recon_luma, target_bytes=export_target_bytes)
            enc_update: Dict[str, Any] = {}
            if enc_bytes:
                enc_update = {
                    "last_encoded_image_bytes": enc_bytes,
                    "last_encoded_image_ext": ".jpg",
                }
                note += f" | Çıktı JPEG Q{jpeg_q}: {len(enc_bytes)/1024:.1f} KB"
                # Önizleme ve indirilen dosya aynı olsun: kaydedilen JPEG'i decode et
                decoded = self._decode_jpeg_bytes(enc_bytes, color=is_color_luma)
                if decoded is not None:
                    recon_luma = decoded
            else:
                print("[WARN] DWT çıktı JPEG kodlama başarısız.")
            result.update({
                "last_coeffs": eval_result["last_coeffs"],
                "original_coeff_bytes": total_orig,
                "export_jpeg_bytes": len(enc_bytes) if enc_bytes else 0,
                **enc_update,
            })

        else:
            jpeg_quality = self._find_best_jpeg_quality(target_bpp, num_pixels)
            quant_mode = snap.get("quant_mode", "scalar")
            q_step = self._find_best_dwt_q_step(
                target_bpp, num_pixels, wavelet, level, mode=quant_mode
            )
            dwt = DWTTransformer(wavelet_name=wavelet, level=level)
            coeffs_raw = dwt.apply_dwt2d(luma.astype(np.float64))
            if isinstance(coeffs_raw, dict):
                coeffs = coeffs_raw["B"]  # type: ignore[index]
            else:
                coeffs = coeffs_raw
            q_coeffs = quantize_coeffs(coeffs, q_step, mode=quant_mode)
            coeffs_for_plot = coeffs
            try:
                from huffman_codec import huffman_compressed_size
                _raw_bytes, huffman_bytes, huff_saving = huffman_compressed_size(q_coeffs)
                compressed_bytes = huffman_bytes
            except Exception as e:
                huffman_bytes = estimate_wavelet_compressed_bytes(q_coeffs)
                compressed_bytes = huffman_bytes
                huff_saving = 0.0
                print(f"[WARN] Huffman fallback (compare): {e}")
            if is_pr_qmf(wavelet):
                recon_luma = waverec2_prqmf(q_coeffs, wavelet)
            else:
                recon_luma = pywt.waverec2(q_coeffs, wavelet, mode="periodization")
            recon_luma = recon_luma[: luma.shape[0], : luma.shape[1]]
            try:
                recon_uint8 = np.clip(recon_luma, 0, 255).astype(np.uint8)
                ok, enc = _cv2_imencode(".jp2", recon_uint8)
                if ok and len(enc) > 0:
                    result["last_encoded_image_bytes"] = enc.tobytes()
                    result["last_encoded_image_ext"] = ".jp2"
                else:
                    ok, enc = _cv2_imencode(".jpg", recon_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    if ok:
                        result["last_encoded_image_bytes"] = enc.tobytes()
                        result["last_encoded_image_ext"] = ".jpg"
            except Exception as e:
                print(f"[WARN] JPEG2000 kodlama başarısız (compare mode): {e}")
            ok, enc = _cv2_imencode(
                ".jpg",
                orig_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            jpeg_bytes = int(enc.nbytes) if ok else compressed_bytes
            note = (
                f"JPEG ({jpeg_quality} kalite) → {jpeg_bytes/1024:.1f} KB  |  "
                f"DWT+Huffman → {huffman_bytes/1024:.1f} KB (↓%{huff_saving:.0f})  |  "
                f"PSNR: {psnr_fn(luma, recon_luma):.1f} dB"
            )

        assert recon_luma is not None
        recon_luma = np.clip(recon_luma, 0, 255).astype(np.float32)
        result.update({
            "recon_luma": recon_luma,
            "coeffs_for_plot": coeffs_for_plot,
            "compressed_bytes": compressed_bytes,
            "note": note,
            "q_step": q_step,
        })
        return result

    def _process_apply_ui(self, result: Dict[str, Any], snap: Dict[str, Any]) -> None:
        """Ana thread: hesap sonuçlarını arayüze yansıt."""
        method = result["method"]
        cr_mode = result["cr_mode"]
        level = result["level"]
        wavelet = result["wavelet"]
        target_ratio = result["target_ratio"]
        target_size_bytes = result["target_size_bytes"]
        target_bpp = result["target_bpp"]
        num_pixels = snap["num_pixels"]
        q_step = result["q_step"]
        recon_luma = result["recon_luma"]
        coeffs_for_plot = result["coeffs_for_plot"]
        compressed_bytes = result["compressed_bytes"]
        luma = snap["luma"]

        self._last_cr_mode = cr_mode
        self._is_dct_mode = bool(result.get("is_dct_mode"))
        self._last_jpeg_quality = result.get("jpeg_quality", 85)
        self._last_encoded_image_bytes = result.get("last_encoded_image_bytes")
        self._last_encoded_image_ext = result.get("last_encoded_image_ext")
        self._recon_luma = recon_luma
        if result.get("last_coeffs") is not None:
            self._last_coeffs = result["last_coeffs"]
        if result.get("current_coeffs") is not None:
            self.current_coeffs = result["current_coeffs"]

        if result.get("lossless_clear_bands") and hasattr(self, "band_labels"):
            for lbl in self.band_labels.values():
                lbl.configure(image="", text="—")
        if result.get("lossless_metric_huffman"):
            self.metric_val_huffman.configure(text=result["lossless_metric_huffman"])

        if isinstance(recon_luma, np.ndarray) and len(recon_luma.shape) == 3:
            recon_bgr_display = cv2.cvtColor(recon_luma.astype(np.uint8), cv2.COLOR_BGR2RGB)
        else:
            recon_bgr_display = self._luma_to_bgr_preview(recon_luma)
        self._display_recon = recon_bgr_display
        self._set_image_label(self.lbl_recon, self._display_recon)

        if isinstance(recon_luma, np.ndarray) and recon_luma.ndim == 3:
            mse_ref = self._get_resized_color_bgr()
            if mse_ref is None:
                mse_ref = luma
        else:
            mse_ref = luma
        mse = float(np.mean((mse_ref.astype(np.float64) - recon_luma.astype(np.float64)) ** 2))
        psnr = 10 * np.log10(255**2 / mse) if mse > 1e-12 else 99.99
        ssim_val = 0.0
        try:
            ref_for_ssim = mse_ref
            recon_for_ssim = recon_luma
            # Kanal sayısı uyumsuzsa 3 kanala indir (BGR/RGB varsay)
            if isinstance(ref_for_ssim, np.ndarray) and ref_for_ssim.ndim == 3 and ref_for_ssim.shape[2] > 3:
                ref_for_ssim = ref_for_ssim[:, :, :3]
            if isinstance(recon_for_ssim, np.ndarray) and recon_for_ssim.ndim == 3 and recon_for_ssim.shape[2] > 3:
                recon_for_ssim = recon_for_ssim[:, :, :3]
            # H,W uyumsuzsa recon'u referansın boyutuna resize et
            if ref_for_ssim.shape[:2] != recon_for_ssim.shape[:2]:
                recon_for_ssim = cv2.resize(
                    recon_for_ssim, (ref_for_ssim.shape[1], ref_for_ssim.shape[0]),
                    interpolation=cv2.INTER_AREA
                )
            if ref_for_ssim.ndim == 3:
                from skimage.metrics import structural_similarity as _ssim
                ssim_val = float(_ssim(
                    ref_for_ssim.astype(np.float64), recon_for_ssim.astype(np.float64),
                    data_range=255.0, channel_axis=-1
                ))
            else:
                ssim_val = ssim_fn(
                    ref_for_ssim.astype(np.float64), recon_for_ssim.astype(np.float64),
                    data_range=255.0
                )
        except Exception as e:
            print(f"[SSIM] Hesaplanamadı: {e}")  # sessiz yutma yerine görünür log
            ssim_val = 0.0

        coeff_bytes = compressed_bytes
        export_jpeg_bytes = len(self._last_encoded_image_bytes) if self._last_encoded_image_bytes else 0
        ref_int16_bytes = int(result.get("original_coeff_bytes") or 0)

        # DWT: ana sıkıştırma metrikleri Huffman tahminine dayanır; JPEG export ayrı gösterilir
        if method == "JPEG2000 (DWT)":
            size_for_ratio = coeff_bytes
            preview_jpeg_bytes = export_jpeg_bytes or int(result.get("export_jpeg_bytes") or 0)
        else:
            size_for_ratio = export_jpeg_bytes if export_jpeg_bytes > 0 else coeff_bytes
            preview_jpeg_bytes = export_jpeg_bytes

        actual_orig_bytes = snap["orig_file_bytes"]
        if actual_orig_bytes <= 0:
            actual_orig_bytes = int(luma.nbytes)
        actual_ratio = actual_orig_bytes / max(1, size_for_ratio) if size_for_ratio > 0 else 0.0
        actual_bpp = (size_for_ratio * 8) / max(1, num_pixels)

        self.metric_val_psnr.configure(text=f"{psnr:.2f}")
        self.metric_val_mse.configure(text=f"{mse:.4f}")
        self._last_psnr = f"{psnr:.2f}"
        self._last_mse = f"{mse:.4f}"
        self._last_ssim = f"{ssim_val:.3f}"
        cr_pct = (
            100.0 * (1.0 - size_for_ratio / actual_orig_bytes)
            if actual_orig_bytes > 0
            else (100.0 * (1.0 - size_for_ratio / num_pixels) if num_pixels > 0 else 0.0)
        )
        self.metric_val_cr.configure(text=f"{cr_pct:.1f}")
        self._last_cr = f"{cr_pct:.1f}"
        self._last_actual_ratio = actual_ratio
        self._last_compressed_bytes = size_for_ratio
        self._last_coeff_bytes = coeff_bytes
        self._last_preview_jpeg_bytes = preview_jpeg_bytes
        self._last_ref_int16_bytes = ref_int16_bytes
        self._last_actual_bpp = actual_bpp
        self._last_orig_bytes = actual_orig_bytes
        if cr_mode:
            self._last_target_size_bytes = target_size_bytes
            self._last_target_ratio = target_ratio
        else:
            self._last_target_size_bytes = 0
            self._last_target_ratio = 0.0

        if method == "JPEG2000 (DWT)":
            preview_note = (
                f" | Preview JPEG Export: {preview_jpeg_bytes/1024:.1f} KB"
                if preview_jpeg_bytes > 0
                else ""
            )
            ref_note = (
                f" | Ham int16 ref: {ref_int16_bytes/1024:.1f} KB"
                if ref_int16_bytes > 0
                else ""
            )
            if cr_mode:
                mode_label = (
                    f"Mod: CR {target_ratio:.1f}:1 → Hedef {target_size_bytes/1024:.1f} KB | "
                    f"Huffman {size_for_ratio/1024:.1f} KB ({actual_ratio:.1f}:1, BPP≈{actual_bpp:.2f})"
                    f"{preview_note}{ref_note}"
                )
            else:
                mode_label = (
                    f"Mod: BPP hedef {target_bpp:.2f} | "
                    f"Huffman {size_for_ratio/1024:.1f} KB (BPP≈{actual_bpp:.2f}, {actual_ratio:.1f}:1)"
                    f"{preview_note}{ref_note}"
                )
            # Hedef ile gerçek sonuç arasındaki farkı kontrol et, büyükse uyar
            if cr_mode:
                target_val, actual_val, label = target_ratio, actual_ratio, "CR"
            else:
                target_val, actual_val, label = target_bpp, actual_bpp, "BPP"
            if target_val > 0:
                rel_diff = abs(actual_val - target_val) / target_val
                if rel_diff > 0.15:  # %15'ten fazla sapma varsa uyar
                    mode_label += f" | ⚠ Hedefe tam ulaşılamadı ({label} hedef {target_val:.2f}, gerçek {actual_val:.2f})"
        elif cr_mode:
            mode_label = (
                f"Mod: CR {target_ratio:.1f}:1 → Hedef {target_size_bytes/1024:.1f} KB | "
                f"Çıktı {preview_jpeg_bytes/1024:.1f} KB ({actual_ratio:.1f}:1, BPP≈{actual_bpp:.2f})"
            )
        else:
            mode_label = (
                f"Mod: BPP hedef {target_bpp:.2f} | "
                f"Çıktı {preview_jpeg_bytes/1024:.1f} KB (BPP≈{actual_bpp:.2f}, {actual_ratio:.1f}:1)"
            )
            if method == "JPEG (DCT)":
                jpeg_floor = float(getattr(self, "_last_jpeg_min_bpp", 0.0) or 0.0)
                if jpeg_floor > 0 and target_bpp < jpeg_floor:
                    mode_label += f" | JPEG alt sınır ≈ {jpeg_floor:.2f} bpp"

        export_ext = self._last_encoded_image_ext or ""
        if method == "JPEG (DCT)":
            jpeg_q = int(result.get("jpeg_quality", 85))
            info_dict = {
                "method": self._history_method_label(method),
                "level_delta": f"Q={jpeg_q}, 8×8 blok DCT",
                "huffman": f"{preview_jpeg_bytes/1024:.1f} KB (JPEG bitstream)",
                "export_file": f"{preview_jpeg_bytes/1024:.1f} KB{(' ' + export_ext) if export_ext else ''}",
                "compression": f"Oran: {actual_ratio:.1f}:1  ↓%{cr_pct:.0f}  |  BPP≈{actual_bpp:.2f}",
            }
            self._draw_dct_pipeline_compact(info_dict=info_dict)
        else:
            if method == "JPEG2000 (DWT)":
                preview_line = (
                    f"Preview JPEG Export: {preview_jpeg_bytes/1024:.1f} KB{(' ' + export_ext) if export_ext else ''}"
                    if preview_jpeg_bytes > 0
                    else "Preview JPEG Export: —"
                )
                info_dict = {
                    "method": self._history_method_label(method),
                    "level_delta": f"{wavelet}, L{level}, Δ≈{q_step:.2f}",
                    "huffman": f"{coeff_bytes/1024:.1f} KB (Huffman estimate)",
                    "export_file": preview_line,
                    "compression": f"Oran: {actual_ratio:.1f}:1 (Huffman)  ↓%{cr_pct:.0f}  |  BPP≈{actual_bpp:.2f}",
                    "ssim": f"{ssim_val:.3f}",
                }
            else:
                info_dict = {
                    "method": self._history_method_label(method),
                    "level_delta": f"{wavelet}, L{level}, Δ≈{q_step:.2f}",
                    "huffman": f"{coeff_bytes/1024:.1f} KB",
                    "export_file": f"{preview_jpeg_bytes/1024:.1f} KB{(' ' + export_ext) if export_ext else ''}",
                    "compression": f"Oran: {actual_ratio:.1f}:1  ↓%{cr_pct:.0f}  |  BPP≈{actual_bpp:.2f}",
                    "ssim": f"{ssim_val:.3f}",
                }

        if method == "JPEG2000 (DWT)":
            if preview_jpeg_bytes > 0:
                self.metric_val_huffman.configure(
                    text=f"{coeff_bytes/1024:.1f} (Preview JPEG {preview_jpeg_bytes/1024:.1f})"
                )
                history_huffman = (
                    f"Huffman: {coeff_bytes/1024:.1f} KB | Preview JPEG Export: {preview_jpeg_bytes/1024:.1f} KB"
                )
            else:
                self.metric_val_huffman.configure(text=f"{coeff_bytes/1024:.1f}")
                history_huffman = f"Huffman: {coeff_bytes/1024:.1f} KB"
            self._last_huff = history_huffman
        elif method != "JPEG (DCT)":
            if preview_jpeg_bytes != coeff_bytes:
                self.metric_val_huffman.configure(
                    text=f"{preview_jpeg_bytes/1024:.1f} (Huffman {coeff_bytes/1024:.1f})"
                )
                history_huffman = f"Çıktı: {preview_jpeg_bytes/1024:.1f} KB | Huffman: {coeff_bytes/1024:.1f} KB"
                self._last_huff = history_huffman
            else:
                self.metric_val_huffman.configure(text=f"{preview_jpeg_bytes/1024:.1f}")
                history_huffman = f"{preview_jpeg_bytes/1024:.1f} KB"
                self._last_huff = history_huffman
        else:
            self.metric_val_huffman.configure(text=f"{preview_jpeg_bytes/1024:.1f} (JPEG)")
            history_huffman = f"{preview_jpeg_bytes/1024:.1f} KB (JPEG bitstream)"
            self._last_huff = history_huffman

        self.metric_quality.configure(text=self._quality_text_from_psnr(psnr))
        self.metric_note.configure(text=mode_label)
        import os
        filename = os.path.basename(self._image_path) if self._image_path else "—"
        if cr_mode:
            history_target_text = (
                f"Target Ratio: {target_ratio:.1f}:1 | Target Size: {target_size_bytes/1024:.1f} KB | "
                f"Actual Ratio: {actual_ratio:.1f}:1"
            )
        else:
            history_target_text = f"Target BPP: {target_bpp:.2f}"
        self._add_history_entry(
            method=method,
            level=level,
            history_target_text=history_target_text,
            psnr=psnr,
            mse=mse,
            ssim=ssim_val,
            huffman_text=history_huffman,
            filename=filename,
        )

        y2 = self._luma_y_2d()
        if method == "JPEG (DCT)":
            self.render_dct_analysis_panel(
                y2,
                level,
                jpeg_quality=int(result.get("jpeg_quality", 85)),
                psnr=psnr,
                mse=mse,
                ssim=ssim_val,
                cr_pct=cr_pct,
                file_kb=preview_jpeg_bytes / 1024.0,
                ratio=actual_ratio,
                for_compare=False,
            )
        elif method == "JPEG vs JPEG2000 (karşılaştır)" and coeffs_for_plot is not None:
            self.render_dct_analysis_panel(
                y2,
                level,
                jpeg_quality=int(result.get("jpeg_quality", 85)),
                psnr=psnr,
                mse=mse,
                ssim=ssim_val,
                cr_pct=cr_pct,
                file_kb=preview_jpeg_bytes / 1024.0,
                ratio=actual_ratio,
                for_compare=True,
            )
            self._preserve_scroll_position(
                self.render_dwt_subband_panel,
                coeffs_for_plot, y2, level, info_dict=info_dict, wavelet=wavelet, for_compare=True
            )
        elif coeffs_for_plot is not None:
            if method == "JPEG2000 (DWT)":
                self._preserve_scroll_position(
                    self.render_dwt_subband_panel,
                    coeffs_for_plot, y2, level, info_dict=info_dict, wavelet=wavelet, for_compare=False
                )
            elif method == "Lossless (Kayıpsız - Tıbbi)":
                coeff_list = _ensure_dwt_coeff_list(coeffs_for_plot, y2, wavelet=wavelet, level=level)
                viz_level = len(coeff_list) - 1
                self.fig_schema.clear()
                self._draw_static_dwt_schema(
                    info_dict=info_dict,
                    coeffs=coeff_list,
                    level=viz_level,
                    luma_2d=y2,
                    wavelet=wavelet,
                )
                figure_canvas_redraw(self.canvas_schema)
                self._draw_histograms(coeff_list, viz_level)
                self._set_dwt_preview_layout("cells")
                from PIL import Image as PilImage

                orig = luma.astype(np.float32)
                recon = recon_luma.astype(np.float32)
                visuals = {
                    "LL": orig,
                    "HL": recon,
                    "LH": np.abs(orig - recon) * 10,
                    "HH": np.abs(orig - recon),
                }
                subtitles = {
                    "LL": "Orijinal",
                    "HL": "Lossless Çıktı",
                    "LH": "Fark x10",
                    "HH": "Ham Fark",
                }
                for name, arr in visuals.items():
                    mn, mx = arr.min(), arr.max()
                    if mx - mn > 1e-6:
                        arr_norm = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
                    else:
                        arr_norm = np.zeros_like(arr, dtype=np.uint8)
                    ts = BAND_THUMB_SIZE
                    img = PilImage.fromarray(arr_norm, mode="L").resize(
                        (ts, ts), Image.Resampling.BILINEAR)
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(ts, ts))
                    self.band_labels[name].configure(image=ctk_img, text="")
                    self._image_refs[f"lossless_band_{name}"] = ctk_img
                for name, subtitle in subtitles.items():
                    if name in self.dwt_band_title_labels:
                        self.dwt_band_title_labels[name].configure(text=f"{name}\n{subtitle}")
            else:
                coeff_list = _ensure_dwt_coeff_list(coeffs_for_plot, y2, wavelet=wavelet, level=level)
                viz_level = len(coeff_list) - 1
                self.fig_schema.clear()
                self._draw_static_dwt_schema(
                    info_dict=info_dict,
                    coeffs=coeff_list,
                    level=viz_level,
                    luma_2d=y2,
                    wavelet=wavelet,
                )
                figure_canvas_redraw(self.canvas_schema)
                self._set_dwt_preview_layout("mosaic")
                self._draw_histograms(coeff_list, viz_level)
                self._draw_band_images(coeff_list, viz_level)
        else:
            self.fig.clf()
            figure_canvas_redraw(self.canvas)

        self._set_status(
            f"Hazır — {self._short_path(self._image_path or '')} | PSNR {psnr:.1f} dB"
        )


    def _run_rd_analysis(self) -> None:
        """Rate-Distortion analizi — kalite vs PSNR grafiği (FR-16)"""
        # Kullanıcı manuel basmalı, otomatik tetiklenmesin
        if not getattr(self, '_manual_trigger', False):
            return
        self._manual_trigger = False
        if self._luma is None:
            self._set_status("Önce görüntü yükleyin.")
            return
        luma_snap = self._luma
        import threading, matplotlib.pyplot as plt, numpy as np
        from core.rate_distortion import RateDistortionAnalyzer

        def _worker():
            try:
                self.after(0, lambda: self._set_status("R-D analizi hesaplanıyor..."))
                
                # Arayüzde seçili olan metodu al
                current_method = self.method_var.get()
                if "DCT" in current_method:
                    method_str = "DCT"
                elif "Lossless" in current_method:
                    method_str = "Lossless"
                else:
                    method_str = "DWT"

                analyzer = RateDistortionAnalyzer(method=method_str)
                
                if method_str == "Lossless":
                    # zlib seviyeleri 1-9
                    analyzer.qualities = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                elif method_str == "DCT":
                    analyzer.qualities = [5, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 99]
                else:
                    # Daha detaylı kalite seviyeleri
                    analyzer.qualities = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99]
                    
                # _luma'yı hazırla - 2D olduğundan emin ol
                image_data = luma_snap.astype(np.float64)
                if len(image_data.shape) == 3:
                    # 3 kanallı ise ilk kanalı al
                    image_data = image_data[:, :, 0]
                
                # R-D analizi yap
                rd_points = analyzer.analyze(image_data)
                
                if not rd_points:
                    self.after(0, lambda: self._set_status("✗ R-D analizi veri üretemedi"))
                    return
                
                bitrates = [p.bitrate_bpp for p in rd_points]
                psnrs = [p.psnr_db if p.psnr_db != float('inf') else 100.0 for p in rd_points]
                qualities = [p.quality for p in rd_points]
                
                def _plot():
                    try:
                        # Önceki grafikleri temizle
                        plt.close('all')
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if not bitrates or not psnrs:
                            ax.text(0.5, 0.5, "Veri yok", ha='center', va='center')
                            plt.tight_layout()
                            self.after(0, lambda: plt.show())
                            return
                        
                        # R-D eğrisini çiz
                        label_str = f"{method_str} R-D Eğrisi" if method_str != "Lossless" else "Lossless R-D (Düz Çizgi)"
                        ax.plot(bitrates, psnrs, "bo-", linewidth=2.5, markersize=7, label=label_str)
                        
                        # Eğri altında doldur (estetik)
                        ax.fill_between(bitrates, psnrs, alpha=0.2, color='blue')
                        
                        # Kalite değerlerini noktaların üzerine yaz
                        for i, (br, psnr, q) in enumerate(zip(bitrates, psnrs, qualities)):
                            q_label = f'L{q}' if method_str == "Lossless" else f'Q{q}'
                            ax.annotate(q_label, (br, psnr), textcoords="offset points", 
                                       xytext=(0, 8), ha='center', fontsize=8, color='darkblue')
                        
                        ax.set_xlabel("Bitrate (bpp)", fontsize=11, fontweight='bold')
                        ax.set_ylabel("PSNR (dB)", fontsize=11, fontweight='bold')
                        ax.set_title(f"Rate-Distortion Analizi ({method_str}) — {len(rd_points)} seviye", 
                                   fontsize=12, fontweight='bold')
                        ax.grid(True, alpha=0.3, linestyle='--')
                        ax.legend(loc='lower right', fontsize=10)
                        
                        # İstatistikler
                        max_psnr = max(psnrs) if psnrs else 0
                        min_bitrate = min(bitrates) if bitrates else 0
                        max_bitrate = max(bitrates) if bitrates else 0
                        ax.text(0.02, 0.98, 
                               f"Maks PSNR: {max_psnr:.2f} dB\nMin Bitrate: {min_bitrate:.4f} bpp\nMaks Bitrate: {max_bitrate:.4f} bpp\nKalite Sayısı: {len(rd_points)}", 
                               transform=ax.transAxes, fontsize=9, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                        
                        plt.tight_layout()
                    except Exception as plot_err:
                        print(f"Grafik çizim hatası: {plot_err}")
                        traceback.print_exc()
                
                self.after(0, _plot)
                self.after(0, lambda: plt.show())
                self.after(0, lambda: self._set_status(f"✓ R-D analizi tamamlandı ({len(rd_points)} nokta)"))
            except Exception as e:
                self._report_error("R-D hata", e, code="ERR-RD-01", threaded=True)

        threading.Thread(target=_worker, daemon=True).start()

    def _run_image_type_test(self) -> None:
        """Görüntü türü karşılaştırması — Yüklenen görüntüyü test et (FR-35)"""
        # Kullanıcı manuel basmalı, otomatik tetiklenmesin
        if not getattr(self, '_manual_trigger', False):
            return
        self._manual_trigger = False
        
        # Yüklenen görüntü var mı kontrol et
        if not hasattr(self, '_luma') or self._luma is None:
            import tkinter.messagebox as mb
            mb.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
        
        luma_snap = self._luma
        import threading
        from core.image_type_comparison import ImageTypeComparisonRunner
        from core.image_classifier import ImageClassifier

        def _worker():
            try:
                self.after(0, lambda: self._set_status("Tür testi çalışıyor (30-60 sn)..."))
                
                # Yüklenen görüntüyü hazırla (2D grayscale)
                image_data = luma_snap.astype(np.float64)
                if len(image_data.shape) == 3:
                    image_data = image_data[:, :, 0]  # İlk kanalı al
                
                # Görüntü türünü sınıflandır
                classifier = ImageClassifier()
                detected_type, features = classifier.classify(image_data)
                image_type_name = detected_type.value  # "natural", "synthetic", "hybrid"
                
                # Tür testi çalıştır (yüklenen görüntüyle)
                runner = ImageTypeComparisonRunner(image_size=128)  # image_size parametre olarak fakat kullanılmayacak
                results = runner.compare_single_image(image_data, image_type_name)
                
                def _show():
                    import tkinter as tk
                    from tkinter import ttk
                    
                    win = tk.Toplevel()
                    win.title("Görüntü Türü Karşılaştırma Sonuçları — YÜKLENEN GÖRÜNTÜ")
                    win.geometry("1200x820")
                    win.minsize(900, 640)
                    
                    # Ana canvas + scrollbar (içerik genişliği = pencere genişliği)
                    canvas = tk.Canvas(win, bg="white", highlightthickness=0)
                    scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
                    scrollable_frame = tk.Frame(canvas, bg="white")
                    
                    def _sync_scroll_region(_event=None):
                        canvas.configure(scrollregion=canvas.bbox("all"))
                    
                    scrollable_frame.bind("<Configure>", lambda e: _sync_scroll_region())
                    
                    inner_win = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                    
                    def _stretch_inner(event):
                        canvas.itemconfigure(inner_win, width=event.width)
                    
                    canvas.bind("<Configure>", _stretch_inner)
                    canvas.configure(yscrollcommand=scrollbar.set)
                    canvas.pack(side="left", fill="both", expand=True)
                    scrollbar.pack(side="right", fill="y")
                    
                    def _on_type_test_wheel(event):
                        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                    
                    canvas.bind("<MouseWheel>", _on_type_test_wheel)
                    
                    # 2×2 grid: eşit sütunlar, iki satır esnek
                    scrollable_frame.columnconfigure(0, weight=1, uniform="type_test_cols")
                    scrollable_frame.columnconfigure(1, weight=1, uniform="type_test_cols")
                    scrollable_frame.rowconfigure(0, weight=1)
                    scrollable_frame.rowconfigure(1, weight=1)
                    
                    legend_wrap = 280  # yan yana iki panel için metin kırılması
                    
                    # ===== (0,0) DETEKSİYON =====
                    detection_frame = tk.LabelFrame(scrollable_frame, text="🔍 GÖRÜNTÜ SINIFLANDIRMASI",
                                                     font=("Arial", 12, "bold"), bg="#E3F2FD", padx=15, pady=15)
                    detection_frame.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=(12, 8))
                    
                    tk.Label(detection_frame, text=f"Sınıflandırılan Tür: {image_type_name.upper()}",
                            font=("Arial", 13, "bold"), bg="#E3F2FD").pack(fill="x", padx=5, pady=5)
                    
                    # ===== (0,1) PSNR ÖLÇEĞİ =====
                    psnr_frame = tk.LabelFrame(scrollable_frame, text="📊 PSNR ÖLÇEĞÜ (Kalite Metriği - Decibel)",
                                                font=("Arial", 12, "bold"), bg="white", padx=15, pady=15)
                    psnr_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=(12, 8))
                    
                    psnr_data = [
                        ("∞ (inf)", "Mükemmel", "Lossless — hatasız sıkıştırma, özgün ile bit-bit aynı", "#00CC00"),
                        ("40+ dB", "Çok İyi", "Gözle fark edilmeyen kalite kaybı", "#00FF00"),
                        ("30-40 dB", "İyi", "Hafif kalite kaybı, kabul edilebilir", "#99FF00"),
                        ("20-30 dB", "Kabul Edilebilir", "Görünür ancak tolere edilebilir artifacts", "#FFFF00"),
                        ("< 20 dB", "Kötü", "Ciddi kalite kaybı, kullanılmaz", "#FF6666"),
                    ]
                    
                    for value, level, desc, color in psnr_data:
                        row = tk.Frame(psnr_frame, bg=color, relief="solid", borderwidth=1)
                        row.pack(fill="x", padx=5, pady=5)
                        tk.Label(row, text=value, font=("Courier", 11, "bold"), width=10, bg=color, fg="black").pack(side="left", padx=8, pady=6)
                        tk.Label(row, text=f"→ {level}", font=("Arial", 11, "bold"), bg=color, fg="black").pack(side="left", padx=4)
                        tk.Label(row, text=desc, font=("Arial", 10), bg=color, fg="black", wraplength=legend_wrap, justify="left").pack(side="left", padx=6, fill="x", expand=True)
                    
                    # ===== (1,0) SSIM ÖLÇEĞİ =====
                    ssim_frame = tk.LabelFrame(scrollable_frame, text="🎯 SSIM ÖLÇEĞÜ (Yapısal Benzerlik - 0 ile 1 arasında)",
                                                font=("Arial", 12, "bold"), bg="white", padx=15, pady=15)
                    ssim_frame.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(8, 12))
                    
                    ssim_data = [
                        ("1.0", "Mükemmel", "Özgün ile tamamen aynı (Lossless)", "#00CC00"),
                        ("0.95-1.0", "Mükemmel", "Gözle fark edilmez", "#00FF00"),
                        ("0.85-0.95", "Çok İyi", "Çok hafif fark, yüksek kalite", "#99FF00"),
                        ("0.75-0.85", "İyi", "Hafif fark, iyi kalite", "#FFFF00"),
                        ("0.65-0.75", "Orta", "Görünür fark", "#FF9999"),
                        ("< 0.65", "Kötü", "Ciddi kalite kaybı", "#FF6666"),
                    ]
                    
                    for value, level, desc, color in ssim_data:
                        row = tk.Frame(ssim_frame, bg=color, relief="solid", borderwidth=1)
                        row.pack(fill="x", padx=5, pady=5)
                        tk.Label(row, text=value, font=("Courier", 11, "bold"), width=10, bg=color, fg="black").pack(side="left", padx=8, pady=6)
                        tk.Label(row, text=f"→ {level}", font=("Arial", 11, "bold"), bg=color, fg="black").pack(side="left", padx=4)
                        tk.Label(row, text=desc, font=("Arial", 10), bg=color, fg="black", wraplength=legend_wrap, justify="left").pack(side="left", padx=6, fill="x", expand=True)
                    
                    # ===== (1,1) TEST SONUÇLARI =====
                    results_frame = tk.LabelFrame(scrollable_frame, text="📈 TEST SONUÇLARI",
                                                   font=("Arial", 12, "bold"), bg="white", padx=8, pady=10)
                    results_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=(8, 12))
                    
                    # Header
                    header = tk.Frame(results_frame, bg="#333333")
                    header.pack(fill="x", padx=5, pady=5)
                    tk.Label(header, text="Tür", font=("Courier", 9, "bold"), width=12, bg="#333333", fg="white").pack(side="left", padx=2, pady=6)
                    tk.Label(header, text="Yöntem", font=("Courier", 9, "bold"), width=10, bg="#333333", fg="white").pack(side="left", padx=2)
                    tk.Label(header, text="Dönüşüm", font=("Courier", 9, "bold"), width=8, bg="#333333", fg="white").pack(side="left", padx=2)
                    tk.Label(header, text="Kuantalama", font=("Courier", 9, "bold"), width=10, bg="#333333", fg="white").pack(side="left", padx=2)
                    tk.Label(header, text="Entropy", font=("Courier", 9, "bold"), width=9, bg="#333333", fg="white").pack(side="left", padx=2)
                    tk.Label(header, text="Kats.", font=("Courier", 9, "bold"), width=5, bg="#FF6600", fg="white").pack(side="left", padx=2)
                    tk.Label(header, text="PSNR (dB)", font=("Courier", 9, "bold"), width=10, bg="#333333", fg="white").pack(side="left", padx=2)
                    tk.Label(header, text="SSIM", font=("Courier", 9, "bold"), width=8, bg="#333333", fg="white").pack(side="left", padx=2)
                    
                    # Metadata mapping
                    method_metadata = {
                        "DWT": ("DWT", "Scalar", "Huffman"),
                        "DCT": ("DCT", "Uniform", "Huffman"),
                        "Lossless": ("--", "--", "DEFLATE"),
                    }
                    
                    # Veri satırları
                    for idx, r in enumerate(results):
                        psnr_s = f"{r.psnr:.1f}" if r.psnr != -1.0 else "∞"
                        color = "#EEEEEE" if idx % 2 == 0 else "white"
                        row = tk.Frame(results_frame, bg=color, relief="solid", borderwidth=1)
                        row.pack(fill="x", padx=5, pady=2)
                        
                        # Get metadata for this method
                        transform, quantization, entropy = method_metadata.get(r.method, ("?", "?", "?"))
                        
                        # Gerçek quality bilgisini r objesi'nden al
                        quality = r.quality
                        
                        # Q değerine göre arka plan rengi (gradient effect)
                        if quality < 60:
                            q_bg = "#FF9999"  # Kırmızı (düşük - Q50)
                        elif quality < 80:
                            q_bg = "#FFFF99"  # Sarı (orta - Q70)
                        else:
                            q_bg = "#99FF99"  # Yeşil (yüksek - Q85, Q95)
                        
                        tk.Label(row, text=r.image_type, font=("Courier", 9), width=12, bg=color).pack(side="left", padx=2, pady=4)
                        tk.Label(row, text=r.method, font=("Courier", 9), width=10, bg=color).pack(side="left", padx=2)
                        tk.Label(row, text=transform, font=("Courier", 9, "bold"), width=8, bg=color, fg="#0066CC").pack(side="left", padx=2)
                        tk.Label(row, text=quantization, font=("Courier", 9, "bold"), width=10, bg=color, fg="#CC0066").pack(side="left", padx=2)
                        tk.Label(row, text=entropy, font=("Courier", 9, "bold"), width=9, bg=color, fg="#006600").pack(side="left", padx=2)
                        tk.Label(row, text=str(quality), font=("Courier", 9, "bold"), width=5, bg=q_bg, fg="black").pack(side="left", padx=2)
                        tk.Label(row, text=psnr_s, font=("Courier", 9, "bold"), width=10, bg=color).pack(side="left", padx=2)
                        tk.Label(row, text=f"{r.ssim:.3f}", font=("Courier", 9, "bold"), width=8, bg=color).pack(side="left", padx=2)
                    
                    # ===== SONUÇLAR (tam genişlik, grid alt satır) =====
                    conclusion_frame = tk.LabelFrame(scrollable_frame, text="✅ SONUÇLAR",
                                                      font=("Arial", 12, "bold"), bg="#E8F5E9", padx=15, pady=15)
                    conclusion_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(4, 14))
                    
                    conclusions = [
                        f"🎯 Yüklenen Görüntü Türü: {image_type_name.upper()}",
                        "",
                        "📊 Bu Test Sonuçları:",
                        f"   → Senin yüklediğin {image_data.shape[0]}×{image_data.shape[1]} görüntüsü test edildi",
                        "   → DWT, DCT ve Lossless yöntemleri karşılaştırıldı",
                        "   → Q=50, 70, 85, 95 kalite seviyelerinde test yapıldı",
                        "",
                        "📈 Rate-Distortion Eğrisi:",
                        "   → X-ekseni = Bitrate (sıkıştırma oranı)",
                        "   → Y-ekseni = PSNR/SSIM (kalite)",
                        "   → Eğri = Kalite-bitrate tradeoff'ını gösterir",
                    ]
                    
                    for conclusion in conclusions:
                        tk.Label(conclusion_frame, text=conclusion, font=("Arial", 10), 
                                bg="#E8F5E9", justify="left", anchor="w").pack(fill="x", padx=5, pady=3)
                
                self.after(0, _show)
                self.after(0, lambda: self._set_status("✓ Tür testi tamamlandı."))
            except Exception as e:
                self._report_error("Tür testi hata", e, code="ERR-TYPE-01", threaded=True)

        threading.Thread(target=_worker, daemon=True).start()


    def _show_layers(self):
        """DWT katmanlarını ayrı pencerede detaylı olarak göster"""
        if not hasattr(self, '_last_coeffs') or self._last_coeffs is None:
            import tkinter.messagebox as mb
            mb.showwarning('Uyarı', 'Henüz sıkıştırma yapılmadı.\n\nÖnce görüntü yükleyin ve sıkıştırma yapın.')
            return
        
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        
        # Yeni pencere aç
        window = tk.Toplevel(self)
        window.title('DWT Katmanları (Detaylı Görünüm)')
        window.geometry('1000x800')
        
        try:
            coeffs = self._last_coeffs
            if coeffs is None:
                return

            def _band_image(band, detail=False):
                arr = np.asarray(band, dtype=np.float64)
                arr = np.abs(arr)
                if detail:
                    arr = np.log1p(arr)
                if arr.size == 0:
                    return arr
                high = float(np.percentile(arr, 99.0)) if np.any(arr > 0) else 1.0
                low = float(np.percentile(arr, 1.0)) if np.any(arr > 0) else 0.0
                if high - low < 1e-9:
                    return np.zeros_like(arr, dtype=np.float32)
                return np.clip((arr - low) / (high - low), 0.0, 1.0)

            images = []
            titles = []
            current_level = max(1, len(coeffs) - 1)

            images.append(_band_image(coeffs[0], detail=False))
            titles.append(f'LL (Seviye {current_level})')

            for detail_coeffs in coeffs[1:]:
                cH, cV, cD = detail_coeffs
                images.append(_band_image(cH, detail=True))
                titles.append(f'LH (Seviye {current_level})')
                images.append(_band_image(cV, detail=True))
                titles.append(f'HL (Seviye {current_level})')
                images.append(_band_image(cD, detail=True))
                titles.append(f'HH (Seviye {current_level})')
                current_level -= 1

            num_images = len(images)
            n_cols = int(np.ceil(np.sqrt(num_images)))
            n_rows = int(np.ceil(num_images / n_cols))

            fig = Figure(figsize=(3.2 * n_cols, 3.0 * n_rows), dpi=100)
            fig.patch.set_facecolor(PANEL_BG)
            fig.suptitle('DWT Alt-Bantları (Detaylı)', fontsize=14, fontweight='bold')

            for idx, (img, title) in enumerate(zip(images, titles), start=1):
                ax = fig.add_subplot(n_rows, n_cols, idx)
                im = ax.imshow(img, cmap='gray', interpolation='nearest')
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.tight_layout(rect=(0, 0, 1, 0.96))
            
            # Canvas'ı pencereye embed et
            canvas = FigureCanvasTkAgg(fig, master=window)
            figure_canvas_redraw(canvas)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            self._report_error("Katman görselleştirme hatası", e, code="ERR-LAYER-01", show_dialog=True)
    
    def _on_image_drag(self, event):
        """Sol panelde sürükleme hareketinde kısa bilgi göster."""
        try:
            self._set_status(f"Sürükleme: x={int(event.x)}, y={int(event.y)}")
        except Exception:
            self._set_status("Sürükleme algılandı")
    
    def run(self):
        self.mainloop()



    def _bgr_to_luma(self, bgr_img):
        import cv2
        # Tüm pipeline boyunca luma ölçeğini 0..255 tut.
        # Girdi zaten grayscale ise doğrudan dön (OpenCV channel hatasını önler).
        arr = np.asarray(bgr_img)
        if arr.ndim == 2:
            return arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[2] == 1:
            return arr[:, :, 0].astype(np.float32)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    def _luma_to_bgr_preview(self, luma):
        import cv2, numpy as np
        arr = np.asarray(luma)
        # Geriye dönük güvenlik: 0..1 ölçek geldiyse 0..255'e çıkar
        if arr.size > 0 and float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        gray = np.clip(arr, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def _draw_histograms(self, coeffs, level):
        """DWT bantlarının histogram dağılımlarını göster"""
        if coeffs is None or not hasattr(self, 'fig'):
            return
        if isinstance(coeffs, dict):
            return
        
        try:
            ll = coeffs[0]
            hl, lh, hh = coeffs[1]
            
            self.fig.clear()
            
            axes = self.fig.subplots(2, 2)
            axes = axes.flatten()
            
            bands = [ll, hl, lh, hh]
            titles = ['LL (Yaklaşım)', 'HL (Yatay)', 'LH (Dikey)', 'HH (Çapraz)']
            colors = ['blue', 'red', 'green', 'purple']
            
            for ax, band, title, color in zip(axes, bands, titles, colors):
                flat_band = band.ravel()
                
                if title != 'LL (Yaklaşım)':
                    flat_band = np.abs(flat_band)
                
                n = flat_band.size
                if n > HISTOGRAM_MAX_SAMPLES:
                    step = max(1, (n + HISTOGRAM_MAX_SAMPLES - 1) // HISTOGRAM_MAX_SAMPLES)
                    flat_band = flat_band[::step]
                
                ax.hist(flat_band, bins=56, color=color, alpha=0.7, edgecolor='black')
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.set_xlabel('Değer', fontsize=8)
                ax.set_ylabel('Frekans', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            self.fig.suptitle(f'DWT Bantları Histogram Dağılımı (Seviye {level})', fontsize=11, fontweight='bold')
            self.fig.tight_layout()
            figure_canvas_redraw(self.canvas)
        except Exception as e:
            self._report_error("Histogram hatası", e, code="ERR-HIST-01")




    def _show_metrics(self):
        """Sıkıştırma metriklerini tablo olarak ayrı pencerede göster"""
        if not hasattr(self, '_last_psnr') or self._last_psnr == '—':
            import tkinter.messagebox as mb
            mb.showinfo('Metrik Tablosu', 'Henüz sıkıştırma yapılmadı.\n\nÖnce görüntü yükleyin ve Uygula butonuna basın.')
            return

        import tkinter as tk
        from tkinter import ttk

        cr_mode = getattr(self, '_last_cr_mode', False)
        mode_str = "Compression Ratio" if cr_mode else "BPP (Bits Per Pixel)"

        window = tk.Toplevel(self)
        window.title('Sıkıştırma Metrikleri')
        window.geometry('560x492' if cr_mode else '560x412')
        window.resizable(True, False)

        frame = ttk.Frame(window, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        title_text = f'Sıkıştırma Sonuçları — Mod: {mode_str}'
        ttk.Label(frame, text=title_text, font=('Arial', 13, 'bold')).pack(pady=(0, 16))

        style = ttk.Style()
        style.configure('Treeview', rowheight=32, font=('Arial', 10))
        style.configure('Treeview.Heading', font=('Arial', 11, 'bold'))

        n_rows = 11 if cr_mode else 7
        columns = ('Metrik', 'Değer')
        tree = ttk.Treeview(frame, columns=columns, height=n_rows, show='headings')
        tree.column('Metrik', width=280, anchor='w')
        tree.column('Değer', width=230, anchor='w')
        tree.heading('Metrik', text='Metrik')
        tree.heading('Değer', text='Değer')

        orig_kb = getattr(self, '_last_orig_bytes', 0) / 1024
        comp_kb = getattr(self, '_last_compressed_bytes', 0) / 1024
        actual_ratio = getattr(self, '_last_actual_ratio', 0.0)

        metrics_data = [
            ('Mod', mode_str),
        ]

        if cr_mode:
            tgt_ratio = getattr(self, '_last_target_ratio', 0.0)
            tgt_kb = getattr(self, '_last_target_size_bytes', 0) / 1024
            metrics_data += [
                ('Hedef Sıkıştırma Oranı', f'{tgt_ratio:.1f}:1'),
                ('Hedef Boyut (KB)', f'{tgt_kb:.2f} KB'),
                ('Orijinal Ham Boyut (KB)', f'{orig_kb:.2f} KB'),
                ('Gerçek Sıkıştırılmış Boyut (KB)', f'{comp_kb:.2f} KB'),
                ('Gerçek Sıkıştırma Oranı', f'{actual_ratio:.2f}:1'),
            ]
        else:
            metrics_data += [
                ('Hedef BPP', str(self.compression_coeff_entry.get())),
                ('Orijinal Ham Boyut (KB)', f'{orig_kb:.2f} KB'),
                ('Gerçek Sıkıştırılmış Boyut (KB)', f'{comp_kb:.2f} KB'),
                ('Gerçek Sıkıştırma Oranı', f'{actual_ratio:.2f}:1'),
            ]

        metrics_data += [
            ('PSNR (dB)', str(self._last_psnr) if hasattr(self, '_last_psnr') else '—'),
            ('MSE', str(self._last_mse) if hasattr(self, '_last_mse') else '—'),
            ('SSIM', str(self._last_ssim) if hasattr(self, '_last_ssim') else '—'),
            ('Huffman / Sıkıştırılmış Boyut (KB)', str(self._last_huff) if hasattr(self, '_last_huff') else '—'),
        ]

        for metric, value in metrics_data:
            tree.insert('', 'end', values=(metric, value))

        tree.pack(fill=tk.BOTH, expand=True)
        ttk.Button(frame, text='Kapat', command=window.destroy).pack(pady=10)

    def _show_jpeg_vs_jpeg2000_comparison(self, *args, **kwargs):
        """JPEG vs JPEG2000 Karşılaştırma Popup'ını açar."""
        if not hasattr(self, '_luma') or self._luma is None:
            import tkinter.messagebox as mb
            mb.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
            
        luma_snap = self._luma
        import threading
        import numpy as np
        from core.image_type_comparison import ImageTypeComparisonRunner
        from core.image_classifier import ImageClassifier

        def _worker():
            try:
                self.after(0, lambda: self._set_status("Yöntemler karşılaştırılıyor (DCT vs DWT)..."))
                
                # Görüntü verisini hazırla
                image_data = luma_snap.astype(np.float64)
                if len(image_data.shape) == 3:
                    image_data = image_data[:, :, 0]
                
                # Görüntü türünü belirle
                classifier = ImageClassifier()
                detected_type, _ = classifier.classify(image_data)
                image_type_name = detected_type.value
                
                # Karşılaştırmayı çalıştır
                runner = ImageTypeComparisonRunner(image_size=image_data.shape[0])
                results = runner.compare_single_image(image_data, image_type_name)
                
                # UI'ı güncelle
                def _done():
                    wv = self.wavelet_var.get()
                    lv = int(self.level_slider.get())
                    self._set_status("Karşılaştırma tamamlandı.")
                    ComparisonWindow(
                        self,
                        results,
                        image_type_name,
                        luma_2d=image_data,
                        wavelet=wv,
                        level=lv,
                        hardness=self._get_compression_coefficient(),
                    )
                
                self.after(0, _done)
                
            except Exception as e:
                self._report_error("Karşılaştırma hatası", e, code="ERR-CMP-01", threaded=True)
        
        threading.Thread(target=_worker, daemon=True).start()

    def _detect_and_show_type(self) -> None:
        """Yüklenen görüntünün türünü algılar ve etiketi günceller."""
        if self._luma is None:
            return
        
        luma_snap = self._luma
        orig_bgr_snap = self._original_bgr
        from core.image_classifier import ImageClassifier
        import threading

        def _worker():
            try:
                # 2D grayscale veriyi hazırla
                image_data = luma_snap.astype(np.float64)
                if len(image_data.shape) == 3:
                    image_data = image_data[:, :, 0]
                
                classifier = ImageClassifier()
                detected_type, _ = classifier.classify(image_data)
                type_name = detected_type.value.upper()
                
                if orig_bgr_snap is not None:
                    res_w, res_h = orig_bgr_snap.shape[1], orig_bgr_snap.shape[0]
                    self.after(0, lambda: self.lbl_orig_caption.configure(
                        text=f"Tür: {type_name} | Çözünürlük: {res_w}x{res_h}",
                        text_color="#1e88e5" if type_name != "UNKNOWN" else TEXT_SECONDARY
                    ))
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

if __name__ == '__main__':
    app = CompressionApp()
    app.run()
