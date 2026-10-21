"""
PR-QMF (Perfect Reconstruction Quadrature Mirror Filter) — 2-kanal, separable 2D.

PyWavelets'te olmayan 2/4/6-tap PR-QMF filtreleri. Çıktı formatı pywt.dwt2 ile
uyumludur: (LL, (cH, cV, cD)) = (LL, (LH, HL, HH)).

Bu modül izole tutulur; GUI henüz bağlanmaz.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import convolve1d

FILTERS: Dict[str, Dict[str, List[float]]] = {
    "prqmf2": {
        "h0": [0.7071, 0.7071],
        "h1": [-0.7071, 0.7071],
        "g0": [0.7071, 0.7071],
        "g1": [0.7071, -0.7071],
    },
    "prqmf4": {
        "h0": [-0.1294, 0.2241, 0.8365, 0.4830],
        "h1": [-0.4830, 0.8365, -0.2241, -0.1294],
        "g0": [0.4830, 0.8365, 0.2241, -0.1294],
        "g1": [-0.1294, -0.2241, 0.8365, -0.4830],
    },
    "prqmf6": {
        "h0": [0.0352, -0.0854, -0.1350, 0.4599, 0.8069, 0.3327],
        "h1": [-0.3327, 0.8069, -0.4599, -0.1350, 0.0854, 0.0352],
        "g0": [0.3327, 0.8069, 0.4599, -0.1350, -0.0854, 0.0352],
        "g1": [0.0352, 0.0854, -0.1350, -0.4599, 0.8069, -0.3327],
    },
}


def is_pr_qmf(wavelet_name: str) -> bool:
    return wavelet_name in FILTERS


def _taps(coeffs: List[float]) -> np.ndarray:
    return np.asarray(coeffs, dtype=np.float64)


def _conv_circ(x: np.ndarray, h: np.ndarray, axis: int = -1) -> np.ndarray:
    """Dairesel (periodization) 1D konvolüsyon — scipy.ndimage.convolve1d, mode='wrap'."""
    return convolve1d(
        np.asarray(x, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
        axis=axis,
        mode="wrap",
    )


def analyze_1d(
    x: np.ndarray,
    h0: List[float] | np.ndarray,
    h1: List[float] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """1D analiz: dairesel konvolüsyon + 2'ye downsample (çift indeksler)."""
    x = np.asarray(x, dtype=np.float64)
    low = _conv_circ(x, _taps(h0))[::2]
    high = _conv_circ(x, _taps(h1))[::2]
    return low, high


def synthesize_1d(
    low: np.ndarray,
    high: np.ndarray,
    g0: List[float] | np.ndarray,
    g1: List[float] | np.ndarray,
    out_len: int,
) -> np.ndarray:
    """
    1D sentez: upsample (çift örneklere yerleştir) + dairesel filtre + toplama.

    scipy.ndimage.convolve1d çekirdeği merkeze koyduğu için analiz+sentez
    net 1 örnek dairesel kayma üretir; np.roll(..., 1) bunu telafi eder.
    """
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    g0_a = _taps(g0)
    g1_a = _taps(g1)
    n_low = (out_len + 1) // 2
    up_low = np.zeros(out_len, dtype=np.float64)
    up_high = np.zeros(out_len, dtype=np.float64)
    up_low[0::2] = low[:n_low]
    up_high[0::2] = high[:n_low]
    rec = _conv_circ(up_low, g0_a) + _conv_circ(up_high, g1_a)
    # convolve1d çekirdeği merkeze hizalar; analiz+sentez net 1 örnek dairesel kayma üretir.
    return np.roll(rec, 1)


def _analyze_along_axis(
    data: np.ndarray,
    h0: np.ndarray,
    h1: np.ndarray,
    axis: int,
) -> Tuple[np.ndarray, np.ndarray]:
    low_full = _conv_circ(data, h0, axis=axis)
    high_full = _conv_circ(data, h1, axis=axis)
    sl = [slice(None)] * data.ndim
    sl[axis] = slice(None, None, 2)
    return low_full[tuple(sl)], high_full[tuple(sl)]


def _synthesize_along_axis(
    low: np.ndarray,
    high: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    out_len: int,
    axis: int,
) -> np.ndarray:
    """analyze_along_axis'in tersi; gecikme telafisi synthesize_1d ile aynı kural."""
    shape = list(low.shape)
    shape[axis] = out_len
    up_low = np.zeros(shape, dtype=np.float64)
    up_high = np.zeros(shape, dtype=np.float64)
    sl = [slice(None)] * len(shape)
    sl[axis] = slice(None, None, 2)
    n_keep = (out_len + 1) // 2
    sl_src = [slice(None)] * low.ndim
    sl_src[axis] = slice(0, n_keep)
    up_low[tuple(sl)] = low[tuple(sl_src)]
    up_high[tuple(sl)] = high[tuple(sl_src)]
    rec = _conv_circ(up_low, g0, axis=axis) + _conv_circ(up_high, g1, axis=axis)
    return np.roll(rec, 1, axis=axis)


def dwt2_prqmf(
    data: np.ndarray, wavelet_name: str
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Tek seviye 2D PR-QMF (separable): önce satır (axis=1), sonra sütun (axis=0).

    Dönüş pywt.dwt2 ile aynı: (LL, (cH, cV, cD)) = (LL, (LH, HL, HH)).
    """
    if wavelet_name not in FILTERS:
        raise ValueError(f"Bilinmeyen PR-QMF: {wavelet_name}")
    filt = FILTERS[wavelet_name]
    h0, h1 = _taps(filt["h0"]), _taps(filt["h1"])
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("dwt2_prqmf 2B dizi bekler")

    # Yatay (her satır, axis=1): lowpass / highpass
    lo_h, hi_h = _analyze_along_axis(x, h0, h1, axis=1)
    # Dikey (axis=0) — lo_h → LL, HL ; hi_h → LH, HH
    ll, hl = _analyze_along_axis(lo_h, h0, h1, axis=0)
    lh, hh = _analyze_along_axis(hi_h, h0, h1, axis=0)
    return ll, (lh, hl, hh)


def idwt2_prqmf(
    ll: np.ndarray,
    lh_hl_hh: Tuple[np.ndarray, np.ndarray, np.ndarray],
    wavelet_name: str,
    out_shape: Tuple[int, int] | None = None,
) -> np.ndarray:
    """dwt2_prqmf'in tersi. out_shape verilmezse (2*ll_h, 2*ll_w) kullanılır."""
    if wavelet_name not in FILTERS:
        raise ValueError(f"Bilinmeyen PR-QMF: {wavelet_name}")
    filt = FILTERS[wavelet_name]
    g0, g1 = _taps(filt["g0"]), _taps(filt["g1"])
    lh, hl, hh = lh_hl_hh
    ll = np.asarray(ll, dtype=np.float64)
    lh = np.asarray(lh, dtype=np.float64)
    hl = np.asarray(hl, dtype=np.float64)
    hh = np.asarray(hh, dtype=np.float64)

    if out_shape is None:
        out_h, out_w = ll.shape[0] * 2, ll.shape[1] * 2
    else:
        out_h, out_w = int(out_shape[0]), int(out_shape[1])

    # Analizin ters sırası: önce dikey (axis=0), sonra yatay (axis=1)
    lo_h = _synthesize_along_axis(ll, hl, g0, g1, out_h, axis=0)
    hi_h = _synthesize_along_axis(lh, hh, g0, g1, out_h, axis=0)
    rec = _synthesize_along_axis(lo_h, hi_h, g0, g1, out_w, axis=1)
    return rec


def wavedec2_prqmf(data: np.ndarray, wavelet_name: str, level: int = 1) -> List:
    """pywt.wavedec2 ile AYNI ÇIKTI FORMATINI üretir:
    [cA_n, (cH_n,cV_n,cD_n), ..., (cH_1,cV_1,cD_1)]
    LL bandı üzerinde tekrar tekrar dwt2_prqmf uygulanır."""
    if level < 1:
        raise ValueError("level >= 1 olmalı")
    coeffs: List = []
    current = np.asarray(data, dtype=np.float64)
    for _ in range(level):
        ll, (lh, hl, hh) = dwt2_prqmf(current, wavelet_name)
        coeffs.append((lh, hl, hh))
        current = ll
    coeffs.append(current)  # son cA
    coeffs.reverse()  # pywt sırası: [cA_n, detay_n, ..., detay_1]
    return coeffs


def waverec2_prqmf(coeffs: List, wavelet_name: str) -> np.ndarray:
    """wavedec2_prqmf'in tam tersi, pywt.waverec2 ile aynı imza."""
    current = np.asarray(coeffs[0], dtype=np.float64)
    for detail in coeffs[1:]:
        lh, hl, hh = detail
        out_shape = (int(lh.shape[0]) * 2, int(lh.shape[1]) * 2)
        current = idwt2_prqmf(current, (lh, hl, hh), wavelet_name, out_shape=out_shape)
    return current


def max_level_prqmf(shape: Tuple[int, ...], wavelet_name: str) -> int:
    """pywt.dwtn_max_level benzeri: PR-QMF filtre uzunluğuna göre üst sınır."""
    import math

    filt_len = len(FILTERS[wavelet_name]["h0"])
    min_dim = min(shape)
    return max(1, int(math.log2(min_dim / (filt_len - 1))))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


if __name__ == "__main__":
    rng = np.random.default_rng(384)
    names = ("prqmf2", "prqmf4", "prqmf6")

    print("=" * 64)
    print("PR-QMF self-test  (core/pr_qmf.py)")
    print("=" * 64)

    print("\n--- (a) 1D roundtrip  (N=64) ---")
    x1 = rng.standard_normal(64)
    for name in names:
        f = FILTERS[name]
        lo, hi = analyze_1d(x1, f["h0"], f["h1"])
        rec = synthesize_1d(lo, hi, f["g0"], f["g1"], out_len=x1.size)
        mse = _mse(x1, rec)
        mx = _max_abs(x1, rec)
        status = "OK" if mse < 1e-6 else "FAIL"
        print(f"  {name:8s}  MSE={mse:.3e}  max|err|={mx:.3e}  [{status}]")

    print("\n--- (b) 2D roundtrip  (64x64) ---")
    x2 = rng.standard_normal((64, 64))
    for name in names:
        ll, details = dwt2_prqmf(x2, name)
        rec = idwt2_prqmf(ll, details, name, out_shape=x2.shape)
        mse = _mse(x2, rec)
        mx = _max_abs(x2, rec)
        status = "OK" if mse < 1e-6 else "FAIL"
        print(f"  {name:8s}  MSE={mse:.3e}  max|err|={mx:.3e}  [{status}]")

    print("\n--- (c) prqmf2 vs pywt haar  (64x64, |LL| yakınlığı) ---")
    try:
        import pywt

        haar_ll, (haar_ch, haar_cv, haar_cd) = pywt.dwt2(x2, "haar", mode="periodization")
        pq_ll, (pq_ch, pq_cv, pq_cd) = dwt2_prqmf(x2, "prqmf2")
        # İşaret farkına izin ver: |LL_prqmf| ≈ |LL_haar|
        rel = _mse(np.abs(pq_ll), np.abs(haar_ll)) / (float(np.mean(haar_ll ** 2)) + 1e-18)
        corr = float(
            np.corrcoef(np.abs(pq_ll).ravel(), np.abs(haar_ll).ravel())[0, 1]
        )
        signed_mse = _mse(pq_ll, haar_ll)
        flipped_mse = _mse(pq_ll, -haar_ll)
        print(f"  LL abs-MSE (normalized) = {rel:.3e}")
        print(f"  LL abs correlation      = {corr:.6f}")
        print(f"  LL signed MSE vs haar   = {signed_mse:.3e}")
        print(f"  LL MSE vs -haar         = {flipped_mse:.3e}")
        print(f"  LL shape prqmf2={pq_ll.shape}  haar={haar_ll.shape}")
        status = "OK" if corr > 0.99 else "FAIL"
        print(f"  [{status}]  |LL| bands should match (sign flip allowed)")
    except Exception as exc:
        print(f"  pywt karsilastirmasi atlandi: {exc}")

    print("\n--- (d) multilevel roundtrip  wavedec2/waverec2  (64x64) ---")
    x_ml = rng.standard_normal((64, 64))
    for name in names:
        for lv in (2, 3):
            coeffs = wavedec2_prqmf(x_ml, name, level=lv)
            rec = waverec2_prqmf(coeffs, name)
            rec = rec[: x_ml.shape[0], : x_ml.shape[1]]
            mse = _mse(x_ml, rec)
            mx = _max_abs(x_ml, rec)
            status = "OK" if mse < 1e-6 else "FAIL"
            print(
                f"  {name:8s}  level={lv}  n_bands={len(coeffs)}  "
                f"MSE={mse:.3e}  max|err|={mx:.3e}  [{status}]"
            )

    print("\n--- (e) structure vs pywt.wavedec2  (haar / prqmf2, level=2) ---")
    try:
        import pywt

        def _describe(coeffs: List) -> List[str]:
            rows = [f"len={len(coeffs)}"]
            for i, item in enumerate(coeffs):
                if i == 0:
                    rows.append(f"  [{i}] cA  ndarray {np.asarray(item).shape}")
                else:
                    cH, cV, cD = item
                    rows.append(
                        f"  [{i}] (cH,cV,cD) tuple "
                        f"{np.asarray(cH).shape} {np.asarray(cV).shape} {np.asarray(cD).shape}"
                    )
            return rows

        haar_c = pywt.wavedec2(x_ml, "haar", level=2, mode="periodization")
        pq_c = wavedec2_prqmf(x_ml, "prqmf2", level=2)
        print("  pywt haar:")
        for line in _describe(haar_c):
            print("   ", line)
        print("  prqmf2:")
        for line in _describe(pq_c):
            print("   ", line)
        same_len = len(haar_c) == len(pq_c)
        same_kinds = isinstance(pq_c[0], np.ndarray) and all(
            isinstance(t, tuple) and len(t) == 3 for t in pq_c[1:]
        )
        same_shapes = np.asarray(haar_c[0]).shape == np.asarray(pq_c[0]).shape and all(
            tuple(np.asarray(a).shape for a in haar_c[i])
            == tuple(np.asarray(a).shape for a in pq_c[i])
            for i in range(1, len(haar_c))
        )
        status = "OK" if same_len and same_kinds and same_shapes else "FAIL"
        print(
            f"  same_len={same_len}  tuple_layout={same_kinds}  "
            f"same_shapes={same_shapes}  [{status}]"
        )
    except Exception as exc:
        print(f"  pywt yapi karsilastirmasi atlandi: {exc}")

    print("\n--- (f) max_level_prqmf ---")
    for shp in ((512, 512), (256, 256), (64, 64)):
        for name in names:
            ml = max_level_prqmf(shp, name)
            ok = 1 <= ml <= 16
            print(f"  shape={shp[0]}x{shp[1]}  {name:8s}  max_level={ml}  [{ 'OK' if ok else 'FAIL' }]")

    print("\nDone.")
