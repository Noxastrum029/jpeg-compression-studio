# Image Compression Comparison Analysis

Systematic evaluation of JPEG (DCT) and JPEG2000 (DWT) compression performance across three image categories.

## Overview

This analysis toolkit helps evaluate and compare:

- **Compression Ratio vs Image Quality** trade-offs
- **Method Performance** (JPEG vs JPEG2000)
- **Image Category Effects** (Natural, Synthetic, Hybrid)
- **Parameter Optimization** (Quality levels, Wavelets, Quantization)

## Image Categories

### 1. 🏞️ Natural Images
**Examples**: Landscapes, portraits, medical imagery, photographs

**Characteristics**:
- Smooth color gradients
- Complex texture patterns
- Subtle color variations
- High-frequency details

**Compression Behavior**:
- JPEG performs well (75-80% quality acceptable)
- Artifacts visible at low quality (blockiness)
- Wavelets preserve detail better than DCT
- Requires higher PSNR for acceptable quality (>30 dB)

---

### 2. 🎨 Synthetic/Computer-Generated Images
**Examples**: Graphics, diagrams, patterns, charts, generated content

**Characteristics**:
- Sharp edges between regions
- Solid color blocks
- Limited color palette
- Geometric shapes and patterns

**Compression Behavior**:
- Both JPEG and JPEG2000 handle well
- Artifacts less noticeable than in natural images
- Lower quality settings (50-75%) may be acceptable
- Maintains quality better at same compression ratio
- Wavelets superior for sharp transitions

---

### 3. 🔀 Hybrid Images
**Examples**: Mixed content (natural photo with text overlay, diagram with photograph)

**Characteristics**:
- Mix of smooth and sharp areas
- Variable complexity
- Different content types in same image
- Conflicting compression requirements

**Compression Behavior**:
- Balanced parameters essential
- No single method dominates
- JPEG2000 with moderate quantization recommended
- Processing parts separately may be beneficial

---

## Workflow

### Step 1: Prepare Test Images

Generate synthetic test images:
```bash
python generate_test_images.py
```

This creates:
- `test_natural_landscape.jpg` - Natural image simulation
- `test_synthetic_graphic.jpg` - Pure synthetic graphic
- `test_hybrid_mixed.jpg` - Hybrid content

---

### Step 2: Run Compression Analysis

Analyze all available images:
```bash
python image_comparison_analyzer.py
```

**What it does**:
1. Categorizes each image (natural/synthetic/hybrid)
2. Tests JPEG compression at 5 quality levels: 100%, 75%, 50%, 25%, 10%
3. Tests JPEG2000 with 3 wavelets (haar, db4, db8) × 3 levels × 4 quantization values
4. Calculates metrics:
   - **Compression Ratio** = Original Size / Compressed Size
   - **PSNR** (Peak Signal-to-Noise Ratio) - Quality [dB]
   - **MSE** (Mean Squared Error) - Reconstruction error
   - **BPP** (Bits Per Pixel) - File efficiency

**Output**:
- `evaluation_results/compression_analysis_YYYYMMDD_HHMMSS.json` - Raw data
- `evaluation_results/compression_analysis_YYYYMMDD_HHMMSS.csv` - Spreadsheet format

---

### Step 3: Generate Visualizations

Create comparison charts:
```bash
python compression_analysis_visualizer.py
```

**Generates**:

1. **Rate-Distortion Curves** (`01_compression_vs_quality.png`)
   - Shows compression ratio vs PSNR for each image type
   - Reveals Pareto frontier (best quality for given ratio)
   - Separate plots for natural, synthetic, hybrid

2. **BPP Comparison** (`02_bpp_comparison.png`)
   - Average bits per pixel across methods
   - Shows compression efficiency
   - Lower is better

3. **PSNR Distribution** (`03_psnr_distribution.png`)
   - Box plot of quality across compression settings
   - Shows consistency and range
   - Identifies outliers

4. **Compression Ratio Comparison** (`04_compression_ratio_comparison.png`)
   - Average compression efficiency by category
   - Direct method comparison (JPEG vs JPEG2000)

---

## Key Metrics Explained

### Compression Ratio
```
Ratio = Original Size (bytes) / Compressed Size (bytes)
```
- **2:1** = 50% file size reduction
- **5:1** = 80% reduction (aggressive)
- **10:1** = 90% reduction (very aggressive)

### PSNR (Peak Signal-to-Noise Ratio)
```
PSNR = 20 × log₁₀(MAX_VALUE / √MSE)  [dB]
```

**Quality Guide**:
- **> 40 dB** = Very high quality (visually lossless)
- **35-40 dB** = High quality (acceptable for most uses)
- **28-35 dB** = Medium quality (some visible artifacts)
- **< 28 dB** = Low quality (noticeable degradation)

### BPP (Bits Per Pixel)
```
BPP = (Compressed Size × 8) / (Width × Height)
```
- Original images typically 8-24 BPP
- Compressed images 0.1-5 BPP depending on quality
- Lower = more aggressive compression

### MSE (Mean Squared Error)
```
MSE = (1/N) × Σ(Original[i] - Compressed[i])²
```
- Measures average pixel-level difference
- Lower is better (0 = identical)
- Sensitive to outliers

---

## Reading the Results

### JSON Output Structure
```json
{
  "summary": { ... },
  "by_category": {
    "natural": {
      "image_count": 2,
      "dct_summary": { ... },
      "dwt_summary": { ... },
      "recommendations": [ ... ]
    },
    "synthetic": { ... },
    "hybrid": { ... }
  },
  "raw_results": {
    "natural": [
      {
        "filename": "test_natural_landscape.jpg",
        "dct_results": {
          "quality_levels": {
            "100": {
              "compression_ratio": 2.5,
              "psnr": 42.8,
              "bpp": 2.4
            },
            ...
          }
        },
        "dwt_results": { ... }
      }
    ]
  }
}
```

### CSV Output Format
Rows with columns:
- `category` - Image type (natural/synthetic/hybrid)
- `image` - Filename
- `method` - Compression method (DCT/DWT)
- `param` - Specific parameters used
- `compression_ratio` - Efficiency ratio
- `bpp` - Bits per pixel
- `psnr` - Quality in dB
- `mse` - Reconstruction error

**Example Query in Excel/Python**:
```python
import pandas as pd

df = pd.read_csv("compression_analysis.csv")

# Best PSNR for synthetic images with DCT
best = df[(df['category'] == 'synthetic') & 
          (df['method'] == 'DCT')].nlargest(1, 'psnr')

# Average BPP by category
avg_bpp = df.groupby('category')['bpp'].mean()
```

---

## Recommendations

### For Natural Images (Landscapes, Portraits, Medical)
```
✓ JPEG (DCT) at 75-80% quality
  - Good balance of quality and file size
  - Standard for photography

✓ JPEG2000 with db4 or db6 wavelet
  - Better detail preservation
  - Use moderate quantization (delta 2-5)
  
✗ Avoid extreme compression (< 50% quality)
  - Blockiness artifacts visible
  - Loss of fine texture details
```

---

### For Synthetic Images (Graphics, Diagrams, Charts)
```
✓ JPEG at 60-75% quality acceptable
  - Less visible compression artifacts
  - Sharp edges handled well

✓ JPEG2000 with any wavelet (prefer db4)
  - Edge preservation superior
  - Can use aggressive quantization

✓ Consider Lossless if pixel-perfect needed
  - Use for diagrams, maps, technical drawings
```

---

### For Hybrid Images (Mixed Content)
```
⚖ Balanced approach required
  - Neither method dominates
  - Trade-offs unavoidable

✓ JPEG2000 with moderate settings:
  - Wavelet: db4 or bior4.4
  - Quantization: delta 2-10
  - Decomposition: level 2-3

⚠ Monitor PSNR carefully
  - Natural parts may need higher quality
  - Synthetic parts tolerate lower quality
  - Consider separate processing
```

---

## Advanced Analysis

### Find Optimal Compression Level
```python
import pandas as pd

df = pd.read_csv("compression_analysis.csv")

# Find point with best PSNR at each compression ratio
for category in ['natural', 'synthetic', 'hybrid']:
    subset = df[df['category'] == category]
    
    # Group by compression ratio, find best PSNR
    optimal = subset.loc[subset.groupby('compression_ratio')['psnr'].idxmax()]
    
    print(f"\n{category.upper()}")
    print(optimal[['method', 'param', 'compression_ratio', 'psnr', 'bpp']])
```

### Compare Methods for Specific Quality Target
```python
# Find settings that achieve 35 dB PSNR with minimum file size
target_psnr = 35.0

for category in ['natural', 'synthetic', 'hybrid']:
    viable = df[(df['category'] == category) & 
                (df['psnr'] >= target_psnr)]
    
    best = viable.loc[viable['compression_ratio'].idxmax()]
    
    print(f"\n{category}: Best {target_psnr} dB quality")
    print(f"  Method: {best['method']}")
    print(f"  Compression: {best['compression_ratio']:.2f}x")
    print(f"  BPP: {best['bpp']:.2f}")
```

---

## Files Generated

```
project_root/
├── image_comparison_analyzer.py      # Main analysis script
├── compression_analysis_visualizer.py # Visualization generator
├── generate_test_images.py            # Test image creator
│
├── evaluation_results/                # Analysis output
│   ├── compression_analysis_*.json    # Raw results
│   └── compression_analysis_*.csv     # Spreadsheet data
│
├── visualization_output/              # Charts
│   ├── 01_compression_vs_quality.png
│   ├── 02_bpp_comparison.png
│   ├── 03_psnr_distribution.png
│   └── 04_compression_ratio_comparison.png
│
└── test_images_generated/             # Synthetic test images
    ├── test_natural_landscape.jpg
    ├── test_synthetic_graphic.jpg
    └── test_hybrid_mixed.jpg
```

---

## Troubleshooting

### ImportError for image_comparison_analyzer
The analyzer imports from your project's core modules. Ensure:
```bash
# Run from project root
cd signalprojelast

# Install dependencies if needed
pip install -r requirements.txt

# Then run analyzer
python image_comparison_analyzer.py
```

### No test images found
Generate them first:
```bash
python generate_test_images.py
```

### Slow analysis
- Analysis is CPU-intensive (DWT calculations)
- Consider reducing parameters in analyzer:
  - Fewer wavelets: `["db4"]` instead of `["haar", "db4", "db8"]`
  - Fewer levels: `[2]` instead of `[1, 2, 3]`
  - Fewer quantization values: `[1.0, 10.0]` instead of `[1.0, 5.0, 10.0, 20.0]`

---

## Next Steps

1. ✅ Generate test images: `python generate_test_images.py`
2. ✅ Run analysis: `python image_comparison_analyzer.py`
3. ✅ Generate visualizations: `python compression_analysis_visualizer.py`
4. 📊 Review results in `evaluation_results/` and `visualization_output/`
5. 🔍 Analyze CSV data in Excel/Python for specific queries
6. 📈 Use insights to optimize parameters for your use case

---

## Project Integration

These analysis tools integrate with your CENG 384 project to:
- **Validate** compression performance claims
- **Optimize** parameter selection
- **Document** method suitability for different image types
- **Support** project report and presentation

Use the generated visualizations and data in your project documentation!
