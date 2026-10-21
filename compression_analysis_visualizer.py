"""
Visualization tool for compression analysis comparisons.

Creates charts showing:
- Compression Ratio vs PSNR (Pareto frontier)
- Compression efficiency by image type
- Method comparison (JPEG vs JPEG2000)
- Quality distribution across categories
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple


class CompressionAnalysisVisualizer:
    """Create visualizations for compression analysis results."""
    
    COLORS = {
        "natural": "#2E86AB",      # Blue
        "synthetic": "#A23B72",    # Purple
        "hybrid": "#F18F01",       # Orange
    }
    
    METHODS = {
        "dct": "JPEG (DCT)",
        "dwt": "JPEG2000 (DWT)",
    }
    
    def __init__(self, results_json: str):
        """Load analysis results from JSON file."""
        with open(results_json, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.output_dir = Path("visualization_output")
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_compression_vs_quality(self):
        """
        Plot compression ratio vs PSNR (rate-distortion curve).
        Shows Pareto frontier for each image category.
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Rate-Distortion Analysis: Compression vs Quality", fontsize=14, fontweight='bold')
        
        categories = ["natural", "synthetic", "hybrid"]
        
        for idx, (ax, category) in enumerate(zip(axes, categories)):
            if category not in self.data["raw_results"]:
                continue
            
            results = self.data["raw_results"][category]
            
            # Collect data for plotting
            dct_data = {"ratio": [], "psnr": [], "image": []}
            dwt_data = {"ratio": [], "psnr": [], "image": []}
            
            for result in results:
                img_name = result.get("filename", "")
                
                # DCT data
                for quality, metrics in result.get("dct_results", {}).get("quality_levels", {}).items():
                    dct_data["ratio"].append(metrics["compression_ratio"])
                    dct_data["psnr"].append(metrics["psnr"])
                    dct_data["image"].append(img_name)
                
                # DWT data
                for wavelet, w_data in result.get("dwt_results", {}).get("wavelets", {}).items():
                    for key, metrics in w_data.get("quantization", {}).items():
                        dwt_data["ratio"].append(metrics["compression_ratio"])
                        dwt_data["psnr"].append(metrics["psnr"])
                        dwt_data["image"].append(img_name)
            
            # Plot
            if dct_data["ratio"]:
                ax.scatter(dct_data["ratio"], dct_data["psnr"], 
                          label="JPEG (DCT)", marker='o', s=100, 
                          color='#FF6B6B', alpha=0.7, edgecolors='black')
            
            if dwt_data["ratio"]:
                ax.scatter(dwt_data["ratio"], dwt_data["psnr"], 
                          label="JPEG2000 (DWT)", marker='^', s=100,
                          color='#4ECDC4', alpha=0.7, edgecolors='black')
            
            ax.set_xlabel("Compression Ratio (Original/Compressed)", fontsize=11)
            ax.set_ylabel("PSNR (dB)", fontsize=11)
            ax.set_title(f"{category.capitalize()} Images", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            # Annotations
            if dct_data["ratio"]:
                best_idx = np.argmax(dct_data["psnr"])
                ax.annotate("Best Quality",
                           xy=(dct_data["ratio"][best_idx], dct_data["psnr"][best_idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        filepath = self.output_dir / "01_compression_vs_quality.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_bpp_comparison(self):
        """
        Plot Bits Per Pixel (BPP) across image types and methods.
        Lower BPP = better compression.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = ["natural", "synthetic", "hybrid"]
        methods = ["dct", "dwt"]
        
        x_pos = np.arange(len(categories))
        width = 0.35
        
        dct_bpp = []
        dwt_bpp = []
        
        for category in categories:
            if category not in self.data["raw_results"]:
                dct_bpp.append(0)
                dwt_bpp.append(0)
                continue
            
            results = self.data["raw_results"][category]
            
            # Calculate average BPP
            dct_bpps = []
            for result in results:
                for quality, metrics in result.get("dct_results", {}).get("quality_levels", {}).items():
                    dct_bpps.append(metrics.get("bpp", 0))
            
            dwt_bpps = []
            for result in results:
                for wavelet, w_data in result.get("dwt_results", {}).get("wavelets", {}).items():
                    for key, metrics in w_data.get("quantization", {}).items():
                        dwt_bpps.append(metrics.get("bpp", 0))
            
            dct_bpp.append(np.mean(dct_bpps) if dct_bpps else 0)
            dwt_bpp.append(np.mean(dwt_bpps) if dwt_bpps else 0)
        
        # Plot
        ax.bar(x_pos - width/2, dct_bpp, width, label="JPEG (DCT)", color='#FF6B6B')
        ax.bar(x_pos + width/2, dwt_bpp, width, label="JPEG2000 (DWT)", color='#4ECDC4')
        
        ax.set_xlabel("Image Type", fontsize=12)
        ax.set_ylabel("Bits Per Pixel (BPP) - Lower is Better", fontsize=12)
        ax.set_title("Compression Efficiency: Average BPP by Image Type", fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([cat.capitalize() for cat in categories])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (d, w) in enumerate(zip(dct_bpp, dwt_bpp)):
            if d > 0:
                ax.text(i - width/2, d + 0.1, f"{d:.2f}", ha='center', va='bottom', fontsize=9)
            if w > 0:
                ax.text(i + width/2, w + 0.1, f"{w:.2f}", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filepath = self.output_dir / "02_bpp_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_psnr_distribution(self):
        """
        Box plot showing PSNR distribution across image types.
        Shows quality variation and consistency.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = ["natural", "synthetic", "hybrid"]
        
        # Prepare data
        all_psnr_data = []
        labels = []
        
        for category in categories:
            if category not in self.data["raw_results"]:
                continue
            
            psnr_values = []
            results = self.data["raw_results"][category]
            
            for result in results:
                # Collect PSNR from both methods
                for quality, metrics in result.get("dct_results", {}).get("quality_levels", {}).items():
                    psnr_values.append(metrics["psnr"])
                
                for wavelet, w_data in result.get("dwt_results", {}).get("wavelets", {}).items():
                    for key, metrics in w_data.get("quantization", {}).items():
                        psnr_values.append(metrics["psnr"])
            
            if psnr_values:
                all_psnr_data.append(psnr_values)
                labels.append(category.capitalize())
        
        # Box plot
        bp = ax.boxplot(all_psnr_data, tick_labels=labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], [self.COLORS["natural"], 
                                              self.COLORS["synthetic"],
                                              self.COLORS["hybrid"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel("PSNR (dB)", fontsize=12)
        ax.set_xlabel("Image Type", fontsize=12)
        ax.set_title("Quality Distribution: PSNR Range by Image Type", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = self.output_dir / "03_psnr_distribution.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_compression_ratio_comparison(self):
        """
        Compare average compression ratios across categories and methods.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = ["natural", "synthetic", "hybrid"]
        
        dct_ratios = []
        dwt_ratios = []
        
        for category in categories:
            if category not in self.data["raw_results"]:
                dct_ratios.append(0)
                dwt_ratios.append(0)
                continue
            
            results = self.data["raw_results"][category]
            
            dct_vals = []
            dwt_vals = []
            
            for result in results:
                for quality, metrics in result.get("dct_results", {}).get("quality_levels", {}).items():
                    dct_vals.append(metrics["compression_ratio"])
                
                for wavelet, w_data in result.get("dwt_results", {}).get("wavelets", {}).items():
                    for key, metrics in w_data.get("quantization", {}).items():
                        dwt_vals.append(metrics["compression_ratio"])
            
            dct_ratios.append(np.mean(dct_vals) if dct_vals else 0)
            dwt_ratios.append(np.mean(dwt_vals) if dwt_vals else 0)
        
        x_pos = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x_pos - width/2, dct_ratios, width, label="JPEG (DCT)", color='#FF6B6B')
        ax.bar(x_pos + width/2, dwt_ratios, width, label="JPEG2000 (DWT)", color='#4ECDC4')
        
        ax.set_xlabel("Image Type", fontsize=12)
        ax.set_ylabel("Compression Ratio (Original/Compressed)", fontsize=12)
        ax.set_title("Average Compression Efficiency by Image Type", fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([cat.capitalize() for cat in categories])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (d, w) in enumerate(zip(dct_ratios, dwt_ratios)):
            if d > 0:
                ax.text(i - width/2, d + 0.05, f"{d:.2f}x", ha='center', va='bottom', fontsize=9)
            if w > 0:
                ax.text(i + width/2, w + 0.05, f"{w:.2f}x", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filepath = self.output_dir / "04_compression_ratio_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def generate_all_visualizations(self) -> List[str]:
        """Generate all comparison visualizations."""
        print("Generating visualizations...")
        
        files = []
        
        try:
            print("  • Rate-Distortion curves...", end=" ")
            files.append(self.plot_compression_vs_quality())
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
        
        try:
            print("  • BPP comparison...", end=" ")
            files.append(self.plot_bpp_comparison())
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
        
        try:
            print("  • PSNR distribution...", end=" ")
            files.append(self.plot_psnr_distribution())
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
        
        try:
            print("  • Compression ratio comparison...", end=" ")
            files.append(self.plot_compression_ratio_comparison())
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
        
        return files


def main():
    """Generate visualizations from latest analysis results."""
    from pathlib import Path
    
    # Find latest results file
    results_dir = Path("evaluation_results")
    if not results_dir.exists():
        print("❌ No evaluation results found!")
        print("   Run: python image_comparison_analyzer.py")
        return
    
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print("❌ No JSON results found!")
        return
    
    latest_file = json_files[-1]
    print(f"Using results: {latest_file.name}\n")
    
    visualizer = CompressionAnalysisVisualizer(str(latest_file))
    files = visualizer.generate_all_visualizations()
    
    print(f"\n✅ Visualizations generated: {len(files)}")
    for f in files:
        print(f"   📊 {Path(f).name}")


if __name__ == "__main__":
    main()
