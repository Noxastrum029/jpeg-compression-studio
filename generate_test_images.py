"""
Generate synthetic test images for compression evaluation.

Creates:
1. Natural image simulation (complex patterns, gradients)
2. Synthetic graphic (simple shapes, solid colors)
3. Hybrid image (mix of patterns and solid areas)
"""

import numpy as np
import cv2
from pathlib import Path


def create_natural_image_sim(height: int = 512, width: int = 512) -> np.ndarray:
    """
    Create synthetic image simulating natural photo characteristics:
    - Smooth gradients
    - Color variation
    - Some high-frequency details
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient background (simulates sky/landscape)
    for y in range(height):
        for x in range(width):
            # Blue-ish sky at top, green-ish at bottom
            b = int(200 - (y / height) * 100)
            g = int(100 + (y / height) * 100)
            r = int(150 + (x / width) * 50)
            img[y, x] = [b, g, r]
    
    # Add Perlin-like noise using sine waves (simulates texture)
    for y in range(height):
        for x in range(width):
            noise = int(30 * np.sin(y / 30) * np.cos(x / 30))
            img[y, x] = np.clip(img[y, x].astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Draw simulated objects (trees, sun)
    # Sun
    cv2.circle(img, (width // 4, height // 4), 50, (0, 200, 255), -1)
    
    # Mountains (triangles)
    pts = np.array([[100, height], [200, height // 2], [300, height]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (100, 150, 50))
    
    pts2 = np.array([[width - 200, height], [width - 100, height // 2.5], [width, height]], dtype=np.int32)
    cv2.fillPoly(img, [pts2], (80, 120, 40))
    
    return img


def create_synthetic_graphic(height: int = 512, width: int = 512) -> np.ndarray:
    """
    Create purely synthetic graphic with:
    - Solid color blocks
    - Sharp edges
    - Geometric patterns
    - Text-like elements
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Solid color blocks (like a UI design)
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
    ]
    
    block_height = height // 3
    block_width = width // 3
    
    idx = 0
    for i in range(3):
        for j in range(3):
            y1 = i * block_height
            y2 = (i + 1) * block_height
            x1 = j * block_width
            x2 = (j + 1) * block_width
            
            color = colors[idx % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
            # Add borders
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
            
            idx += 1
    
    # Add geometric patterns
    # Lines
    cv2.line(img, (0, 0), (width, height), (0, 0, 0), 2)
    cv2.line(img, (width, 0), (0, height), (0, 0, 0), 2)
    
    # Circles
    cv2.circle(img, (width // 2, height // 2), 50, (0, 0, 0), 3)
    
    return img


def create_hybrid_image(height: int = 512, width: int = 512) -> np.ndarray:
    """
    Create hybrid image combining natural and synthetic elements:
    - Left half: natural-like gradient and patterns
    - Right half: synthetic geometric shapes
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Left half: Natural-like
    for y in range(height):
        for x in range(width // 2):
            # Smooth gradient with subtle variation
            b = int(200 - (y / height) * 100 + 20 * np.sin(y / 50))
            g = int(100 + (y / height) * 100)
            r = int(150)
            img[y, x] = [b, g, r]
    
    # Right half: Synthetic geometric
    for y in range(height):
        for x in range(width // 2, width):
            img[y, x] = [200, 200, 200]  # Light gray base
    
    # Add patterns to synthetic side
    for i in range(10):
        cv2.rectangle(
            img,
            (width // 2 + i * 20, i * 20),
            (width // 2 + (i + 1) * 20, (i + 1) * 20),
            (100, 150, 200),
            -1
        )
    
    # Add curves to natural side (simulate water)
    pts = []
    for x in range(width // 2):
        y = int(height // 2 + 50 * np.sin(x / 30))
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], False, (0, 100, 255), 3)
    
    return img


def main():
    """Generate and save test images."""
    output_dir = Path("test_images_generated")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating synthetic test images for compression evaluation...\n")
    
    # Natural image simulation
    print("🏞️  Creating natural image simulation...")
    natural_img = create_natural_image_sim(512, 512)
    natural_path = output_dir / "test_natural_landscape.jpg"
    cv2.imwrite(str(natural_path), natural_img)
    print(f"   ✓ Saved: {natural_path}")
    
    # Synthetic graphic
    print("🎨 Creating synthetic graphic...")
    synthetic_img = create_synthetic_graphic(512, 512)
    synthetic_path = output_dir / "test_synthetic_graphic.jpg"
    cv2.imwrite(str(synthetic_path), synthetic_img)
    print(f"   ✓ Saved: {synthetic_path}")
    
    # Hybrid image
    print("🔀 Creating hybrid image...")
    hybrid_img = create_hybrid_image(512, 512)
    hybrid_path = output_dir / "test_hybrid_mixed.jpg"
    cv2.imwrite(str(hybrid_path), hybrid_img)
    print(f"   ✓ Saved: {hybrid_path}")
    
    print("\n✅ Test images generated successfully!")
    print(f"📁 Location: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
