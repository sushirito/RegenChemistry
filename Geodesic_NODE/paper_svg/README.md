# Paper SVG Visualizations - Geodesic Spectral Interpolation

## Overview
This folder contains publication-ready SVG visualizations for the Geodesic Spectral Interpolation paper, demonstrating why geometry matters for arsenic detection and how the GEOSPEC algorithm works.

## Quick Start

### Local Development
1. Open `index.html` in any modern web browser:
   ```bash
   open paper_svg/index.html
   ```
   Or simply double-click the `index.html` file.

2. The visualizations will load automatically using your actual arsenic spectral data.

### Using Python Server (Optional)
If you prefer to run a local server:
```bash
cd paper_svg
python3 -m http.server 8000
# Then open http://localhost:8000
```

## Features

### Figure 1: Problem Motivation
- **3D Riemannian Manifold**: Shows concentration-wavelength space as a curved surface
- **Path Comparison**: Euclidean (red, straight) vs Geodesic (green, curved) paths
- **Real Arsenic Spectra**: Displays actual data at 4 anchor points (0, 20, 40, 60 ppb)
- **Great Circle Analogy**: Earth visualization showing the same principle

### Figure 2: GEOSPEC Algorithm (4 Panels)
1. **Geodesic Distance**: Metric tensor heatmap with Christoffel symbols and Hamiltonian contours
2. **Neural Architecture**: Metric and Flow networks visualization
3. **Computational Efficiency**: 
   - Memory: O(BT) → O(B) via adjoint method
   - Time: O(B·T·C_NN) → O(B·T) via Christoffel caching
4. **Shooting & Results**: Artillery metaphor and spectral reconstruction using real data

## Export Options
- Click the "Export SVG" button under each figure to download publication-ready vector graphics
- SVGs can be imported directly into Adobe Illustrator, Inkscape, or LaTeX documents

## File Structure
```
paper_svg/
├── index.html              # Main visualization page
├── css/
│   └── styles.css         # Scientific styling
├── js/
│   ├── svg1_motivation.js # Problem visualization
│   ├── svg2_algorithm.js  # Algorithm panels
│   ├── data_loader.js     # CSV parsing
│   └── geometry_utils.js  # Mathematical helpers
├── data/
│   └── arsenic_data.csv  # Real spectral measurements
└── README.md              # This file
```

## Browser Compatibility
- Chrome/Edge: Fully supported
- Firefox: Fully supported
- Safari: Fully supported
- Internet Explorer: Not supported

## Customization
- Colors can be adjusted in `css/styles.css` (CSS variables)
- Panel dimensions can be modified in the respective JS files
- Data source can be changed in `data_loader.js`

## Troubleshooting
- If data doesn't load, ensure you're running from the correct directory
- For CORS issues, use a local server instead of file:// protocol
- Check browser console for any error messages

## Citation
If you use these visualizations, please cite the Geodesic Spectral Interpolation paper.

---
Created for the Geodesic Neural ODE project