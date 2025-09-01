# Project Reorganization Plan for Geodesic_NODE

## Current State
The project has 37 files all in the root directory, making it difficult to navigate and understand the structure.

## Proposed Folder Structure

```
Geodesic_NODE/
│
├── data/
│   └── 0.30MB_AuNP_As.csv
│
├── src/
│   ├── models/
│   │   ├── __init__.py (new - for imports)
│   │   ├── metric_network.py
│   │   ├── decoder_network.py
│   │   └── geodesic_model.py
│   │
│   ├── core/
│   │   ├── __init__.py (new)
│   │   ├── christoffel.py
│   │   ├── geodesic_ode.py
│   │   └── shooting_solver.py
│   │
│   ├── data/
│   │   ├── __init__.py (new)
│   │   └── data_loader.py
│   │
│   ├── training/
│   │   ├── __init__.py (new)
│   │   ├── train.py
│   │   ├── train_fast.py
│   │   └── validate.py
│   │
│   └── analysis/
│       ├── __init__.py (new)
│       ├── spectral_validation_metrics.py
│       └── analyze_metric_variability.py
│
├── visualization/
│   ├── spectral_3d_visualization.py
│   ├── spectral_holdout_validation.py
│   ├── visualize_geodesic_validation.py
│   ├── quick_geodesic_plot.py
│   └── geodesic_visualization_server.py
│
├── demos/
│   └── demo_training.py
│
├── outputs/
│   ├── plots/
│   │   ├── baseline_failure_analysis.html
│   │   ├── spectral_holdout_validation.html
│   │   ├── spectral_manifold.html
│   │   ├── spectral_metrics_analysis.html
│   │   └── geodesic_quick_demo.html
│   │
│   └── results/
│       ├── spectral_validation_metrics.csv
│       └── metric_variability_analysis.csv
│
├── checkpoints/
│   └── (keep existing checkpoint files)
│
├── checkpoints_fast/
│   └── (keep existing checkpoint files)
│
├── docs/
│   ├── CLAUDE.md
│   ├── research_brainstorming_agent.md
│   └── context/
│       ├── manifold-ode-geodesic.md
│       ├── manifold-ode-implementation.md
│       └── manifold-ode.md
│
├── web/
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       └── js/
│
├── requirements.txt (new - list dependencies)
├── README.md (new - project overview)
└── .gitignore (new - ignore __pycache__, checkpoints, etc.)
```

## Import Path Updates Required

After reorganization, imports will need to be updated:

### 1. In training scripts (train.py, validate.py, train_fast.py):
```python
# Old imports
from data_loader import SpectralDataset, create_data_loaders
from geodesic_model import GeodesicSpectralModel
from metric_network import MetricNetwork

# New imports
from src.data.data_loader import SpectralDataset, create_data_loaders
from src.models.geodesic_model import GeodesicSpectralModel
from src.models.metric_network import MetricNetwork
```

### 2. In model files (geodesic_model.py):
```python
# Old imports
from metric_network import MetricNetwork
from christoffel import ChristoffelComputer
from geodesic_ode import GeodesicODE
from shooting_solver import ShootingSolver
from decoder_network import AbsorbanceDecoder

# New imports
from src.models.metric_network import MetricNetwork
from src.core.christoffel import ChristoffelComputer
from src.core.geodesic_ode import GeodesicODE
from src.core.shooting_solver import ShootingSolver
from src.models.decoder_network import AbsorbanceDecoder
```

### 3. In core files (christoffel.py, geodesic_ode.py, shooting_solver.py):
```python
# Old imports
from metric_network import MetricNetwork

# New imports
from src.models.metric_network import MetricNetwork
```

### 4. In visualization scripts:
```python
# Old imports
from data_loader import SpectralDataset
from geodesic_model import GeodesicSpectralModel

# New imports
from src.data.data_loader import SpectralDataset
from src.models.geodesic_model import GeodesicSpectralModel
```

### 5. Add to Python path in scripts that need to run independently:
```python
import sys
import os
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

## Additional Files to Create

### 1. requirements.txt
```
torch>=2.0.0
numpy
pandas
scipy
scikit-learn
plotly
tqdm
matplotlib
flask  # if using web server
```

### 2. README.md
Should include:
- Project overview
- Installation instructions
- Usage examples
- Model architecture description
- Results summary

### 3. __init__.py files
Create empty `__init__.py` files in each package directory to make them Python packages:
- src/__init__.py
- src/models/__init__.py
- src/core/__init__.py
- src/data/__init__.py
- src/training/__init__.py
- src/analysis/__init__.py

### 4. .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Model checkpoints
checkpoints/
checkpoints_fast/
*.pth
*.pt

# Outputs
*.log
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
```

## Benefits of This Organization

1. **Clear separation of concerns**: Models, core algorithms, data handling, training, and visualization are separated
2. **Easier navigation**: Related files are grouped together  
3. **Better import structure**: Clear hierarchy makes imports more intuitive
4. **Output management**: All generated files go to outputs/ folder
5. **Documentation centralized**: All docs in one place
6. **Web components isolated**: Flask/web related files separated
7. **Reusability**: src/ package can be imported as a module
8. **Professional structure**: Follows Python best practices

## Migration Steps

1. **Create directory structure**:
   ```bash
   mkdir -p src/{models,core,data,training,analysis}
   mkdir -p visualization demos outputs/{plots,results}
   mkdir -p docs/context web
   ```

2. **Move files to appropriate locations**:
   - Move all .py model files to src/models/
   - Move core algorithm files to src/core/
   - Move data_loader.py to src/data/
   - Move training scripts to src/training/
   - Move analysis scripts to src/analysis/
   - Move visualization scripts to visualization/
   - Move HTML outputs to outputs/plots/
   - Move CSV outputs to outputs/results/
   - Move documentation to docs/
   - Move web files to web/

3. **Create __init__.py files** in all package directories

4. **Update all import statements** in every Python file

5. **Test all scripts** to ensure they still run correctly:
   ```bash
   python demos/demo_training.py
   python src/training/train.py
   python visualization/quick_geodesic_plot.py
   ```

6. **Create requirements.txt and README.md**

7. **Clean up root directory** - should only contain:
   - src/
   - visualization/
   - demos/
   - outputs/
   - checkpoints*/
   - docs/
   - web/
   - data/
   - requirements.txt
   - README.md
   - .gitignore

## Notes on Import Management

To avoid import issues after reorganization:

1. **Use absolute imports** from the project root
2. **Add project root to PYTHONPATH** when running scripts:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/Users/aditya/CodingProjects/Geodesic_NODE"
   ```
3. **Or use relative imports** within packages:
   ```python
   from .metric_network import MetricNetwork  # within same package
   from ..core.christoffel import ChristoffelComputer  # from sibling package
   ```

## Testing After Refactor

Create a simple test script to verify all imports work:

```python
# test_imports.py
try:
    from src.models.geodesic_model import GeodesicSpectralModel
    from src.data.data_loader import SpectralDataset
    from src.training.train import Trainer
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import error: {e}")
```