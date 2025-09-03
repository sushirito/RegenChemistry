#!/usr/bin/env python3
"""
Wrapper script to run fair leave-one-out comparison
Handles import paths correctly
"""

import sys
import os
from pathlib import Path

# Add the geodesic_mps directory to Python path
script_dir = Path(__file__).parent
geodesic_path = script_dir / "geodesic_mps"
sys.path.insert(0, str(geodesic_path))

# Now import and run the comparison
if __name__ == "__main__":
    # Import the main function from compare_methods
    from utils.compare_methods import main
    
    print("🧪 Running FAIR Leave-One-Out Comparison")
    print("This will train 3 separate models, one for each test concentration")
    print("Expected time: ~3-5 minutes")
    print()
    
    # Run the comparison
    main()