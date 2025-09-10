#!/bin/bash

# Repository Cleanup Script - Keeps only A100, M1 implementations and essential files
# Run with: bash cleanup_repo.sh

echo "Starting repository cleanup..."
echo "This will keep only A100, M1 implementations and essential documentation."
echo ""

# Create backup directory name with timestamp
BACKUP_DIR="../Geodesic_NODE_backup_$(date +%Y%m%d_%H%M%S)"

echo "Creating backup at: $BACKUP_DIR"
cp -R . "$BACKUP_DIR"
echo "Backup created successfully."
echo ""

# List of directories to DELETE
DIRS_TO_DELETE=(
    "geodesic_mps"      # Old MPS implementation (replaced by geodesic_m1)
    "src"               # Old implementation
    "checkpoints"       # Old checkpoints
    "checkpoints_fast"  # Old checkpoints
    "demos"             # Demo files
    "logs"              # Old logs
    "outputs"           # Old outputs
    "paper_svg"         # Paper visualizations
    "visualization"     # Old visualizations
    "web"               # Web interface
    "__pycache__"       # Python cache
)

# List of root files to DELETE
FILES_TO_DELETE=(
    "A100-implementation.md"
    "create_fancy_3d_plots.py"
    "create_geodesic_comparison.py"
    "debug_device_manager.py"
    "debug_training_fixed.py"
    "debug_training_m3.py"
    "fair_comparison_demo.html"
    "generate_3d_surfaces.py"
    "generate_test_data.py"
    "geodesic_a100_colab_old.ipynb"  # Keep the newer one
    "geodesic_colab.ipynb"
    "geodesic_validation_demo.html"
    "notebook_modification_instructions.md"
    "REFACTORING_COMPLETE.md"
    "regeneron_sts_prompt.md"
    "run_fair_comparison.py"
    "run_geodesic_comparison.py"
    "run_mvp.py"
)

# Delete test files
echo "Removing test files..."
find . -name "test_*.py" -type f -delete 2>/dev/null

# Delete HTML files (except any you might want to keep)
echo "Removing HTML files..."
find . -name "*.html" -type f -delete 2>/dev/null

# Delete directories
echo "Removing deprecated directories..."
for dir in "${DIRS_TO_DELETE[@]}"; do
    if [ -d "$dir" ]; then
        echo "  Deleting $dir/"
        rm -rf "$dir"
    fi
done

# Delete files
echo "Removing deprecated files..."
for file in "${FILES_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        echo "  Deleting $file"
        rm -f "$file"
    fi
done

# Clean up Python cache files throughout the project
echo "Cleaning Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type f -name ".DS_Store" -delete 2>/dev/null

echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "Preserved directories:"
echo "  - geodesic_a100/          (A100 implementation)"
echo "  - geodesic_m1/            (M1 implementation)"
echo "  - data/                   (Spectral datasets)"
echo "  - docs/                   (Documentation including research_brainstorming_agent.md)"
echo "  - explanations/           (Explanation documents)"
echo "  - .claude/                (Claude configuration)"
echo ""
echo "Preserved files:"
echo "  - geodesic_a100_colab.ipynb (Colab notebook)"
echo "  - CLAUDE.md               (Project instructions)"
echo "  - README.md               (Project overview)"
echo "  - requirements.txt        (Dependencies)"
echo "  - .gitignore             (Git configuration)"
echo ""
echo "Backup saved at: $BACKUP_DIR"
echo ""
echo "To restore if needed: cp -R $BACKUP_DIR/* ."