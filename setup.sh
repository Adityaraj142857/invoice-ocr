#!/bin/bash
# ============================================================
# setup.sh — Mac / Linux setup script
# Run: bash setup.sh
# ============================================================
set -e

echo "=========================================="
echo " Invoice OCR — Setup (Mac / Linux)"
echo "=========================================="

# Python check
python3 --version >/dev/null 2>&1 || { echo "ERROR: Python 3 not found. Install from python.org"; exit 1; }

# Create venv
echo ""
echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "[2/5] Upgrading pip..."
pip install --upgrade pip --quiet

# Install PyTorch (correct for platform)
echo "[3/5] Installing PyTorch..."
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "      → Apple Silicon detected — installing MPS-enabled torch"
    pip install torch torchvision --quiet
else
    echo "      → Intel/AMD detected"
    pip install torch torchvision --quiet
fi

# Install all requirements
echo "[4/5] Installing requirements..."
pip install -r requirements.txt --quiet

# Poppler for PDF support
echo "[5/5] Checking poppler (PDF support)..."
if command -v pdfinfo >/dev/null 2>&1; then
    echo "      → poppler already installed"
else
    if command -v brew >/dev/null 2>&1; then
        echo "      → Installing via Homebrew..."
        brew install poppler
    else
        echo "      ⚠ poppler not found. Install with: brew install poppler"
        echo "        (PDF input will not work without it)"
    fi
fi

echo ""
echo "=========================================="
echo " Setup complete!"
echo " To run:"
echo "   source venv/bin/activate"
echo "   python executable.py --input data/train/"
echo "=========================================="
