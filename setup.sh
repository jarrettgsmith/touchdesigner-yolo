#!/bin/bash
set -e

echo "========================================"
echo "YOLO Server Setup"
echo "========================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

echo "[1/3] Creating virtual environment..."
python3 -m venv venv_yolo

echo "[2/3] Activating virtual environment..."
source venv_yolo/bin/activate

echo "[3/3] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To run the YOLO server:"
echo "  ./run.sh"
echo ""
