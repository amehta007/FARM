#!/bin/bash

# Demo script for Linux/macOS

echo "===================================="
echo "Worker Detection System - Demo"
echo "===================================="
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download models
echo "Downloading and converting models..."
python -m src.models.download_models

# Run demo
echo "Running demo..."
python -m src.main demo

echo
echo "Demo complete! Check data/outputs/ for results."
echo
echo "To view the dashboard, run:"
echo "  streamlit run src/app.py"
echo

