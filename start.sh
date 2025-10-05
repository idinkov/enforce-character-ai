#!/usr/bin/env bash
set -e

echo "Checking for Python 3.10..."

# Try to locate python3.10
if ! command -v python3.10 &> /dev/null; then
  echo "Python 3.10 not found."

  # Determine OS type
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing Python 3.10 via apt (requires sudo)..."
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv python3.10-distutils
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing Python 3.10 via Homebrew..."
    if ! command -v brew &> /dev/null; then
      echo "Homebrew not found. Please install Homebrew first: https://brew.sh/"
      exit 1
    fi
    brew install python@3.10
  else
    echo "Unsupported OS. Please install Python 3.10 manually."
    exit 1
  fi
fi

# Step 2: Create venv
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3.10 -m venv .venv
fi

# Step 3: Activate and install
echo "Activating virtual environment and installing dependencies..."
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Step 4: Launch the app
echo "Launching app..."
python main.py
