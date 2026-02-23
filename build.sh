#!/usr/bin/env bash
# Build script for Render deployment (runs inside /backend)
set -e

pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
"
