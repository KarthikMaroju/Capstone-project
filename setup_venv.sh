#!/usr/bin/env bash
# One-time environment setup for the whole project (all 3 modules).
#
# Usage:
#   chmod +x setup_venv.sh
#   ./setup_venv.sh
#   source venv/bin/activate      # (Windows: venv\Scripts\activate)
set -e

python3 -m venv venv

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  source venv/Scripts/activate
fi

pip install --upgrade pip

# Install CPU-only torch FIRST. Plain `pip install torch` on Linux pulls the
# full CUDA build (3+ GB of nvidia-* wheels) even on machines with no GPU,
# which can blow past disk quotas unnecessarily. sentence-transformers only
# needs torch to run the small MiniLM embedding model on CPU, so we pin the
# lightweight CPU wheel first; pip then sees torch is already satisfied and
# skips the CUDA build when it processes the rest of requirements.txt.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

echo ""
echo "Done. Activate the venv in new shells with:"
echo "  source venv/bin/activate   (Windows: venv\\Scripts\\activate)"
