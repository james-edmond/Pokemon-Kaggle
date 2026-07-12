#!/usr/bin/env bash
# Rented-box setup for phase-5 expert iteration (Linux x86, Ubuntu-ish).
# Usage: bash scripts/cloud_setup.sh [cpu|cu121]
set -euo pipefail
FLAVOR="${1:-cpu}"
# python3.11 is apt-available on Ubuntu 22.04; on 24.04 add the deadsnakes PPA
# or swap python3.11 -> the system python3.12 (torch 2.5.1 has 3.12 wheels).
sudo apt-get update -y && sudo apt-get install -y python3.11 python3.11-venv rsync
python3.11 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
if [ "$FLAVOR" = "cu121" ]; then
  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
else
  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
fi
pip install numpy pytest
export PYTHONIOENCODING=utf-8
# engine: the repo's default PTCG_ENGINE_DIR resolution works from repo root
python -c "import sys; sys.path.insert(0,'.'); from ptcg.cards import build_tables; t=build_tables(); print('engine ok, rows', t.n_rows)"
echo "setup ok ($FLAVOR)"
