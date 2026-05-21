#!/usr/bin/env bash
set -e

# Clone DGB if not already present
if [ ! -d "DGB/.git" ]; then
    git clone https://github.com/fpour/DGB.git DGB
fi

# Apply Python 3.12 compatibility patch (random.sample no longer accepts sets)
patch --forward --silent DGB/tgn/utils/data_processing.py patches/dgb_python312_compat.patch || true

echo "DGB setup complete."
