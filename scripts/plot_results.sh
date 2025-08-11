#!/bin/bash
set -euo pipefail

RESULTS_DIR=${1:-results}
shift || true  # shift only if there was a first arg

python plot_results.py --results-dir "$RESULTS_DIR" "$@"
