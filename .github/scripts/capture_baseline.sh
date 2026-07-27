#!/usr/bin/env bash
set -euo pipefail

echo "Starting baseline capture protocol..."

# 1. Thermal Soak (60s)
echo "Performing 60s thermal soak..."
START=$(date +%s)
while [ $(($(date +%s) - START)) -lt 60 ]; do
    cargo run -p metaltile-cli -- bench --match-group ffai --quiet
done

# 2. Capture measurements
echo "Capturing median-of-5 measurements..."
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
MACHINE=$(sysctl -n hw.model)

cargo run -p metaltile-cli -- capture-baseline \
    --output "baselines/baseline-${MACHINE}.json" \
    --timestamp "$TIMESTAMP" \
    --machine "$MACHINE" \
    --iterations 5