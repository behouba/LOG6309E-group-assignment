#!/bin/bash

set -e
cd "$(dirname "$0")"

mkdir -p logs results

rm -f logs/*.log 2>/dev/null || true
rm -rf results/* 2>/dev/null || true

echo "Replication outputs cleared."
