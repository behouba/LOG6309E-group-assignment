#!/bin/bash

set -e
cd "$(dirname "$0")"

mkdir -p logs results models

rm -f logs/*.log 2>/dev/null || true
rm -rf results/* models/* 2>/dev/null || true

echo "Extension outputs cleared."
