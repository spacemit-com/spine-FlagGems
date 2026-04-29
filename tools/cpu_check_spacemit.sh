#!/bin/bash

echo "Running Spacemit runner pre-check (CPU backend)."

if ! command -v lscpu &> /dev/null; then
    echo "Error: lscpu command not found."
    exit 1
fi

if ! command -v python &> /dev/null; then
    echo "Error: python command not found."
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "Error: uv command not found."
    exit 1
fi

echo "CPU information:"
lscpu

echo "Python version:"
python --version

echo "uv version:"
uv --version