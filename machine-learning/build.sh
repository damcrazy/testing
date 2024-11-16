#!/bin/bash
set -e

# Clean previous builds
make clean

# Build for simulation mode
SGX_MODE=SW make

# Set permissions
chmod +x bin/app 