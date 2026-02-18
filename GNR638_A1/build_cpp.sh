#!/bin/bash
# build_cpp.sh - Build C++ extensions for optimized performance

set -e  # Exit on error

echo "=========================================="
echo "Building C++ Extensions"
echo "=========================================="

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo "❌ ERROR: CMake is not installed!"
    echo ""
    echo "Please install CMake:"
    echo "  Ubuntu/Debian: sudo apt-get install cmake"
    echo "  macOS: brew install cmake"
    echo "  Windows: Download from https://cmake.org/download/"
    echo ""
    exit 1
fi

# Check if pybind11 is installed
echo ""
echo "Checking for pybind11..."
python3 -c "import pybind11" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing pybind11..."
    pip install pybind11
fi

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p build
cd build

# Run CMake
echo ""
echo "Running CMake..."
cmake ../cpp

# Build
echo ""
echo "Building..."
cmake --build . --config Release

# Copy the built library to the main directory
echo ""
echo "Installing extension..."
if [ -f "conv_ops*.so" ]; then
    cp conv_ops*.so ..
    echo "✅ C++ extension built successfully!"
    echo "   File: conv_ops.so"
elif [ -f "conv_ops*.pyd" ]; then
    cp conv_ops*.pyd ..
    echo "✅ C++ extension built successfully!"
    echo "   File: conv_ops.pyd"
else
    echo "⚠️  Warning: Could not find built extension"
    ls -la
fi

cd ..

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Test the extension:"
echo "  python3 -c 'import conv_ops; print(\"✅ C++ backend loaded!\")'"
echo ""
