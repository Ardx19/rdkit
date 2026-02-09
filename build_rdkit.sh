#!/bin/bash
# build_rdkit.sh - Build script for RDKit batch descriptors with ccache support

set -e  # Exit on error

echo "=== RDKit Batch Descriptors Build Script ==="
echo "This will build Phase 1 (45 C++ batch descriptors)"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Setup conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rdkit

# Clean PATH to avoid conflicts
unset PYTHONPATH
export PATH=/home/swarnavas/miniconda3/envs/rdkit/bin:$PATH

# Set environment variables
export RDBASE=/home/swarnavas/Work/PhD_Work/Covaln_Dev_work/rdkit
export PYTHONPATH=$RDBASE:$PYTHONPATH
export LD_LIBRARY_PATH=$RDBASE/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install ccache if not present
if ! command -v ccache &> /dev/null; then
    echo "Installing ccache for faster rebuilds..."
    conda install -c conda-forge ccache -y
fi

# Setup ccache
export CC="ccache gcc"
export CXX="ccache g++"
mkdir -p $RDBASE/.ccache
export CCACHE_DIR=$RDBASE/.ccache
export CCACHE_MAXSIZE=5G

echo ""
echo "Environment:"
echo "  RDBASE: $RDBASE"
echo "  CONDA_PREFIX: $CONDA_PREFIX"
echo "  CCACHE_DIR: $CCACHE_DIR"
echo "  Python: $(which python)"
echo ""

# Create build directory
cd $RDBASE
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir -p build
fi

cd build

echo ""
echo "=== Step 1: Configure with CMake ==="
if [ ! -f "CMakeCache.txt" ]; then
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DRDK_INSTALL_INTREE=ON \
        -DRDK_BUILD_PYTHON_WRAPPERS=ON \
        -DRDK_BUILD_CPP_TESTS=ON \
        -DRDK_BUILD_OPENMP=ON \
        -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python \
        -DBoost_DIR=/home/swarnavas/miniconda3/envs/rdkit/lib/cmake/Boost-1.82.0 \
        -DBOOST_ROOT=/home/swarnavas/miniconda3/envs/rdkit \
        -DCMAKE_PREFIX_PATH=/home/swarnavas/miniconda3/envs/rdkit \
        -DEIGEN3_INCLUDE_DIR=$CONDA_PREFIX/include/eigen3 \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DRDK_BUILD_DESCRIPTORS3D=OFF \
        -DRDK_BUILD_MAEPARSER_SUPPORT=OFF \
        -DRDK_BUILD_COORDGEN_SUPPORT=OFF \
        -DRDK_BUILD_THREADSAFE_SSS=OFF \
        -DRDK_TEST_MULTITHREADED=OFF \
        -DRDK_BUILD_CHEMDRAW_SUPPORT=OFF \
        -DRDK_BUILD_AVALON_SUPPORT=OFF \
        -DRDK_BUILD_INCHI_SUPPORT=OFF \
        -DRDK_BUILD_FREESASA_SUPPORT=OFF \
        -DRDK_BUILD_YAEHMOP_SUPPORT=OFF \
        -DRDK_BUILD_XYZ2MOL_SUPPORT=OFF \
        -DRDK_BUILD_CAIRO_SUPPORT=OFF \
        -DRDK_BUILD_SWIG_WRAPPERS=OFF \
        -DRDK_BUILD_JAVA_WRAPPERS=OFF \
        -DRDK_BUILD_COMIC_FONT=OFF \
        -DRDK_BUILD_MOLDRAW2D=OFF
    echo -e "${GREEN}✓ CMake configuration complete${NC}"
else
    echo "CMake already configured (using existing cache)"
    echo "To reconfigure, delete build/CMakeCache.txt"
fi

echo ""
echo "=== Step 2: Build Core Libraries ==="
echo "This may take 10-20 minutes on first build..."
echo "(Subsequent builds will be faster with ccache)"
echo ""

# Build order: core libs first, then Python modules
echo "Building RDGeneral..."
make -j$(nproc) RDGeneral

echo "Building RDBoost..."
make -j$(nproc) RDBoost

echo "Building DataStructs..."
make -j$(nproc) DataStructs

echo "Building RDGeometryLib..."
make -j$(nproc) RDGeometryLib

echo "Building GraphMol..."
make -j$(nproc) GraphMol

echo "Building SmilesParse..."
make -j$(nproc) SmilesParse

echo "Building FileParsers..."
make -j$(nproc) FileParsers

echo "Building Descriptors..."
make -j$(nproc) Descriptors

echo "Building Fingerprints..."
make -j$(nproc) Fingerprints

echo ""
echo "=== Step 3: Build Python Modules ==="

PYTHON_MODULES="
rdBase
cDataStructs
rdGeometry
rdchem
rdmolops
rdmolfiles
rdMolDescriptors
"

for mod in $PYTHON_MODULES; do
    echo "Building $mod..."
    make -j$(nproc) $mod
done

echo ""
echo "=== Step 4: Install ==="
echo "Copying libraries and Python modules..."

# Copy libraries
mkdir -p $RDBASE/lib
cp $RDBASE/build/lib/*.so* $RDBASE/lib/ 2>/dev/null || true

# Copy Python modules
find $RDBASE/build -name "*.so" -path "*/rdkit/*" -exec cp {} $RDBASE/rdkit/ \; 2>/dev/null || true
find $RDBASE/build -name "*.so" -path "*/rdkit/Chem/*" -exec cp {} $RDBASE/rdkit/Chem/ \; 2>/dev/null || true
find $RDBASE/build -name "*.so" -path "*/rdkit/DataStructs/*" -exec cp {} $RDBASE/rdkit/DataStructs/ \; 2>/dev/null || true
find $RDBASE/build -name "*.so" -path "*/rdkit/Geometry/*" -exec cp {} $RDBASE/rdkit/Geometry/ \; 2>/dev/null || true

echo -e "${GREEN}✓ Installation complete${NC}"

echo ""
echo "=== Step 5: Test ==="
echo "Running basic tests..."

cd $RDBASE
python << 'PYTEST'
import sys
print("Testing RDKit batch descriptors...")

try:
    from rdkit import rdBase
    print(f"✓ RDKit version: {rdBase.rdkitVersion}")
    
    from rdkit import Chem
    print("✓ Chem module imported")
    
    from rdkit.Chem import rdMolDescriptors as rdMD
    print("✓ rdMolDescriptors imported")
    
    # Test descriptor count
    names = rdMD.GetBatchDescriptorNames()
    print(f"✓ {len(names)} batch descriptors available")
    assert len(names) == 45, f"Expected 45, got {len(names)}"
    
    # Test molecules
    mols = [Chem.MolFromSmiles(s) for s in ['CCO', 'c1ccccc1', 'CC(=O)O']]
    print(f"✓ Created {len(mols)} test molecules")
    
    # Test batch calculation
    results = rdMD.CalcDescriptorsBatch(mols, "all")
    print(f"✓ Batch calculation works: shape {results.shape}")
    assert results.shape == (3, 45), f"Shape mismatch: {results.shape}"
    
    # Test individual functions
    chi0v = rdMD.CalcChi0v(mols)
    kappa1 = rdMD.CalcKappa1(mols)
    print(f"✓ CalcChi0v: {chi0v}")
    print(f"✓ CalcKappa1: {kappa1}")
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("="*50)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTEST

echo ""
echo "=== Build Summary ==="
echo -e "${GREEN}✓ Phase 1 build complete${NC}"
echo ""
echo "Cache statistics:"
ccache -s

echo ""
echo "To run full tests:"
echo "  cd $RDBASE/build && ctest -R pyBatchDescriptors --output-on-failure"
echo ""
echo "To rebuild after code changes:"
echo "  cd $RDBASE/build && make -j\$(nproc) Descriptors rdMolDescriptors"
