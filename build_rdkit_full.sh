#!/bin/bash
# build_rdkit_full.sh - Full build script with ccache

set -e  # Exit on error

echo "=== RDKit Full Build Script ==="
echo "Building all components with ccache support"
echo ""

# Setup conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rdkit

# Clean PATH
unset PYTHONPATH
export PATH=/home/swarnavas/miniconda3/envs/rdkit/bin:$PATH

# Environment
export RDBASE=/home/swarnavas/Work/PhD_Work/Covaln_Dev_work/rdkit
export PYTHONPATH=$RDBASE:$PYTHONPATH
export LD_LIBRARY_PATH=$RDBASE/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install ccache
if ! command -v ccache &> /dev/null; then
    echo "Installing ccache..."
    conda install -c conda-forge ccache -y
fi

# Setup ccache
export CC="ccache gcc"
export CXX="ccache g++"
mkdir -p $RDBASE/.ccache
export CCACHE_DIR=$RDBASE/.ccache
export CCACHE_MAXSIZE=5G

echo "Environment configured:"
echo "  RDBASE: $RDBASE"
echo "  CCACHE_DIR: $CCACHE_DIR"
echo "  Python: $(which python)"
echo ""

# Build
cd $RDBASE
mkdir -p build
cd build

echo "=== Configuring CMake ==="
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

echo ""
echo "=== Building Everything (this may take 30-60 minutes) ==="
echo "First build: ~30-60 min | With ccache: ~5-10 min"
echo ""

# Build everything
make -j$(nproc) 2>&1

echo ""
echo "=== Installing ==="
make install 2>&1 || true

# Manual copy in case install fails
echo "Copying files..."
mkdir -p $RDBASE/lib
cp $RDBASE/build/lib/*.so* $RDBASE/lib/ 2>/dev/null || true
find $RDBASE/build -name "*.so" -path "*/rdkit/*" -exec cp {} $RDBASE/rdkit/ \; 2>/dev/null || true
find $RDBASE/build -name "*.so" -path "*/rdkit/Chem/*" -exec cp {} $RDBASE/rdkit/Chem/ \; 2>/dev/null || true
find $RDBASE/build -name "*.so" -path "*/rdkit/DataStructs/*" -exec cp {} $RDBASE/rdkit/DataStructs/ \; 2>/dev/null || true
find $RDBASE/build -name "*.so" -path "*/rdkit/Geometry/*" -exec cp {} $RDBASE/rdkit/Geometry/ \; 2>/dev/null || true

echo ""
echo "=== Testing ==="
cd $RDBASE
python << 'PYTEST'
import sys
try:
    from rdkit import rdBase
    print(f"RDKit version: {rdBase.rdkitVersion}")
    
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD
    
    names = rdMD.GetBatchDescriptorNames()
    print(f"Batch descriptors: {len(names)}")
    assert len(names) == 45
    
    mols = [Chem.MolFromSmiles(s) for s in ['CCO', 'c1ccccc1']]
    results = rdMD.CalcDescriptorsBatch(mols, "all")
    print(f"Batch result shape: {results.shape}")
    assert results.shape == (2, 45)
    
    print("\n✅ BUILD SUCCESSFUL!")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTEST

echo ""
echo "=== Cache Stats ==="
ccache -s

echo ""
echo "Run tests: cd $RDBASE/build && ctest -R pyBatchDescriptors --output-on-failure"
