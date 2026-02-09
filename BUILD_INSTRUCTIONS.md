# Local Build Guide with Conda

This guide sets up a local Conda environment for building and testing RDKit batch descriptors.

## Prerequisites

```bash
# Ensure conda is available
which conda

# If not, install miniconda from https://docs.conda.io/en/latest/miniconda.html
```

## Step 1: Create Conda Environment

```bash
# Create environment with Python 3.10 (stable for RDKit)
conda create -n rdkit python=3.10 -y

# Activate environment
conda activate rdkit

# Verify
which python
python --version
```

## Step 2: Install Build Dependencies

```bash
# Core build tools
conda install -c conda-forge cmake=3.26 boost=1.82 eigen -y

# Python dependencies
conda install -c conda-forge numpy -y

# Compilers (Linux)
conda install -c conda-forge gcc_linux-64 gxx_linux-64 gfortran_linux-64 -y

# Or macOS:
# conda install -c conda-forge clang_osx-64 clangxx_osx-64 -y

# Additional dependencies
conda install -c conda-forge pkg-config make -y
```

## Step 3: Set Environment Variables

Create a setup script `setup_env.sh`:

```bash
#!/bin/bash
# setup_env.sh - Source this file: source setup_env.sh

export RDBASE=$(pwd)
export PYTHONPATH=$RDBASE:$PYTHONPATH

# Linux
export LD_LIBRARY_PATH=$RDBASE/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# macOS
# export DYLD_FALLBACK_LIBRARY_PATH=$RDBASE/lib:$CONDA_PREFIX/lib:$DYLD_FALLBACK_LIBRARY_PATH

# Compiler flags
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++

# Boost
export BOOST_ROOT=$CONDA_PREFIX

echo "Environment configured:"
echo "  RDBASE: $RDBASE"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  CC: $CC"
echo "  CXX: $CXX"
```

Activate the environment:
```bash
source setup_env.sh
```

## Step 4: Configure Build

```bash
# Clean previous builds
rm -rf build
mkdir -p build && cd build

# Configure with minimal options for faster builds
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DRDK_INSTALL_INTREE=ON \
  -DRDK_BUILD_PYTHON_WRAPPERS=ON \
  -DRDK_BUILD_CPP_TESTS=ON \
  -DRDK_BUILD_OPENMP=ON \
  -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python \
  -DBOOST_ROOT=$CONDA_PREFIX \
  -DEIGEN3_INCLUDE_DIR=$CONDA_PREFIX/include/eigen3 \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
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
```

## Step 5: Build

```bash
# Build only what we need (much faster than full build)
make -j$(nproc) Descriptors 2>&1 | tee build_descriptors.log
make -j$(nproc) rdMolDescriptors 2>&1 | tee build_python.log
```

Expected time: 5-15 minutes depending on CPU cores

## Step 6: Install

```bash
# Install Python wrappers
make install 2>&1 | tee install.log
```

## Step 7: Verify Installation

```bash
# Test Python import
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
    assert len(names) == 45, f"Expected 45, got {len(names)}"
    print(f"✓ {len(names)} batch descriptors available")
    
    # Test batch calculation
    mols = [Chem.MolFromSmiles(s) for s in ['CCO', 'c1ccccc1', 'CC(=O)O']]
    results = rdMD.CalcDescriptorsBatch(mols, "all")
    assert results.shape == (3, 45), f"Shape mismatch: {results.shape}"
    print(f"✓ Batch calculation works: shape {results.shape}")
    
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
```

## Step 8: Run CTest

```bash
cd $RDBASE/build

# Run batch descriptor tests
ctest -R pyBatchDescriptors --output-on-failure

# Run all descriptor tests
ctest -R Descriptor --output-on-failure

# Verbose
ctest -R pyBatchDescriptors -V
```

## Quick Rebuild After Changes

```bash
# If you modify C++ code
cd $RDBASE/build
make -j$(nproc) Descriptors rdMolDescriptors
make install

# If you modify Python code only
# No rebuild needed - just edit and test
```

## Performance Benchmark

```bash
cd $RDBASE
python << 'PYBENCH'
import time
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

# Generate test molecules
smiles = ['C' * i for i in range(1, 101)]
mols = [Chem.MolFromSmiles(s) for s in smiles]

print(f"Benchmarking with {len(mols)} molecules")
print("="*50)

# Benchmark
start = time.time()
results = rdMD.CalcDescriptorsBatch(mols, "all")
batch_time = time.time() - start

print(f"CalcDescriptorsBatch('all'): {batch_time:.2f}s")
print(f"Throughput: {len(mols)/batch_time:.1f} mol/s")
print(f"Shape: {results.shape}")
print(f"\n✅ Benchmark complete!")
PYBENCH
```

## Troubleshooting

### ImportError: No module named 'rdkit'
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=$RDBASE:$PYTHONPATH

# Verify install worked
ls -la $RDBASE/rdkit/Chem/rdMolDescriptors*.so
```

### Library not found
```bash
# Linux
export LD_LIBRARY_PATH=$RDBASE/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# macOS
export DYLD_FALLBACK_LIBRARY_PATH=$RDBASE/lib:$CONDA_PREFIX/lib:$DYLD_FALLBACK_LIBRARY_PATH
```

### CMake can't find Boost
```bash
# Ensure BOOST_ROOT is set
export BOOST_ROOT=$CONDA_PREFIX

# Reconfigure
cmake .. -DBOOST_ROOT=$CONDA_PREFIX
```

### Out of memory
```bash
# Reduce parallel jobs
make -j2 Descriptors

# Add swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Full Setup Script

Save as `setup_and_build.sh`:

```bash
#!/bin/bash
set -e

echo "=== Setting up RDKit build environment ==="

# Create and activate conda environment
if ! conda env list | grep -q "^rdkit"; then
    echo "Creating conda environment..."
    conda create -n rdkit python=3.10 -y
fi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate rdkit

# Install dependencies
echo "Installing dependencies..."
conda install -c conda-forge cmake=3.26 boost=1.82 eigen numpy -y
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y

# Set environment
export RDBASE=$(pwd)
export PYTHONPATH=$RDBASE:$PYTHONPATH
export LD_LIBRARY_PATH=$RDBASE/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export BOOST_ROOT=$CONDA_PREFIX

# Clean and configure
echo "Configuring build..."
rm -rf build
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DRDK_INSTALL_INTREE=ON \
  -DRDK_BUILD_PYTHON_WRAPPERS=ON \
  -DRDK_BUILD_CPP_TESTS=ON \
  -DRDK_BUILD_OPENMP=ON \
  -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python \
  -DBOOST_ROOT=$CONDA_PREFIX \
  -DEIGEN3_INCLUDE_DIR=$CONDA_PREFIX/include/eigen3 \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DRDK_BUILD_DESCRIPTORS3D=OFF \
  -DRDK_BUILD_MAEPARSER_SUPPORT=OFF \
  -DRDK_BUILD_COORDGEN_SUPPORT=OFF

# Build
echo "Building..."
make -j$(nproc) Descriptors
make -j$(nproc) rdMolDescriptors

# Install
echo "Installing..."
make install

# Test
echo "Testing..."
cd $RDBASE
python -c "from rdkit.Chem import rdMolDescriptors; print(f'Descriptors: {len(rdMolDescriptors.GetBatchDescriptorNames())}')"

echo "✅ Build complete!"
echo "Activate environment: conda activate rdkit"
echo "Source env: source setup_env.sh"
```

Run with:
```bash
chmod +x setup_and_build.sh
./setup_and_build.sh
```
