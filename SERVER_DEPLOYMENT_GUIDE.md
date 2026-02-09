# RDKit Batch Descriptors - Deployment Guide

**Quick Start:** Build and deploy RDKit with batch descriptor support.

**For Project Status:** See `PROGRESS.md`  
**For Developer Guidelines:** See `AGENTS.md`

---

## Overview

This guide covers building and deploying RDKit with the batch descriptor expansion (Phase 1 complete: 44 C++ descriptors with OpenMP, 19.8x speedup).

**What's Included:**
- 44 C++ batch descriptors with OpenMP parallelization
- Python API: `rdMolDescriptors.CalcDescriptorsBatch(mols, "all")`
- Returns: numpy array `(n_molecules, 44)`
- Performance: 19.8x faster than serial calculation

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04/22.04, CentOS 8, RHEL 8+ | Ubuntu 22.04 LTS |
| CPU | 4 cores | 8+ cores (OpenMP benefits) |
| RAM | 8 GB | 16-32 GB |
| Disk | 10 GB free | 20 GB free (build artifacts) |
| Compiler | GCC 9+ or Clang 10+ | GCC 11+ |
| CMake | 3.18+ | 3.26+ |

---

## Quick Deploy (Conda - Recommended)

### 1. Setup Environment

```bash
# Install miniconda if not present
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment
conda create -n rdkit python=3.11 -y
conda activate rdkit

# Install dependencies
conda install -c conda-forge cmake=3.26 boost=1.82 eigen numpy -y
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y
```

### 2. Build

```bash
export RDBASE=$(pwd)
export PYTHONPATH=$RDBASE:$PYTHONPATH
export LD_LIBRARY_PATH=$RDBASE/lib:$LD_LIBRARY_PATH

mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DRDK_INSTALL_INTREE=ON \
    -DRDK_BUILD_PYTHON_WRAPPERS=ON \
    -DRDK_BUILD_CPP_TESTS=ON \
    -DRDK_BUILD_OPENMP=ON \
    -DRDK_BUILD_DESCRIPTORS3D=OFF \
    -DRDK_BUILD_MAEPARSER_SUPPORT=OFF \
    -DRDK_BUILD_COORDGEN_SUPPORT=OFF

make -j$(nproc) Descriptors rdMolDescriptors
make install
```

**Build Time:** 20-30 minutes (first time), 5-10 minutes (with ccache)

### 3. Verify

```bash
cd $RDBASE
python << 'PYTEST'
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

# Test
mols = [Chem.MolFromSmiles(s) for s in ['CCO', 'c1ccccc1']]
results = rdMD.CalcDescriptorsBatch(mols, "all")

print(f"✓ {len(rdMD.GetBatchDescriptorNames())} batch descriptors")
print(f"✓ Shape: {results.shape}")
print("✓ Deployment successful!")
PYTEST
```

---

## Alternative: System Package Manager

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libeigen3-dev \
    python3-dev \
    python3-numpy

# Build (same commands as above)
```

### CentOS/RHEL

```bash
# Enable EPEL
sudo yum install -y epel-release

# Install dependencies
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    cmake \
    boost-devel \
    eigen3-devel \
    python3-devel \
    python3-numpy

# Build (same commands as above)
```

---

## Full Build Script

Save as `deploy.sh`:

```bash
#!/bin/bash
set -e

echo "=== RDKit Batch Descriptors Deployment ==="

# Check conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install miniconda first."
    exit 1
fi

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rdkit 2>/dev/null || conda create -n rdkit python=3.11 -y
conda activate rdkit

# Install dependencies
echo "Installing dependencies..."
conda install -c conda-forge cmake=3.26 boost=1.82 eigen numpy -y
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y

# Environment
export RDBASE=$(pwd)
export PYTHONPATH=$RDBASE:$PYTHONPATH
export LD_LIBRARY_PATH=$RDBASE/lib:$LD_LIBRARY_PATH

echo "Building RDKit..."
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DRDK_INSTALL_INTREE=ON \
    -DRDK_BUILD_PYTHON_WRAPPERS=ON \
    -DRDK_BUILD_CPP_TESTS=ON \
    -DRDK_BUILD_OPENMP=ON \
    -DRDK_BUILD_DESCRIPTORS3D=OFF \
    -DRDK_BUILD_MAEPARSER_SUPPORT=OFF \
    -DRDK_BUILD_COORDGEN_SUPPORT=OFF

make -j$(nproc) Descriptors rdMolDescriptors
make install

echo "Testing..."
cd $RDBASE
python -c "
from rdkit.Chem import rdMolDescriptors
print(f'Descriptors: {len(rdMolDescriptors.GetBatchDescriptorNames())}')
assert len(rdMolDescriptors.GetBatchDescriptorNames()) == 44
print('✓ Deployment successful!')
"

echo ""
echo "=== Deployment Complete ==="
echo "Run tests: cd build && ctest -R pyBatchDescriptors --output-on-failure"
```

Run: `bash deploy.sh`

---

## Testing

### Run Test Suite

```bash
# Full test suite
cd $RDBASE/build
ctest -R pyBatchDescriptors --output-on-failure

# Expected: 100% tests passed (67 tests)
```

### Performance Benchmark

```bash
cd $RDBASE
python << 'PYBENCH'
import time
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

mols = [Chem.MolFromSmiles('C' * i) for i in range(1, 101)]

# Benchmark
start = time.time()
results = rdMD.CalcDescriptorsBatch(mols, "all")
elapsed = time.time() - start

print(f"Molecules: {len(mols)}")
print(f"Descriptors: {results.shape[1]}")
print(f"Time: {elapsed:.3f} seconds")
print(f"Throughput: {len(mols)/elapsed:.1f} mol/s")
print(f"✓ Performance: {elapsed:.2f}s for 100 molecules (44 descriptors)")
PYBENCH
```

**Expected:** ~0.3-0.5 seconds for 100 molecules

---

## Usage

### Basic Usage

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

# Load molecules
mols = [Chem.MolFromSmiles(s) for s in ['CCO', 'c1ccccc1', 'CC(=O)O']]

# Calculate all 44 descriptors
results = rdMD.CalcDescriptorsBatch(mols, "all")
print(f"Shape: {results.shape}")  # (3, 44)

# Calculate specific descriptors
subset = ["CalcExactMolWt", "CalcTPSA", "CalcClogP"]
results = rdMD.CalcDescriptorsBatch(mols, subset)
print(f"Shape: {results.shape}")  # (3, 3)

# Individual batch functions
weights = rdMD.CalcExactMolWt(mols)
chi_values = rdMD.CalcChi0v(mols)
```

### Integration with Pandas

```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
mols = [Chem.MolFromSmiles(s) for s in smiles]

# Calculate descriptors
results = rdMD.CalcDescriptorsBatch(mols, "all")
names = rdMD.GetBatchDescriptorNames()

# Create DataFrame
df = pd.DataFrame(results, columns=names)
df['smiles'] = smiles

print(df.head())
```

---

## Troubleshooting

### ImportError: No module named 'rdkit'

```bash
# Fix PYTHONPATH
export PYTHONPATH=/path/to/rdkit:$PYTHONPATH
export LD_LIBRARY_PATH=/path/to/rdkit/lib:$LD_LIBRARY_PATH

# Verify
ls $RDBASE/rdkit/Chem/rdMolDescriptors*.so
```

### CMake can't find Boost

```bash
# Specify Boost location
cmake .. -DBOOST_ROOT=/usr/local/boost
# or
cmake .. -DBOOST_ROOT=$CONDA_PREFIX
```

### Descriptor count mismatch

```bash
# Clean rebuild
cd $RDBASE/build
make clean
make -j$(nproc) Descriptors rdMolDescriptors
make install
```

### Out of memory during build

```bash
# Use fewer parallel jobs
make -j2 Descriptors

# Add swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Tests fail

```bash
# Check environment
export RDBASE=/path/to/rdkit
export PYTHONPATH=$RDBASE:$PYTHONPATH
export LD_LIBRARY_PATH=$RDBASE/lib:$LD_LIBRARY_PATH

# Rebuild tests
make -j$(nproc) pyBatchDescriptors
ctest -R pyBatchDescriptors -V
```

---

## Performance Reference

| Configuration | 100 Molecules | 1000 Molecules | Throughput |
|--------------|---------------|----------------|------------|
| Serial (Python loop) | ~6.5s | ~65s | 3 mol/s |
| **C++ Batch (OpenMP)** | **~0.3s** | **~3s** | **300+ mol/s** |
| **Speedup** | **19.8x** | **20x** | - |

*Benchmark: 44 descriptors, Intel i7-8700K (6 cores), GCC 11.4.0*

---

## Docker Deployment (Optional)

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libeigen3-dev \
    python3-dev \
    python3-numpy \
    python3-pip

WORKDIR /app
COPY . .

RUN mkdir -p build && cd build && \
    cmake .. -DRDK_INSTALL_INTREE=ON \
             -DRDK_BUILD_PYTHON_WRAPPERS=ON \
             -DRDK_BUILD_OPENMP=ON && \
    make -j$(nproc) Descriptors rdMolDescriptors && \
    make install

ENV RDBASE=/app
ENV PYTHONPATH=/app:$PYTHONPATH
ENV LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH

CMD ["python3", "-c", "from rdkit.Chem import rdMolDescriptors; print('Ready')"]
```

Build: `docker build -t rdkit-batch .`  
Run: `docker run -it rdkit-batch`

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `RDBASE` | RDKit source directory | `/home/user/rdkit` |
| `PYTHONPATH` | Python module search path | `$RDBASE:$PYTHONPATH` |
| `LD_LIBRARY_PATH` | Shared library path | `$RDBASE/lib:$LD_LIBRARY_PATH` |

Add to `~/.bashrc` for persistence:
```bash
export RDBASE=/home/swarnavas/Work/PhD_Work/Covaln_Dev_work/rdkit
export PYTHONPATH=$RDBASE:$PYTHONPATH
export LD_LIBRARY_PATH=$RDBASE/lib:$LD_LIBRARY_PATH
```

---

## Support

**Documentation:**
- `PROGRESS.md` - Project status and roadmap
- `AGENTS.md` - Developer guidelines
- `BUILD_INSTRUCTIONS.md` - Detailed build guide

**Testing:**
```bash
# Run all tests
cd $RDBASE/build && ctest --output-on-failure

# Run specific test
ctest -R pyBatchDescriptors -V
```

---

**Version:** 2.1  
**Last Updated:** February 9, 2026  
**Status:** Phase 1 Complete (44 C++ descriptors with OpenMP)
