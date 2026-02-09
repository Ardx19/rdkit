# Build and Test Instructions for Expanded Batch Descriptors

## Summary of Changes

This implementation expands the batch descriptor API from 10 to 45 descriptors:

**New Descriptors Added (35):**
- Basic: CalcAMW, CalcNumAtoms
- H-bond: CalcLipinskiHBD, CalcLipinskiHBA  
- Atoms: CalcNumHeteroatoms, CalcNumAmideBonds, CalcNumSpiroAtoms, CalcNumBridgeheadAtoms
- Rings: CalcNumRings, CalcNumAromaticRings, CalcNumAliphaticRings, CalcNumSaturatedRings
- Cycles: CalcNumHeterocycles, CalcNumAromaticHeterocycles, CalcNumAromaticCarbocycles, CalcNumSaturatedHeterocycles, CalcNumSaturatedCarbocycles, CalcNumAliphaticHeterocycles, CalcNumAliphaticCarbocycles
- Chi indices: CalcChi0v, CalcChi1v, CalcChi2v, CalcChi3v, CalcChi4v, CalcChi0n, CalcChi1n, CalcChi2n, CalcChi3n, CalcChi4n
- Kappa: CalcHallKierAlpha, CalcKappa1, CalcKappa2, CalcKappa3, CalcPhi

**Files Modified:**
1. `Code/GraphMol/Descriptors/Wrap/rdMolDescriptors.cpp` - Expanded registry and added batch functions
2. `Code/GraphMol/Descriptors/Wrap/test_batch_descriptors.py` - Updated tests for 45 descriptors

## Build Instructions

```bash
# 1. Set environment variables
export RDBASE=$(pwd)
export PYTHONPATH=$RDBASE:$PYTHONPATH

# macOS:
export DYLD_FALLBACK_LIBRARY_PATH=$RDBASE/lib:$CONDA_PREFIX/lib

# Linux:
# export LD_LIBRARY_PATH=$RDBASE/lib:$LD_LIBRARY_PATH

# 2. Create build directory
mkdir -p build && cd build

# 3. Configure CMake (minimal build for faster iteration)
cmake .. \
  -DRDK_INSTALL_INTREE=ON \
  -DRDK_BUILD_PYTHON_WRAPPERS=ON \
  -DRDK_BUILD_CPP_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

# 4. Build only the descriptors module and Python wrappers
make -j$(nproc) Descriptors rdMolDescriptors

# 5. Install (required for Python to find the module)
make install
```

## Test Instructions

```bash
# Run the batch descriptors test
RDBASE=$RDBASE ctest -R pyBatchDescriptors --output-on-failure

# Run all descriptor-related tests
RDBASE=$RDBASE ctest -R Descriptor --output-on-failure

# Run with verbose output
RDBASE=$RDBASE ctest -R pyBatchDescriptors -V
```

## Manual Testing

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD
import numpy as np

# Load test molecules
smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN(CC)CC']
mols = [Chem.MolFromSmiles(s) for s in smiles]

# Get available descriptor names
names = rdMD.GetBatchDescriptorNames()
print(f"Number of descriptors: {len(names)}")  # Should be 45

# Calculate all descriptors at once
results = rdMD.CalcDescriptorsBatch(mols, "all")
print(f"Result shape: {results.shape}")  # Should be (4, 45)

# Calculate specific descriptors
results = rdMD.CalcDescriptorsBatch(mols, ["CalcExactMolWt", "CalcTPSA", "CalcClogP"])
print(f"Result shape: {results.shape}")  # Should be (4, 3)

# Use individual batch functions
weights = rdMD.CalcExactMolWt(mols)
tpsa = rdMD.CalcTPSA(mols)
print(f"Weights: {weights}")
print(f"TPSA: {tpsa}")

# New batch functions
chi0v = rdMD.CalcChi0v(mols)
kappa1 = rdMD.CalcKappa1(mols)
print(f"Chi0v: {chi0v}")
print(f"Kappa1: {kappa1}")
```

## Expected Test Results

All tests should pass:
- `test_count`: Expects 45 descriptor names
- `test_known_names`: Checks all 45 expected names are present
- `test_order_matches_all`: Verifies order matches registry
- `test_all_descriptors_correctness`: Verifies batch results match individual calls

## Troubleshooting

**Issue: Module not found**
- Ensure `make install` was run
- Check `PYTHONPATH` includes `$RDBASE`
- Check `LD_LIBRARY_PATH` or `DYLD_FALLBACK_LIBRARY_PATH` is set

**Issue: ImportError for rdBase**
- Make sure you're using the built Python, not system Python
- Verify `RDBASE` points to the repo root

**Issue: Tests fail with mismatch**
- Ensure you're on the correct branch: `git branch` should show `feature/expand-batch-descriptors`
- Clean build: `rm -rf build && mkdir build && cd build && cmake .. && make`
