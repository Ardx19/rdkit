# AGENTS.md - RDKit Batch Descriptors Expansion Project

## Project Overview

**Phase 1** (COMPLETE ✅): 44 C++ batch descriptors with OpenMP parallelization
**Phase 2** (IN PROGRESS 🚧): Hybrid C++/Python batch API with ProcessPoolExecutor for Python descriptors

### Phase 1 Status: COMPLETE
- ✅ 44 C++ descriptors implemented
- ✅ OpenMP parallelization with GIL release
- ✅ Thread-safe with duplicate molecule handling
- ✅ Comprehensive test suite (67 tests)
- ✅ 19.8x speedup over serial calculation
- ✅ Build and installation verified

### Architecture Evolution

```
Phase 1: C++ Only (44 descriptors)
┌─────────────────────────────────────────────────────────────┐
│ Python: rdMD.CalcDescriptorsBatch(mols, "all")              │
│                    ↓                                         │
│ C++: OpenMP parallel-for (GIL released)                     │
│   - 44 C++ descriptors                                      │
└─────────────────────────────────────────────────────────────┘

Phase 2: Hybrid C++ + Python (200+ descriptors)
┌─────────────────────────────────────────────────────────────┐
│ Python: Descriptors.CalcMolDescriptorsBatch(mols)           │
│                    ↓                                         │
│         ┌──────────────────────┬──────────────────────┐     │
│         ↓                      ↓                      ↓     │
│   C++ Batch (44)      Python Wrapper (24)      Python Only   │
│   - OpenMP           - Thin wrappers          - ProcessPool  │
│   - GIL released     - Call C++ batch         - NumPy ops   │
└─────────────────────────────────────────────────────────────┘
```

## Complete Descriptor Catalog (~434 total)

### Category A: C++ Batch Descriptors (44 total) - COMPLETE
Already implemented in Phase 1 with OpenMP parallelization:
- Basic Properties (4): CalcAMW, CalcExactMolWt, CalcNumAtoms, CalcNumHeavyAtoms
- Surface & Polarity (2): CalcTPSA, CalcLabuteASA
- Crippen Properties (2): CalcClogP, CalcMR
- H-Bond (4): CalcNumHBD, CalcNumHBA, CalcLipinskiHBD, CalcLipinskiHBA
- Flexibility (2): CalcNumRotatableBonds, CalcFractionCSP3
- Special Atoms (4): CalcNumHeteroatoms, CalcNumAmideBonds, CalcNumSpiroAtoms, CalcNumBridgeheadAtoms
- Ring Counts (4): CalcNumRings, CalcNumAromaticRings, CalcNumAliphaticRings, CalcNumSaturatedRings
- Heterocycles (7): CalcNumHeterocycles, CalcNumAromaticHeterocycles, CalcNumAromaticCarbocycles, CalcNumSaturatedHeterocycles, CalcNumSaturatedCarbocycles, CalcNumAliphaticHeterocycles, CalcNumAliphaticCarbocycles
- Chi Valence (5): CalcChi0v, CalcChi1v, CalcChi2v, CalcChi3v, CalcChi4v
- Chi Non-Valence (5): CalcChi0n, CalcChi1n, CalcChi2n, CalcChi3n, CalcChi4n
- Kappa Indices (5): CalcHallKierAlpha, CalcKappa1, CalcKappa2, CalcKappa3, CalcPhi

### Category B: Python Lambda Wrappers → Migrate to C++ Batch (24 descriptors)
These are thin Python lambdas calling C++ single-molecule functions. Should be replaced with batch-aware wrappers calling the C++ batch API:

**In GraphDescriptors.py:**
- HallKierAlpha, Kappa1, Kappa2, Kappa3 (lines 215-222)
- Chi0v, Chi1v, Chi2v, Chi3v, Chi4v (lines 428-437)
- Chi0n, Chi1n, Chi2n, Chi3n, Chi4n (lines 441-450)

**In Lipinski.py:**
- NumHDonors, NumHAcceptors, NumHeteroatoms, NumRotatableBonds
- NOCount, NHOHCount, RingCount (lines 49-72)

**Migration Pattern:**
```python
# OLD: Single-molecule lambda
Kappa1 = lambda x: rdMolDescriptors.CalcKappa1(x)

# NEW: Batch-aware wrapper
def Kappa1(mols):
    if isinstance(mols, Chem.Mol):
        return rdMolDescriptors.CalcKappa1([mols])[0]
    return rdMolDescriptors.CalcKappa1(mols)  # C++ batch
```

### Category C: Simple Python Descriptors → MIGRATE to C++ (12 descriptors)
Fast pure-Python calculations that CAN be migrated to C++:
- **NumValenceElectrons**: Sum of outer electrons (periodic table lookup)
- **NumRadicalElectrons**: Count radical electrons (atom iteration)
- **HeavyAtomMolWt**: Wrapper around MolWt (already in C++)
- **Chi0, Chi1** (Python versions): NumPy sqrt on degree/bond arrays
- **EState Indices**: Max/Min EState values (O(N) operations)

**Could migrate but uses C++ internally:**
- **MaxPartialCharge/MinPartialCharge**: Uses `rdPartialCharges.ComputeGasteigerCharges()` - would need C++ Gasteiger implementation
- **FpDensityMorgan***: Uses `rdFingerprintGenerator.GetMorganGenerator()` - C++ fingerprinting

**Migration Priority: HIGH** - These are simple O(N) operations perfect for C++ batch.

### Category D: NumPy-Based Descriptors → TO BE EXPLORED (6 descriptors)
These use NumPy operations that could potentially be reimplemented in C++:
- **BertzCT** (GraphDescriptors.py:624): Distance matrices + entropy calculations
- **BalabanJ** (GraphDescriptors.py:455): Adjacency matrices + double loops with NumPy sqrt
- **Ipc** (GraphDescriptors.py:109): Characteristic polynomial of adjacency matrix + entropy
- **AvgIpc** (GraphDescriptors.py:140): Wrapper around Ipc
- **EStateIndices** (EState/EState.py): Distance matrix + accumulation

**Why "to be explored"?**
- NumPy is a C library - operations can be reimplemented in C++
- **C++ Distance Matrix**: Available via `RDKit::MolOps::getDistanceMat()` in `GraphMol/MolOps.h`
- **BertzCT/BalabanJ**: Just need C++ distance matrix + algorithm port (medium effort)
- **Ipc**: Needs characteristic polynomial calculation - NOT currently in C++, would need eigenvalue solver
- These descriptors cache intermediate results on molecule objects (`mol._balabanMat`, `mol._adjMat`)

**ProcessPoolExecutor** is a practical interim solution for Phase 2.

### Category E: SMARTS-Based Descriptors (126 descriptors) → HYBRID
- **Fragment Descriptors** (85): fr_Al_COO, fr_aldehyde, etc. - SMARTS from CSV file
- **EState_VSA** (10): VSA bins + EState
- **VSA Descriptors** (35+): SMR_VSA, SlogP_VSA, PEOE_VSA from MolSurf.py

**SMARTS in C++:**
- **Available**: `RDKit::ROMol::getSubstructMatches()` in C++
- **Pattern pre-compilation**: Recommended for batch processing performance
- **Effort**: Medium - requires reimplementing the pattern matching logic

**Migration Priority: MEDIUM** - Complex but could benefit from C++ if heavily used. Needs assessment of SMARTS matching performance in C++ vs Python.

### Category F: 3D Descriptors (11 descriptors) - Already C++
Require 3D conformers, already implemented in C++:
- PMI1, PMI2, PMI3, NPR1, NPR2
- RadiusOfGyration, InertialShapeFactor
- Eccentricity, Asphericity, SpherocityIndex, PBF

### Category G: Vector Descriptors (210 descriptors) - Already C++
These return arrays rather than scalars, already in C++:
- **BCUT2D** (8): Eigenvalue descriptors
- **AUTOCORR2D** (192): 2D autocorrelation
- **MQNs** (42): Molecular quantum numbers

## Phase 2 Implementation Strategy

### 1. C++ Batch Path (Fastest)
```python
# 45 descriptors, OpenMP parallelized
from rdkit.Chem import rdMolDescriptors as rdMD
cpp_results = rdMD.CalcDescriptorsBatch(mols, "all")
# Returns: numpy array (n_mols x 45), dtype=float64
```

### 2. Python Wrapper Path (Medium)
Create batch wrapper functions that:
- Extract descriptor name → map to C++ batch function
- Call C++ batch API for multiple molecules at once
- Return numpy arrays

Example pattern:
```python
def CalcKappa1(mols):
    """Batch wrapper around C++ CalcKappa1."""
    if isinstance(mols, list):
        return rdMD.CalcKappa1(mols)  # Use C++ batch
    else:
        return rdMD.CalcKappa1([mols])[0]  # Single molecule
```

### 3. Python-Only Path (ProcessPool)
For NumPy-heavy descriptors (BertzCT, BalabanJ, Ipc):
```python
from concurrent.futures import ProcessPoolExecutor

def _calc_python_descriptor_batch(fn, mols, n_jobs=-1):
    """Calculate Python descriptor across molecules using ProcessPool."""
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(fn, mols))
    return np.array(results)
```

**Why ProcessPool?**
- Python GIL prevents true thread parallelism
- NumPy operations release GIL but still benefit from process isolation
- Molecules are pickled for inter-process transfer (RDKit supports pickle)

### 4. Migration Priority Matrix

**MIGRATE TO C++ (High Priority):**
- Simple O(N) atom/bond iterations
- No external dependencies (NumPy, Graphs module)
- Currently pure Python wrappers (lambdas)
- Examples: NumValenceElectrons, NumRadicalElectrons, Chi0/Chi1

**USE ProcessPool (Medium Priority):**
- Heavy NumPy operations on matrices
- Uses Python Graphs module
- Complex caching behavior
- Examples: BertzCT, BalabanJ, Ipc

**KEEP IN PYTHON (Low Priority):**
- Rarely used descriptors
- Complex domain-specific logic
- External dependencies (SMARTS, 3D)

## Key Technical Details

### Thread Safety (Critical)
RDKit molecules have lazy-initialized state (ring info, properties).
**Solution in C++**: `extractMolPtrs()` deep-copies duplicates.
**Solution in Python**: Each process gets its own molecule copy via pickle.

### Error Handling Convention
- C++: Returns NaN for failed molecules (doesn't crash batch)
- Python: Should follow same convention
- ProcessPool: Handle exceptions in worker, return NaN

### C++ Capabilities Summary

**Available in C++:**
- All 45 Phase 1 descriptors
- Distance matrix calculation (`MolOps::getDistanceMat`)
- SMARTS matching (`SubstructMatch`)
- Eigenvalue solvers (via Eigen3, optional)
- PMI/Moment calculations (3D)
- VSA calculations

**Not Available in C++:**
- Characteristic polynomial (needed for Ipc)
- EState indices calculation
- BertzCT complexity algorithm
- BalabanJ algorithm

### Memory Considerations

**ProcessPool overhead**:
- Molecules pickled → sent to process → unpickled
- For 10K+ molecules, use chunked processing:
```python
def chunked_batch(mols, chunk_size=1000):
    for i in range(0, len(mols), chunk_size):
        yield mols[i:i+chunk_size]
```

**Distance matrix caching**:
- BertzCT/BalabanJ cache `mol._balabanMat` and `mol._adjMat`
- Cache is per-molecule, not shared across processes
- Each process computes its own matrices (inefficient for small batches)

## Build Commands

```bash
# Setup environment
export RDBASE=$(pwd)
export PYTHONPATH=$RDBASE:$PYTHONPATH
export LD_LIBRARY_PATH=$RDBASE/lib:$LD_LIBRARY_PATH

# Configure (minimal for development)
mkdir -p build && cd build
cmake .. \
  -DRDK_INSTALL_INTREE=ON \
  -DRDK_BUILD_PYTHON_WRAPPERS=ON \
  -DRDK_BUILD_CPP_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DRDK_BUILD_OPENMP=ON

# Build Phase 1 components
make -j$(nproc) Descriptors
make -j$(nproc) rdMolDescriptors
make install
```

## Test Commands

```bash
# Phase 1: C++ batch tests
RDBASE=$RDBASE ctest -R pyBatchDescriptors --output-on-failure

# Phase 2: Python batch tests (to be added)
python -m pytest rdkit/Chem/test_descriptors_batch.py -v

# Run specific test class
python -m unittest rdkit.Chem.test_descriptors_batch.TestBertzCTBatch
```

## Code Style

### C++ (RDKit Standard)
- **Standard**: C++20
- **Format**: `.clang-format` (Google-based, 2-space indent, 80 char limit)
- **Naming**: `CamelCase` classes, `camelCase` functions, `d_` prefix members
- **Exports**: `RDKIT_<MODULE>_EXPORT` macros

### Python
- **Indent**: 4 spaces
- **Naming**: `camelCase` functions (follows C++ conventions), `CamelCase` classes
- **Docstrings**: Google-style with Args/Returns sections
- **Type hints**: Use for batch function signatures

## Implementation Patterns

### Adding a Batch Wrapper (Phase 2)

1. **Check if C++ batch exists**:
   - If yes: Create thin wrapper calling `rdMD.Calc<Descriptor>(mols)`
   - If no: Use ProcessPool for Python implementation

2. **Update `_descList` registration**:
```python
# In Descriptors.py _setupDescriptors()
# Replace lambda with batch-aware function
```

3. **Add test**:
```python
class TestBertzCTBatch(unittest.TestCase):
    def test_batch_matches_serial(self):
        mols = [Chem.MolFromSmiles(s) for s in ['CCO', 'c1ccccc1']]
        serial = [BertzCT(m) for m in mols]
        batch = CalcBertzCTBatch(mols)  # New batch function
        np.testing.assert_array_almost_equal(serial, batch)
```

## Project Files

```
Code/GraphMol/Descriptors/Wrap/
├── rdMolDescriptors.cpp      # Phase 1: 45 C++ batch descriptors (COMPLETE)
├── BatchUtils.h               # OpenMP parallelization utilities
└── test_batch_descriptors.py  # Phase 1 tests (COMPLETE)

rdkit/Chem/
├── Descriptors.py            # Phase 2: Add CalcMolDescriptorsBatch()
├── GraphDescriptors.py       # Phase 2: Add batch wrappers for BertzCT, etc.
└── test_descriptors_batch.py # Phase 2: Hybrid batch tests (TO CREATE)
```

## Performance Expectations

- **C++ Batch (45 desc)**: 5-10x speedup vs serial
- **Python Wrappers (45 desc)**: Same as C++ batch (thin overhead)
- **Python-Only (NumPy)**: 2-4x speedup with ProcessPool (depends on cores)
- **Combined (200+ desc)**: Overall 3-6x speedup (C++ dominates runtime)

## Troubleshooting

**ImportError for rdBase**:
```bash
export PYTHONPATH=$RDBASE:$PYTHONPATH
export LD_LIBRARY_PATH=$RDBASE/lib:$LD_LIBRARY_PATH
make install  # Must run install
```

**ProcessPool hangs**:
- Check if descriptor functions are picklable (no lambdas)
- Wrap in `if __name__ == '__main__':` guard
- Reduce `max_workers` if memory constrained

**Descriptor count mismatch**:
```bash
# Clean rebuild
cd $RDBASE/build
make clean
make -j$(nproc) Descriptors rdMolDescriptors
make install
```

## Batch API Usage Examples

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors as rdMD

mols = [Chem.MolFromSmiles(s) for s in ['CCO', 'c1ccccc1', 'CC(=O)O']]

# Phase 1: C++ only (45 descriptors)
cpp_results = rdMD.CalcDescriptorsBatch(mols, "all")  # Shape: (3, 45)

# Phase 2: Hybrid (all 200+ descriptors)
all_results = Descriptors.CalcMolDescriptorsBatch(mols)
# Automatically routes to:
# - C++ batch for 45 fast descriptors
# - ProcessPool for Python/NumPy descriptors
```
