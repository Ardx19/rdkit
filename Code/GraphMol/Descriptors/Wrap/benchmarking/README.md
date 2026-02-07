# NEW BATCH API REFERENCE - RDKit Parallel Descriptor Computation

Date: 2026-02-08

## Table of Contents

1. Overview
2. Prerequisites and Installation
3. New Python Functions Summary
4. API Reference - Scalar Functions (2 new)
5. API Reference - Batch Overloads (10)
6. API Reference - Multi-Descriptor Batch (new)
7. API Reference - Descriptor Discovery (new)
8. Usage Patterns and Examples
9. Thread Control
10. Error Handling and Edge Cases
11. Quick-Start Copy-Paste Examples

## Overview

This fork adds OpenMP-based parallel batch descriptor computation to RDKit.
Instead of calling a descriptor function once per molecule in a Python loop
(which holds the GIL on every call), you pass a Python list of molecules and
get back a `numpy.ndarray` with all results computed in parallel in C++.

Key properties of all batch APIs:

- Input: Python list of RDKit molecule objects (None entries allowed)
- Output: `numpy.ndarray`, `dtype=float64`
- None or failed molecules produce `NaN` in the output array
- GIL is released during computation (other Python threads can run)
- OpenMP parallelism when built with `-DRDK_BUILD_OPENMP=ON`
- Without OpenMP, batch APIs still work (single-threaded C++ loop,
  still faster than Python loops due to GIL release and no per-molecule
  Python object allocation)

There are two ways to use the batch APIs:

- Individual batch overloads: call the same function name with a list
  instead of a single molecule. Returns a 1D numpy array.

  ```python
  rdMD.CalcExactMolWt(mols)   # returns 1D array, shape (N,)
  ```

- Multi-descriptor batch: compute multiple descriptors at once with
  a single call. Returns a 2D numpy array.

  ```python
  rdMD.CalcDescriptorsBatch(mols, ["CalcExactMolWt", "CalcTPSA"])
  # returns 2D array, shape (N, 2)
  ```

## Prerequisites and Installation

Step 1: Build RDKit from source with OpenMP enabled.

You need a C++ compiler with OpenMP support (GCC 4.9+, Clang 3.8+).

```bash
git clone https://github.com/Ardx19/rdkit.git
cd rdkit
mkdir build && cd build

cmake .. \
  -DRDK_BUILD_OPENMP=ON \
  -DRDK_INSTALL_INTREE=ON \
  -DRDK_BUILD_PYTHON_WRAPPERS=ON \
  -DPYTHON_EXECUTABLE=$(which python3) \
  <any other cmake flags you normally use>

# Build (use -j4 to -j6 max to avoid out-of-memory)
make -j4
# Or if using ninja:
ninja -j4
```

Step 2: Set `PYTHONPATH` so Python finds the built modules.

```bash
export PYTHONPATH=/path/to/rdkit:$PYTHONPATH
```

Step 3: Verify it works.

```bash
python3 -c "
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD
mols = [Chem.MolFromSmiles(s) for s in ['CCO', 'c1ccccc1', 'CC(=O)O']]
print('ExactMolWt:', rdMD.CalcExactMolWt(mols))
print('Batch:', rdMD.CalcDescriptorsBatch(mols, 'all'))
print('Names:', rdMD.GetBatchDescriptorNames())
"
```

Expected output (values approximate):

```text
ExactMolWt: [46.04186484 78.04695021 60.02112937]
Batch: [[ 46.04186484  45.15       -0.0014  ...snip... ]
        [ 78.04695021   0.           1.6866  ...snip... ]
        [ 60.02112937  37.3        -0.2682  ...snip... ]]
Names: ['CalcExactMolWt', 'CalcTPSA', 'CalcClogP', 'CalcMR',
        'CalcNumHBD', 'CalcNumHBA', 'CalcNumRotatableBonds',
        'CalcFractionCSP3', 'CalcLabuteASA', 'CalcNumHeavyAtoms']
```

Without OpenMP:

If you build without `-DRDK_BUILD_OPENMP=ON`, all batch APIs still work
exactly the same way (same function signatures, same return types),
but computation runs single-threaded. You lose multi-core parallelism
but still benefit from:

- No Python GIL contention per molecule
- No Python float/int object allocation per result
- Direct numpy array construction in C++

## New Python Functions Summary

All functions live in `rdkit.Chem.rdMolDescriptors` (imported as `rdMD` below).

**New scalar functions (2):**

- `rdMD.CalcClogP(mol) -> float`
- `rdMD.CalcMR(mol) -> float`

**Batch overloads (10) - pass a list, get a 1D numpy.ndarray:**

- `rdMD.CalcExactMolWt(mols, onlyHeavy=False)`
- `rdMD.CalcTPSA(mols, includeSandP=False)`
- `rdMD.CalcClogP(mols)`
- `rdMD.CalcMR(mols)`
- `rdMD.CalcNumHBD(mols)`
- `rdMD.CalcNumHBA(mols)`
- `rdMD.CalcNumRotatableBonds(mols, strict=Default)`
- `rdMD.CalcFractionCSP3(mols)`
- `rdMD.CalcLabuteASA(mols, includeHs=True)`
- `rdMD.CalcNumHeavyAtoms(mols)`

**Multi-descriptor batch (1) - compute multiple descriptors, get 2D array:**

- `rdMD.CalcDescriptorsBatch(mols, descriptors) -> numpy.ndarray (2D)`

**Descriptor discovery (1) - list valid descriptor names:**

- `rdMD.GetBatchDescriptorNames() -> list[str]`

Boost.Python dispatches automatically: pass a single mol and you get a
scalar (float or int); pass a list and you get a `numpy.ndarray`.

## API Reference - Scalar Functions (2 new)

### `CalcClogP(mol) -> float` [new scalar]

Returns the Wildman-Crippen LogP for a single molecule.
Equivalent to `CalcCrippenDescriptors(mol)[0]` but faster (no tuple).

Parameters:

- `mol`: RDKit Mol object

Example:

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

mol = Chem.MolFromSmiles('c1ccccc1')
rdMD.CalcClogP(mol)
```

### `CalcMR(mol) -> float` [new scalar]

Returns the Wildman-Crippen MR for a single molecule.
Equivalent to `CalcCrippenDescriptors(mol)[1]` but faster (no tuple).

Parameters:

- `mol`: RDKit Mol object

Example:

```python
rdMD.CalcMR(mol)
```

## API Reference - Batch Overloads (10)

Each function below accepts either a single molecule (returns scalar) or
a Python list of molecules (returns 1D `numpy.ndarray`, `dtype=float64`).
Boost.Python dispatches automatically based on the argument type.

### `CalcExactMolWt(mols, onlyHeavy=False) -> numpy.ndarray` [batch]

Exact molecular weight.

Parameters:

- `mols`: list of RDKit Mol objects (None allowed)
- `onlyHeavy`: bool, default False - exclude hydrogen mass if True

Example:

```python
mols = [Chem.MolFromSmiles(s) for s in ['C', 'CC', 'CCC']]
rdMD.CalcExactMolWt(mols)
```

### `CalcTPSA(mols, includeSandP=False) -> numpy.ndarray` [batch]

Topological polar surface area.

Parameters:

- `mols`: list of RDKit Mol objects (None allowed)
- `includeSandP`: bool, default False - include S and P in TPSA

Example:

```python
mols = [Chem.MolFromSmiles(s) for s in ['O', 'N', 'C']]
rdMD.CalcTPSA(mols)
```

### `CalcClogP(mols) -> numpy.ndarray` [batch]

Wildman-Crippen LogP.

Parameters:

- `mols`: list of RDKit Mol objects (None allowed)

Example:

```python
mols = [Chem.MolFromSmiles('c1ccccc1'), Chem.MolFromSmiles('CCCO')]
rdMD.CalcClogP(mols)
```

### `CalcMR(mols) -> numpy.ndarray` [batch]

Wildman-Crippen MR.

Parameters:

- `mols`: list of RDKit Mol objects (None allowed)

Example:

```python
rdMD.CalcMR(mols)
```

### `CalcNumHBD(mols) -> numpy.ndarray` [batch]

Number of H-bond donors.

Parameters:

- `mols`: list of RDKit Mol objects (None allowed)

Note: Returns `float64` (not int) so `NaN` can represent None/failures.

Example:

```python
mols = [Chem.MolFromSmiles('O'), Chem.MolFromSmiles('C')]
rdMD.CalcNumHBD(mols)
```

### `CalcNumHBA(mols) -> numpy.ndarray` [batch]

Number of H-bond acceptors.

Parameters:

- `mols`: list of RDKit Mol objects (None allowed)

Note: Returns `float64` (not int) so `NaN` can represent None/failures.

Example:

```python
mols = [Chem.MolFromSmiles('O'), Chem.MolFromSmiles('C')]
rdMD.CalcNumHBA(mols)
```

### `CalcNumRotatableBonds(mols, strict=Default) -> numpy.ndarray` [batch]

Number of rotatable bonds.

Parameters:

- `mols`: list of RDKit Mol objects (None allowed)
- `strict`: `NumRotatableBondsOptions` enum, default `Default`
  - Options: `NonStrict`, `Strict`, `StrictLinkages`, `Default`

Note: Returns `float64` (not int) so `NaN` can represent None/failures.

Example:

```python
from rdkit.Chem.rdMolDescriptors import NumRotatableBondsOptions

mols = [Chem.MolFromSmiles('CCCCCC')]
rdMD.CalcNumRotatableBonds(mols)
rdMD.CalcNumRotatableBonds(mols, strict=NumRotatableBondsOptions.Strict)
```

### `CalcFractionCSP3(mols) -> numpy.ndarray` [batch]

Fraction of C atoms that are SP3 hybridized.

Parameters:

- `mols`: list of RDKit Mol objects (None allowed)

Example:

```python
mols = [Chem.MolFromSmiles('CCCC'), Chem.MolFromSmiles('c1ccccc1')]
rdMD.CalcFractionCSP3(mols)
```

### `CalcLabuteASA(mols, includeHs=True) -> numpy.ndarray` [batch]

Labute approximate surface area.

Parameters:

- `mols`: list of RDKit Mol objects (None allowed)
- `includeHs`: bool, default True - include implicit H contributions

Example:

```python
mols = [Chem.MolFromSmiles('C'), Chem.MolFromSmiles('CC')]
rdMD.CalcLabuteASA(mols)
```

### `CalcNumHeavyAtoms(mols) -> numpy.ndarray` [batch]

Number of heavy (non-hydrogen) atoms.

Parameters:

- `mols`: list of RDKit Mol objects (None allowed)

Note: Returns `float64` (not int) so `NaN` can represent None/failures.

Example:

```python
mols = [Chem.MolFromSmiles('C'), Chem.MolFromSmiles('c1ccccc1')]
rdMD.CalcNumHeavyAtoms(mols)
```

## API Reference - Multi-Descriptor Batch (new)

### `CalcDescriptorsBatch(mols, descriptors) -> numpy.ndarray (2D)`

Compute multiple descriptors over a list of molecules in a single call.
Each descriptor is computed in parallel via OpenMP (multi-pass strategy:
one parallel pass per descriptor, results assembled into a 2D array).

Parameters:

- `mols`: list of RDKit Mol objects (None allowed)
- `descriptors`: either a list of descriptor name strings,
  or the string `"all"` to compute all 10 descriptors

Returns:

- `numpy.ndarray`, `dtype=float64`, `shape (len(mols), len(descriptors))`
- Row `i` = molecule `i`. Column `j` = descriptor `j`.
- None/failed molecules produce `NaN` for the entire row.

Valid descriptor names (must match exactly, case-sensitive):

- `"CalcExactMolWt"` - Exact molecular weight
- `"CalcTPSA"` - Topological polar surface area
- `"CalcClogP"` - Wildman-Crippen LogP
- `"CalcMR"` - Wildman-Crippen MR
- `"CalcNumHBD"` - Number of H-bond donors
- `"CalcNumHBA"` - Number of H-bond acceptors
- `"CalcNumRotatableBonds"` - Number of rotatable bonds (default mode)
- `"CalcFractionCSP3"` - Fraction of SP3 carbons
- `"CalcLabuteASA"` - Labute approximate surface area
- `"CalcNumHeavyAtoms"` - Number of heavy atoms

Important: The multi-descriptor batch API uses default parameter values
for all descriptors. If you need non-default parameters (for example
`CalcExactMolWt` with `onlyHeavy=True`, or `CalcNumRotatableBonds` with
`strict=Strict`), use the individual batch overload instead.

Raises:

- `ValueError` if any descriptor name is not recognized
- `ValueError` if `descriptors` is a string other than `"all"`

Examples:

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

mols = [Chem.MolFromSmiles(s) for s in ['CCO', 'c1ccccc1', 'CC(=O)O']]
result = rdMD.CalcDescriptorsBatch(
    mols, ["CalcExactMolWt", "CalcTPSA", "CalcClogP", "CalcNumHBD"]
)
result.shape
result[0]  # ethanol: MolWt, TPSA, ClogP, NumHBD
```

```python
# Compute all 10 descriptors at once:
result = rdMD.CalcDescriptorsBatch(mols, "all")
result.shape
```

```python
# Column order matches GetBatchDescriptorNames():
names = rdMD.GetBatchDescriptorNames()
names[0]
result[:, 0]  # ExactMolWt for all molecules
```

```python
# With None molecules (invalid SMILES):
mols = [Chem.MolFromSmiles('CCO'), None, Chem.MolFromSmiles('C')]
result = rdMD.CalcDescriptorsBatch(mols, ["CalcTPSA", "CalcClogP"])
result
```

```python
# Single descriptor:
result = rdMD.CalcDescriptorsBatch(mols, ["CalcClogP"])
result.shape
```

## API Reference - Descriptor Discovery (new)

### `GetBatchDescriptorNames() -> list[str]`

Returns the list of valid descriptor name strings that
`CalcDescriptorsBatch()` accepts. The order matches the column order
when using `CalcDescriptorsBatch(mols, "all")`.

Takes no arguments.

Example:

```python
from rdkit.Chem import rdMolDescriptors as rdMD

names = rdMD.GetBatchDescriptorNames()
names
len(names)
```

Typical use: iterate over names to build column headers for a DataFrame:

```python
import pandas as pd

names = rdMD.GetBatchDescriptorNames()
result = rdMD.CalcDescriptorsBatch(mols, "all")
df = pd.DataFrame(result, columns=names)
```

## Usage Patterns and Examples

### Pattern 1: Basic batch call (individual descriptors)

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O', 'NCCCC']
mols = [Chem.MolFromSmiles(s) for s in smiles_list]

weights = rdMD.CalcExactMolWt(mols)  # numpy.ndarray, shape (4,)
logp = rdMD.CalcClogP(mols)          # numpy.ndarray, shape (4,)
tpsa = rdMD.CalcTPSA(mols)           # numpy.ndarray, shape (4,)
```

### Pattern 2: Multi-descriptor batch (one call, all descriptors)

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD
import numpy as np

smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O', 'NCCCC']
mols = [Chem.MolFromSmiles(s) for s in smiles_list]

# Compute all 10 descriptors at once:
result = rdMD.CalcDescriptorsBatch(mols, "all")
# result.shape == (4, 10), dtype=float64

# Or pick specific descriptors:
result = rdMD.CalcDescriptorsBatch(mols, ["CalcClogP", "CalcTPSA"])
# result.shape == (4, 2)
```

### Pattern 3: Handling None molecules (e.g. invalid SMILES)

```python
mols = [Chem.MolFromSmiles(s) for s in ['CCO', 'INVALID', 'c1ccccc1']]
# mols == [<Mol>, None, <Mol>]

result = rdMD.CalcExactMolWt(mols)
# result == array([46.041..., nan, 78.046...])

import numpy as np
valid_mask = ~np.isnan(result)          # array([True, False, True])
valid_values = result[valid_mask]       # array([46.041..., 78.046...])

# Same for multi-descriptor:
result2d = rdMD.CalcDescriptorsBatch(mols, ["CalcClogP", "CalcTPSA"])
# Row 1 (the None molecule) is [nan, nan]
valid_rows = ~np.isnan(result2d).any(axis=1)   # [True, False, True]
```

### Pattern 4: Converting integer descriptors

```python
hbd = rdMD.CalcNumHBD(mols)             # array([1., nan, 0.])
# To get integers for valid entries:
valid = hbd[~np.isnan(hbd)].astype(int) # array([1, 0])
```

### Pattern 5: Building a descriptor DataFrame (individual calls)

```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O', 'NCCCC']
mols = [Chem.MolFromSmiles(s) for s in smiles_list]

df = pd.DataFrame({
    'SMILES': smiles_list,
    'MolWt': rdMD.CalcExactMolWt(mols),
    'LogP': rdMD.CalcClogP(mols),
    'MR': rdMD.CalcMR(mols),
    'TPSA': rdMD.CalcTPSA(mols),
    'HBD': rdMD.CalcNumHBD(mols),
    'HBA': rdMD.CalcNumHBA(mols),
    'RotBonds': rdMD.CalcNumRotatableBonds(mols),
    'FracCSP3': rdMD.CalcFractionCSP3(mols),
    'LabuteASA': rdMD.CalcLabuteASA(mols),
    'HeavyAtoms': rdMD.CalcNumHeavyAtoms(mols),
})
```

### Pattern 6: Building a descriptor DataFrame (CalcDescriptorsBatch, easiest)

```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O', 'NCCCC']
mols = [Chem.MolFromSmiles(s) for s in smiles_list]

names = rdMD.GetBatchDescriptorNames()   # 10 descriptor names
result = rdMD.CalcDescriptorsBatch(mols, "all")  # shape (4, 10)

df = pd.DataFrame(result, columns=names)
df.insert(0, 'SMILES', smiles_list)

print(df)
```

### Pattern 7: Large-scale processing (millions of molecules)

```python
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Set BEFORE importing rdkit

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD
import numpy as np

# Read molecules from an SDF file
suppl = Chem.SDMolSupplier('large_dataset.sdf')
mols = [m for m in suppl]  # may contain None entries

# One call computes all 10 descriptors for all molecules:
result = rdMD.CalcDescriptorsBatch(mols, "all")
print(
    f"Computed {result.shape[1]} descriptors for {result.shape[0]} molecules"
)

# Filter out failed molecules:
valid = ~np.isnan(result).any(axis=1)
clean = result[valid]
print(f"Valid molecules: {clean.shape[0]}")
```

## Thread Control

OpenMP must be enabled at CMake configure time:

```bash
cmake -DRDK_BUILD_OPENMP=ON <other flags> ..
```

This sets `-fopenmp` for compilation and links `libgomp`. Without this flag,
the batch APIs still work (GIL release + single-threaded C++ loop) but
you will not get multi-core parallelism.

Control thread count at runtime:

```bash
export OMP_NUM_THREADS=4
python3 my_script.py
```

Or from Python before importing rdkit:

```python
import os
os.environ['OMP_NUM_THREADS'] = '4'
```

Tips:

- Setting `OMP_NUM_THREADS` to the number of physical cores usually
  gives the best performance. Hyper-threaded virtual cores rarely help.
- For very cheap descriptors (ExactMolWt, NumHeavyAtoms) the overhead
  of thread creation may reduce the speedup. The batch API still avoids
  Python overhead, so it is always at least as fast as a Python loop.
- For expensive descriptors (NumRotatableBonds, NumHBA, NumHBD) the
  speedup is nearly linear with thread count (4-6x on 6 threads).

## Error Handling and Edge Cases

None molecules:

- Individual batch: produces `NaN` at the corresponding index.
- Multi-descriptor batch: produces `NaN` for the entire row.

Empty molecule list:

- Individual batch: returns an empty `numpy.ndarray` (shape `(0,)`).
- Multi-descriptor batch: returns shape `(0, D)` where `D` is the number of
  descriptors requested.

Invalid SMILES:

- `Chem.MolFromSmiles('INVALID')` returns None.
- Pass None through to the batch API; it produces `NaN`.

Exception during descriptor computation:

- If a descriptor function throws an exception for a particular molecule,
  that entry is set to `NaN`. Other molecules are unaffected.
- This is handled inside the OpenMP loop via try/catch.

Invalid descriptor name in `CalcDescriptorsBatch`:

- Raises `ValueError` with a message indicating the unknown name.
- Example: `ValueError: Unknown descriptor name: 'CalcBogus'`.
  Call `GetBatchDescriptorNames()` for valid names.

Invalid shortcut string in `CalcDescriptorsBatch`:

- Raises `ValueError`. Only `"all"` is accepted as a string shortcut.

## Quick-Start Copy-Paste Examples

These are complete, self-contained scripts you can copy-paste and run.

### Example A: Compute LogP for a list of molecules

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCCCCCCC', 'INVALID']
mols = [Chem.MolFromSmiles(s) for s in smiles]

logp = rdMD.CalcClogP(mols)
for s, v in zip(smiles, logp):
    print(f'{s:15s} LogP = {v:8.4f}')
```

Expected output:

```text
CCO             LogP =  -0.0014
c1ccccc1        LogP =   1.6866
CC(=O)O         LogP =  -0.2682
CCCCCCCC        LogP =   3.4518
INVALID         LogP =      nan
```

### Example B: Compute all descriptors and save to CSV

```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'NCCCC', 'c1ccncc1']
mols = [Chem.MolFromSmiles(s) for s in smiles]

names = rdMD.GetBatchDescriptorNames()
data = rdMD.CalcDescriptorsBatch(mols, "all")

df = pd.DataFrame(data, columns=names)
df.insert(0, 'SMILES', smiles)
df.to_csv('descriptors.csv', index=False)
print(df.to_string())
```

### Example C: Pick specific descriptors

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
mols = [Chem.MolFromSmiles(s) for s in smiles]

result = rdMD.CalcDescriptorsBatch(
    mols, ["CalcExactMolWt", "CalcClogP", "CalcNumHBD", "CalcTPSA"]
)

print("Shape:", result.shape)   # (3, 4)
print("Row 0 (ethanol):", result[0])
print("Column 1 (ClogP for all):", result[:, 1])
```

### Example D: List available descriptor names

```python
from rdkit.Chem import rdMolDescriptors as rdMD

for name in rdMD.GetBatchDescriptorNames():
    print(name)
```

Expected output:

```text
CalcExactMolWt
CalcTPSA
CalcClogP
CalcMR
CalcNumHBD
CalcNumHBA
CalcNumRotatableBonds
CalcFractionCSP3
CalcLabuteASA
CalcNumHeavyAtoms
```

### Example E: Running tests

```bash
# From the repository root:
OMP_NUM_THREADS=4 python3 -m pytest \
    Code/GraphMol/Descriptors/Wrap/test_batch_descriptors.py -v

# Expected: 67 tests pass (52 individual batch + 15 multi-descriptor)
```
