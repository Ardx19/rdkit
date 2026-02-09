#!/bin/bash
# test_rdkit.sh - Run RDKit batch descriptor tests with benchmark

echo "=== Running RDKit Batch Descriptor Tests ==="

# Setup conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rdkit

# Clean PATH
unset PYTHONPATH
export PATH=/home/swarnavas/miniconda3/envs/rdkit/bin:$PATH

# Environment variables
export RDBASE=/home/swarnavas/Work/PhD_Work/Covaln_Dev_work/rdkit
export PYTHONPATH=$RDBASE:$PYTHONPATH
export LD_LIBRARY_PATH=$RDBASE/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Environment:"
echo "  RDBASE: $RDBASE"
echo "  Python: $(which python)"
echo ""

# Run CTest
echo "=== Running CTest ==="
cd $RDBASE/build
ctest -R pyBatchDescriptors --output-on-failure

echo ""
echo "=== Running Benchmark ==="
cd $RDBASE
python << 'PYBENCH'
import time
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD
import numpy as np

# Generate test molecules
smiles = ['C' * i for i in range(1, 101)]  # C, CC, CCC, ..., CCCCCCCCCC
mols = [Chem.MolFromSmiles(s) for s in smiles]
names = rdMD.GetBatchDescriptorNames()

print(f"Benchmark with {len(mols)} molecules, {len(names)} descriptors")
print("="*60)

# Benchmark 1: Simulated serial (one molecule at a time)
print("\n1. Simulated Serial Calculation")
start = time.time()
serial_results = []
for mol in mols:
    row = rdMD.CalcDescriptorsBatch([mol], "all")[0]
    serial_results.append(row)
serial_time = time.time() - start
print(f"   Time: {serial_time:.3f} seconds")
print(f"   Throughput: {len(mols)/serial_time:.1f} mol/s")

# Benchmark 2: Batch calculation
print("\n2. Batch Calculation (C++ with OpenMP)")
start = time.time()
batch_results = rdMD.CalcDescriptorsBatch(mols, "all")
batch_time = time.time() - start
print(f"   Time: {batch_time:.3f} seconds")
print(f"   Throughput: {len(mols)/batch_time:.1f} mol/s")

# Calculate speedup
speedup = serial_time / batch_time
print("\n" + "="*60)
print(f"SPEEDUP: {speedup:.1f}x faster with batch API")
print("="*60)

# Verify correctness
serial_array = np.array(serial_results)
match = np.allclose(batch_results, serial_array, rtol=1e-5, atol=1e-8)
print("\n3. Verification")
if match:
    print("   ✅ Batch results match serial calculations")
else:
    print("   ❌ Results mismatch!")

# Benchmark 3: Hybrid approach demonstration
print("\n4. Hybrid Approach Demonstration (Phase 2 Preview)")
print("   Showing how to combine C++ batch + Python descriptors")

# Python-only descriptors (for demonstration)
def NumValenceElectrons(mol):
    """Python implementation of NumValenceElectrons"""
    tbl = Chem.GetPeriodicTable()
    return sum(
        tbl.GetNOuterElecs(atom.GetAtomicNum()) - atom.GetFormalCharge() + atom.GetTotalNumHs()
        for atom in mol.GetAtoms()
    )

def NumRadicalElectrons(mol):
    """Python implementation of NumRadicalElectrons"""
    return sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())

python_descriptors = [
    ("NumValenceElectrons", NumValenceElectrons),
    ("NumRadicalElectrons", NumRadicalElectrons),
]

print(f"\n   Hybrid calculation ({len(names)} C++ batch + {len(python_descriptors)} Python)")
start = time.time()

# Step 1: Calculate C++ descriptors using batch API
cpp_results = rdMD.CalcDescriptorsBatch(mols, "all")

# Step 2: Calculate Python descriptors in a loop
python_results = []
for mol in mols:
    row = []
    for name, func in python_descriptors:
        try:
            row.append(func(mol))
        except:
            row.append(float('nan'))
    python_results.append(row)

hybrid_time = time.time() - start
print(f"   Time: {hybrid_time:.3f} seconds")
print(f"   Throughput: {len(mols)/hybrid_time:.1f} mol/s")

# Show combined results for first molecule
print(f"\n   Example results for first molecule (CC):")
print(f"   C++ descriptors (first 5): {cpp_results[0, :5]}")
print(f"   Python descriptors: {python_results[0]}")
print(f"   Total features: {len(names) + len(python_descriptors)}")

print("\n" + "="*60)
print(f"Phase 2 Hybrid approach ready: {len(names)} C++ + {len(python_descriptors)} Python")
print("="*60)

PYBENCH

echo ""
echo "=== Test Complete ==="
