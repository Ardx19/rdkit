import time
import os
import sys
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def benchmark():
    # 1. Load Data
    # Try different paths to find the test data
    paths_to_try = [
        os.path.join(os.path.dirname(__file__), "..", "test_data", "PBF_egfr.sdf"),
        os.path.join(os.environ.get('RDBASE', ''), "Code", "GraphMol", "Descriptors", "test_data", "PBF_egfr.sdf"),
        "../Code/GraphMol/Descriptors/test_data/PBF_egfr.sdf"
    ]
    
    sdf_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            sdf_path = p
            break
            
    if not sdf_path:
        print(f"Error: PBF_egfr.sdf not found in tried paths.")
        sys.exit(1)
    
    print(f"RDKit imported from: {rdMolDescriptors.__file__}")
    print(f"Loading molecules from {sdf_path}...")
    suppl = Chem.SDMolSupplier(sdf_path)
    mols = [m for m in suppl if m is not None]
    
    # Scale up molecules
    target_size = 10000
    while len(mols) < target_size:
        mols.extend(mols)
    mols = mols[:target_size]
    
    print(f"Benchmarking on {len(mols)} molecules.")
    print("")

    # 2. Measure Baseline (Serial)
    print("--- Serial Loop (Standard) ---")
    t0 = time.time()
    res_serial = [rdMolDescriptors.CalcExactMolWt(m) for m in mols]
    t1 = time.time()
    serial_time = t1 - t0
    print(f"Time: {serial_time:.4f} s")
    print(f"Per mol: {(serial_time/len(mols))*1e6:.2f} µs")
    print("")
    
    # 3. Measure Batch (Parallel)
    print("--- Batch Parallel (New) ---")
    t2 = time.time()
    res_batch = rdMolDescriptors.CalcExactMolWt(mols) 
    t3 = time.time()
    batch_time = t3 - t2
    print(f"Time: {batch_time:.4f} s")
    print(f"Per mol: {(batch_time/len(mols))*1e6:.2f} µs")
    
    speedup = serial_time / batch_time if batch_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")
    
    # Validation
    if len(res_serial) != len(res_batch):
        print("Validation: FAIL (Length mismatch)")
    elif abs(res_serial[0] - res_batch[0]) > 1e-4:
        print(f"Validation: FAIL (Value mismatch)")
    else:
        print("Validation: PASS (Results match)")

if __name__ == "__main__":
    benchmark()
