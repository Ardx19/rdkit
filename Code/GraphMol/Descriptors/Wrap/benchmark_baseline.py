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
    target_size = 1000000
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
        print(f"Validation: FAIL (Length mismatch: {len(res_serial)} vs {len(res_batch)})")
    else:
        errors = 0
        for i, (v1, v2) in enumerate(zip(res_serial, res_batch)):
            if abs(v1 - v2) > 1e-4:
                print(f"Mismatch at index {i}: Serial={v1}, Batch={v2}")
                errors += 1
                if errors >= 10: 
                    print("... stopping validation output after 10 errors")
                    break
        
        if errors == 0:
            print(f"Validation: PASS (All {len(mols)} results match)")
        else:
            print(f"Validation: FAIL ({errors} mismatches found)")

    # ---------------------------------------------------------
    # TPSA BENCHMARK
    # ---------------------------------------------------------
    print("")
    print("--- TPSA BENCHMARK ---")
    
    # 1. Serial TPSA
    print("--- Serial TPSA ---")
    t0_tpsa = time.time()
    res_serial_tpsa = [rdMolDescriptors.CalcTPSA(m) for m in mols]
    t1_tpsa = time.time()
    serial_time_tpsa = t1_tpsa - t0_tpsa
    print(f"Time: {serial_time_tpsa:.4f} s")
    print(f"Per mol: {(serial_time_tpsa/len(mols))*1e6:.2f} µs")

    # 2. Batch TPSA
    print("--- Batch TPSA ---")
    t2_tpsa = time.time()
    res_batch_tpsa = rdMolDescriptors.CalcTPSA(mols) 
    t3_tpsa = time.time()
    batch_time_tpsa = t3_tpsa - t2_tpsa
    print(f"Time: {batch_time_tpsa:.4f} s")
    print(f"Per mol: {(batch_time_tpsa/len(mols))*1e6:.2f} µs")
    
    speedup_tpsa = serial_time_tpsa / batch_time_tpsa if batch_time_tpsa > 0 else 0
    print(f"TPSA Speedup: {speedup_tpsa:.2f}x")
    
    # Validation TPSA
    if len(res_serial_tpsa) != len(res_batch_tpsa):
        print(f"Validation: FAIL (Length mismatch)")
    else:
        max_diff = 0.0
        for v1, v2 in zip(res_serial_tpsa, res_batch_tpsa):
             d = abs(v1 - v2)
             if d > max_diff: max_diff = d
        
        if max_diff < 1e-4:
            print(f"Validation: PASS (Max diff: {max_diff:.6f})")
        else:
            print(f"Validation: FAIL (Max diff: {max_diff:.6f})")

if __name__ == "__main__":
    benchmark()
