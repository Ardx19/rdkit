#!/usr/bin/env python3
"""Benchmark all 10 batch descriptor APIs across varying OMP thread counts.

Usage:
    # Run from repo root — the wrapper script handles OMP_NUM_THREADS
    python3 Code/GraphMol/Descriptors/Wrap/benchmark_all_descriptors.py

    # Or run a single thread count directly:
    OMP_NUM_THREADS=4 python3 Code/GraphMol/Descriptors/Wrap/benchmark_all_descriptors.py --threads 4

    # Quick mode (100k molecules instead of 1M):
    python3 Code/GraphMol/Descriptors/Wrap/benchmark_all_descriptors.py --quick
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

from rdkit import Chem
from rdkit import DataStructs
import numpy as np
from rdkit.Chem import rdMolDescriptors as rdMD



def morgan_serial(mols):
    out = np.zeros((len(mols), 2048), dtype=np.uint8)
    for i, m in enumerate(mols):
        bv = rdMD.GetMorganFingerprintAsBitVect(m, 2, 2048)
        DataStructs.ConvertToNumpyArray(bv, out[i])
    return out

def maccs_serial(mols):
    out = np.zeros((len(mols), 167), dtype=np.uint8)
    for i, m in enumerate(mols):
        bv = rdMD.GetMACCSKeysFingerprint(m)
        DataStructs.ConvertToNumpyArray(bv, out[i])
    return out

def tt_serial(mols):
    out = np.zeros((len(mols), 2048), dtype=np.uint8)
    for i, m in enumerate(mols):
        bv = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(m)
        DataStructs.ConvertToNumpyArray(bv, out[i])
    return out

# ---------------------------------------------------------------------------

# Descriptor definitions
# ---------------------------------------------------------------------------
# Each entry: (name, serial_fn, batch_fn, validation_fn)
# serial_fn:     mol -> float
# batch_fn:      [mol] -> ndarray
# validation_fn: (serial_val, batch_val) -> bool  (True = match)

DESCRIPTORS = [
    (
        "CalcExactMolWt",
        lambda m: rdMD.CalcExactMolWt(m),
        lambda mols: rdMD.CalcExactMolWt(mols),
    ),
    (
        "CalcTPSA",
        lambda m: rdMD.CalcTPSA(m),
        lambda mols: rdMD.CalcTPSA(mols),
    ),
    (
        "CalcClogP",
        lambda m: rdMD.CalcCrippenDescriptors(m)[0],  # scalar uses tuple API
        lambda mols: rdMD.CalcClogP(mols),
    ),
    (
        "CalcMR",
        lambda m: rdMD.CalcCrippenDescriptors(m)[1],
        lambda mols: rdMD.CalcMR(mols),
    ),
    (
        "CalcNumHBD",
        lambda m: float(rdMD.CalcNumHBD(m)),
        lambda mols: rdMD.CalcNumHBD(mols),
    ),
    (
        "CalcNumHBA",
        lambda m: float(rdMD.CalcNumHBA(m)),
        lambda mols: rdMD.CalcNumHBA(mols),
    ),
    (
        "CalcNumRotatableBonds",
        lambda m: float(rdMD.CalcNumRotatableBonds(m)),
        lambda mols: rdMD.CalcNumRotatableBonds(mols),
    ),
    (
        "CalcFractionCSP3",
        lambda m: rdMD.CalcFractionCSP3(m),
        lambda mols: rdMD.CalcFractionCSP3(mols),
    ),
    (
        "CalcLabuteASA",
        lambda m: rdMD.CalcLabuteASA(m),
        lambda mols: rdMD.CalcLabuteASA(mols),
    ),
    (
        "CalcNumHeavyAtoms",
        lambda m: float(rdMD.CalcNumHeavyAtoms(m)),
        lambda mols: rdMD.CalcNumHeavyAtoms(mols),
    ),

    (
        "CalcNumAromaticRings",
        lambda m: float(rdMD.CalcNumAromaticRings(m)),
        lambda mols: rdMD.CalcNumAromaticRings(mols),
    ),
    (
        "CalcNumSaturatedRings",
        lambda m: float(rdMD.CalcNumSaturatedRings(m)),
        lambda mols: rdMD.CalcNumSaturatedRings(mols),
    ),
    (
        "CalcNumAliphaticRings",
        lambda m: float(rdMD.CalcNumAliphaticRings(m)),
        lambda mols: rdMD.CalcNumAliphaticRings(mols),
    ),
    (
        "CalcNumHeterocycles",
        lambda m: float(rdMD.CalcNumHeterocycles(m)),
        lambda mols: rdMD.CalcNumHeterocycles(mols),
    ),
    (
        "CalcNumAromaticHeterocycles",
        lambda m: float(rdMD.CalcNumAromaticHeterocycles(m)),
        lambda mols: rdMD.CalcNumAromaticHeterocycles(mols),
    ),
    (
        "CalcNumSaturatedHeterocycles",
        lambda m: float(rdMD.CalcNumSaturatedHeterocycles(m)),
        lambda mols: rdMD.CalcNumSaturatedHeterocycles(mols),
    ),
    (
        "CalcNumAliphaticHeterocycles",
        lambda m: float(rdMD.CalcNumAliphaticHeterocycles(m)),
        lambda mols: rdMD.CalcNumAliphaticHeterocycles(mols),
    ),
    (
        "CalcNumAromaticCarbocycles",
        lambda m: float(rdMD.CalcNumAromaticCarbocycles(m)),
        lambda mols: rdMD.CalcNumAromaticCarbocycles(mols),
    ),
    (
        "CalcNumSaturatedCarbocycles",
        lambda m: float(rdMD.CalcNumSaturatedCarbocycles(m)),
        lambda mols: rdMD.CalcNumSaturatedCarbocycles(mols),
    ),
    (
        "CalcNumAliphaticCarbocycles",
        lambda m: float(rdMD.CalcNumAliphaticCarbocycles(m)),
        lambda mols: rdMD.CalcNumAliphaticCarbocycles(mols),
    ),
    (
        "CalcNumHeteroatoms",
        lambda m: float(rdMD.CalcNumHeteroatoms(m)),
        lambda mols: rdMD.CalcNumHeteroatoms(mols),
    ),
    (
        "CalcNumAmideBonds",
        lambda m: float(rdMD.CalcNumAmideBonds(m)),
        lambda mols: rdMD.CalcNumAmideBonds(mols),
    ),
    (
        "CalcNumAtoms",
        lambda m: float(rdMD.CalcNumAtoms(m)),
        lambda mols: rdMD.CalcNumAtoms(mols),
    ),
    (
        "CalcNumSpiroAtoms",
        lambda m: float(rdMD.CalcNumSpiroAtoms(m)),
        lambda mols: rdMD.CalcNumSpiroAtoms(mols),
    ),
    (
        "CalcNumBridgeheadAtoms",
        lambda m: float(rdMD.CalcNumBridgeheadAtoms(m)),
        lambda mols: rdMD.CalcNumBridgeheadAtoms(mols),
    ),
    (
        "_CalcMolWt",
        lambda m: rdMD._CalcMolWt(m),
        lambda mols: rdMD._CalcMolWt(mols),
    ),
    (
        "CalcKappa1",
        lambda m: rdMD.CalcKappa1(m),
        lambda mols: rdMD.CalcKappa1(mols),
    ),
    (
        "CalcKappa2",
        lambda m: rdMD.CalcKappa2(m),
        lambda mols: rdMD.CalcKappa2(mols),
    ),
    (
        "CalcKappa3",
        lambda m: rdMD.CalcKappa3(m),
        lambda mols: rdMD.CalcKappa3(mols),
    ),
    (
        "CalcChi0v",
        lambda m: rdMD.CalcChi0v(m),
        lambda mols: rdMD.CalcChi0v(mols),
    ),
    (
        "CalcChi1v",
        lambda m: rdMD.CalcChi1v(m),
        lambda mols: rdMD.CalcChi1v(mols),
    ),
    (
        "CalcChi2v",
        lambda m: rdMD.CalcChi2v(m),
        lambda mols: rdMD.CalcChi2v(mols),
    ),
    (
        "CalcChi3v",
        lambda m: rdMD.CalcChi3v(m),
        lambda mols: rdMD.CalcChi3v(mols),
    ),
    (
        "CalcChi4v",
        lambda m: rdMD.CalcChi4v(m),
        lambda mols: rdMD.CalcChi4v(mols),
    ),
    (
        "CalcChi0n",
        lambda m: rdMD.CalcChi0n(m),
        lambda mols: rdMD.CalcChi0n(mols),
    ),
    (
        "CalcChi1n",
        lambda m: rdMD.CalcChi1n(m),
        lambda mols: rdMD.CalcChi1n(mols),
    ),
    (
        "CalcChi2n",
        lambda m: rdMD.CalcChi2n(m),
        lambda mols: rdMD.CalcChi2n(mols),
    ),
    (
        "CalcChi3n",
        lambda m: rdMD.CalcChi3n(m),
        lambda mols: rdMD.CalcChi3n(mols),
    ),
    (
        "CalcChi4n",
        lambda m: rdMD.CalcChi4n(m),
        lambda mols: rdMD.CalcChi4n(mols),
    ),
    (
        "CalcHallKierAlpha",
        lambda m: rdMD.CalcHallKierAlpha(m),
        lambda mols: rdMD.CalcHallKierAlpha(mols),
    ),
    (
        "GetMorganFingerprintAsBitVect",
        morgan_serial,
        lambda mols: rdMD.GetMorganFingerprintAsBitVect(mols, 2, 2048),
        False, True
    ),
    (
        "GetMACCSKeysFingerprint",
        maccs_serial,
        lambda mols: rdMD.GetMACCSKeysFingerprint(mols),
        False, True
    ),
    (
        "GetHashedTopologicalTorsionFingerprintAsBitVect",
        tt_serial,
        lambda mols: rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(mols),
        False, True
    ),
]



def load_mols(target_size):
    """Load molecules from SMILES to ensure unique C++ ROMol pointers.
    This prevents the OpenMP thread-safety mechanism from triggering deep copies
    for duplicated pointers, accurately reflecting real-world parsing pipelines.
    """
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "test_data", "PBF_egfr.sdf"),
        os.path.join(
            os.environ.get("RDBASE", ""),
            "Code", "GraphMol", "Descriptors", "test_data", "PBF_egfr.sdf",
        ),
    ]
    sdf_path = None
    for p in paths:
        if os.path.exists(p):
            sdf_path = p
            break
    if not sdf_path:
        print("Error: PBF_egfr.sdf not found")
        sys.exit(1)

    suppl = Chem.SDMolSupplier(sdf_path)
    base = [m for m in suppl if m is not None]
    if not base:
        raise RuntimeError(f"No molecules loaded from {sdf_path}")

    smiles = [Chem.MolToSmiles(m) for m in base]
    smiles_pool = (smiles * (target_size // len(smiles) + 1))[:target_size]
    
    # print(f"Parsing {target_size} unique SMILES strings...")
    mols = [Chem.MolFromSmiles(s) for s in smiles_pool]
    return [m for m in mols if m is not None]


def benchmark_single_descriptor(name, serial_fn, batch_fn, mols, skip_validation=False, serial_takes_list=False):
    """Benchmark one descriptor. Returns dict with timing results."""
    n = len(mols)

    # --- Warm-up ---
    # Warms up JIT, OpenMP thread pool, and NumPy allocator
    _ = batch_fn(mols[:100])

    # --- Batch (timed, 3 runs, take best) ---
    batch_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        res_batch = batch_fn(mols)
        batch_times.append(time.perf_counter() - t0)
    batch_time = min(batch_times)

    # --- Validation & Serial ---
    max_diff = 0.0
    mismatches = 0
    serial_time = 0.0
    
    if not skip_validation:
        t0 = time.perf_counter()
        if serial_takes_list:
            res_serial = serial_fn(mols)
        else:
            res_serial = [serial_fn(m) for m in mols]
        serial_time = time.perf_counter() - t0
        
        if serial_takes_list:
            mismatches = int(np.sum(res_serial != res_batch))
            if mismatches > 0:
                max_diff = float(np.max(np.abs(res_serial.astype(int) - res_batch.astype(int))))
        else:
            for s, b in zip(res_serial, res_batch):
                d = abs(s - b)
                if d > max_diff:
                    max_diff = d
                if d > 1e-4:
                    mismatches += 1

    speedup = serial_time / batch_time if (batch_time > 0 and not skip_validation) else 0.0

    return {
        "name": name,
        "serial_time": serial_time,
        "batch_time": batch_time,
        "batch_times_all": batch_times,
        "speedup": speedup,
        "max_diff": max_diff,
        "mismatches": mismatches,
        "n_mols": n,
        "valid": mismatches == 0 if not skip_validation else True,
        "validated": not skip_validation
    }

def run_benchmarks(mols, thread_count, skip_validation=False):
    """Run all descriptor benchmarks, return list of result dicts."""
    omp = os.environ.get("OMP_NUM_THREADS", "?")
    # print(f"=== OMP_NUM_THREADS={omp} (requested: {thread_count}) ===")
    # print(f"Molecules: {len(mols):,}")
    # print()

    results = []
    for name, serial_fn, batch_fn, *rest in DESCRIPTORS:
        entry_skip_val = rest[0] if rest else False
        serial_takes_list = rest[1] if len(rest) > 1 else False
        r = benchmark_single_descriptor(name, serial_fn, batch_fn, mols,
                                        skip_validation or entry_skip_val,
                                        serial_takes_list)
        status = "PASS" if r["valid"] else "FAIL"
        if skip_validation: status = "SKIP_VAL"
        
        speedup_str = f"{r['speedup']:.2f}x" if not skip_validation else "N/A"
        serial_str = f"{r['serial_time']:.3f}s" if not skip_validation else "N/A"
        
        # We output this to be captured by the main process
        print(
            f"  {name:<25s}  serial={serial_str:<8s}  "
            f"batch={r['batch_time']:.3f}s  "
            f"speedup={speedup_str:<8s}  [{status}]"
        )
        results.append(r)
    print()
    return results

def print_summary_table(all_results, thread_counts):
    """Print a formatted summary table across all thread counts."""
    desc_names = [entry[0] for entry in DESCRIPTORS]

    # Header
    sep = "=" * 120
    print(sep)
    print("  Summary Table: Batch Speedup vs Serial Python Loop")
    print(sep)
    print()

    # Column headers
    hdr = f"  {'Descriptor':<25s}"
    for tc in thread_counts:
        hdr += f" | {tc:>2d} thr (time)  spdup"
    print(hdr)
    print("  " + "-" * 25 + ("-+" + "-" * 21) * len(thread_counts))

    for desc_name in desc_names:
        row = f"  {desc_name:<25s}"
        for tc_idx, tc in enumerate(thread_counts):
            r = all_results[tc_idx]
            match = [x for x in r if x["name"] == desc_name]
            if match:
                m = match[0]
                row += f" | {m['batch_time']:>6.3f}s  {m['speedup']:>6.2f}x"
            else:
                row += f" |       -        -   "
        print(row)

    print()

    # Also print serial baseline times (from first run — they're similar across runs)
    print("  Serial baseline times (Python loop, from first thread-count run):")
    first_run = all_results[0]
    for r in first_run:
        print(f"    {r['name']:<25s}  {r['serial_time']:.3f}s")

    print()
    print(sep)
    print("  Notes:")
    print('  - "Serial" = Python list comprehension (one C++ call per mol, GIL held)')
    print('  - "Batch"  = Single C++ call, GIL released, OpenMP parallel for schedule(dynamic)')
    print("  - Speedup = Serial time / Best-of-3 batch time")
    print("  - All batch APIs return numpy.ndarray (dtype=float64)")
    print(f"  - Dataset: PBF_egfr.sdf replicated to {first_run[0]['n_mols']:,} molecules")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Benchmark all batch descriptors")
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Run only this thread count (for subprocess mode)",
    )
    parser.add_argument(
        "--scale", type=int, default=10000,
        help="Number of molecules to benchmark (e.g. 1000, 10000, 100000, 1000000). Default: 10000",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip the serial python loop correctness validation. Highly recommended for >100k molecules.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save raw JSON results to this file",
    )
    # kept for backwards compatibility but hidden
    parser.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    target_size = args.scale
    if args.quick:
        target_size = 100_000

    thread_counts = [1, 2, 4, 6]

    if args.threads is not None:
        # Single-thread-count mode (called by parent process)
        mols = load_mols(target_size)
        results = run_benchmarks(mols, args.threads, args.no_validate)
        # Output JSON to stdout for the parent to collect
        print("__JSON_START__")
        print(json.dumps(results))
        print("__JSON_END__")
        return

    # Multi-thread-count orchestrator mode
    print("=" * 120)
    print(f"  RDKit Batch Descriptor Benchmark — All {len(DESCRIPTORS)} Descriptors")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dataset: {target_size:,} molecules (PBF_egfr.sdf re-parsed to ensure unique pointers)")
    print(f"  Thread counts: {thread_counts}")
    print(f"  Validation: {'SKIPPED' if args.no_validate else 'ENABLED (calculating serial base times)'}")
    print(f"  Build: -DRDK_BUILD_OPENMP=ON, schedule(dynamic)")
    print(f"  .so: {rdMD.__file__}")
    print("=" * 120)
    print()

    all_results = []
    script = os.path.abspath(__file__)

    for tc in thread_counts:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(tc)
        cmd = [
            sys.executable, script,
            "--threads", str(tc),
            "--scale", str(target_size),
        ]
        if args.no_validate:
            cmd.append("--no-validate")

        print(f"=== OMP_NUM_THREADS={tc} ===")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Print the subprocess stdout (the per-descriptor lines)
        for line in proc.stdout.splitlines():
            if line.startswith("__JSON_"):
                continue
            print(line)

        if proc.returncode != 0:
            print(f"ERROR: subprocess for {tc} threads failed:")
            print(proc.stderr)
            sys.exit(1)

        # Extract JSON
        lines = proc.stdout.splitlines()
        json_lines = []
        capture = False
        for line in lines:
            if line.strip() == "__JSON_START__":
                capture = True
                continue
            if line.strip() == "__JSON_END__":
                capture = False
                continue
            if capture:
                json_lines.append(line)

        results = json.loads("\n".join(json_lines))
        all_results.append(results)

    # Print summary
    print()
    print_summary_table(all_results, thread_counts)

    # Save JSON if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "thread_counts": thread_counts,
                    "n_mols": target_size,
                    "results": {
                        str(tc): all_results[i]
                        for i, tc in enumerate(thread_counts)
                    },
                },
                f,
                indent=2,
            )
        print(f"\nRaw results saved to {args.output}")


if __name__ == "__main__":
    main()
