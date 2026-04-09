# RDKit Descriptor Parallelization Project

This note covers only the newer descriptor-parallelization work added in `2026` for `Code/GraphMol/Descriptors/Wrap/`.

It is based on the git history for the relevant files in:

- `Code/GraphMol/Descriptors/Wrap/BatchUtils.h`
- `Code/GraphMol/Descriptors/Wrap/rdMolDescriptors.cpp`
- `Code/GraphMol/Descriptors/Wrap/test_batch_descriptors.py`
- `Code/GraphMol/Descriptors/Wrap/benchmark_baseline.py`
- `Code/GraphMol/Descriptors/Wrap/benchmark_all_descriptors.py`
- `Code/GraphMol/Descriptors/Wrap/CMakeLists.txt`
- `CMakeLists.txt`
- `.github/workflows/smoke_test.yml`

## Scope

The goal of this project was to move descriptor computation over lists of molecules from Python loops into C++, release the GIL, and parallelize the work with OpenMP while preserving the existing scalar Python API.

The project grew in three practical stages:

1. Prove the idea on a very small surface area.
2. Make the implementation safe, testable, and actually multi-threaded in the build.
3. Expand the API surface and benchmarking harness to something useful for real workloads.

## Executive Summary

From February to March 2026, this project added batch descriptor overloads to `rdkit.Chem.rdMolDescriptors`, introduced reusable batch infrastructure in `BatchUtils.h`, enabled OpenMP at the build-system level, added dedicated validation tests, switched batch results from Python lists to `numpy.ndarray`, added a multi-descriptor batch API, fixed a duplicate-pointer safety bug, and expanded the batchable Phase 1 descriptor set from `2` descriptors to `40`.

There are two distinct committed states to keep in mind:

- `origin/master` currently includes the work through `2026-02-08`, which means the pushed GitHub state has the initial `10`-descriptor batch API plus the duplicate-pointer safety fix.
- The local branch continues through `2026-03-17`, where the API expands to `40` descriptors and the benchmark/test harness is improved further.

## Timeline

### 1. 2026-02-04 — `27eccb9a4` — Project bootstrap with `CalcExactMolWt`

Files changed:

- `Code/GraphMol/Descriptors/Wrap/BatchUtils.h`
- `Code/GraphMol/Descriptors/Wrap/rdMolDescriptors.cpp`
- `Code/GraphMol/Descriptors/Wrap/benchmark_baseline.py`

What changed:

- Added `BatchUtils.h` as the reusable C++ batch execution layer.
- Added `extractMolPtrs(...)` to convert a Python list of molecules into raw `ROMol*` pointers before entering the compute loop.
- Added `runBatch<T>(...)` to run the descriptor loop in C++ with GIL release and optional OpenMP parallelism.
- Added the first batch overload: `CalcExactMolWt(mols, onlyHeavy=False)`.
- Added `benchmark_baseline.py` to compare scalar Python looping with the new batch call.

Reasoning:

This was the proof-of-concept stage. The first problem to solve was not API breadth, but architecture. The project needed to answer three questions early:

- Can list inputs be handled without breaking the existing scalar API?
- Can the Python loop be moved behind the wrapper boundary cleanly?
- Is there enough performance upside to justify continuing?

`extractMolPtrs(...)` was the key safety boundary. Python objects cannot be touched safely inside worker threads without the GIL, so the wrapper first converts the input list into plain C++ pointers while the GIL is still held. After that, the compute loop can run on C++ data only.

Important nuance:

`BatchUtils.h` already contained `_OPENMP` guards at this point, but the build system had not yet been updated to propagate `-fopenmp`. So this commit established the batch architecture and the wrapper pattern, but it did not yet guarantee real OpenMP execution in every build.

### 2. 2026-02-06 — `3f555d3d1` — Add `CalcTPSA` batch support

Files changed:

- `Code/GraphMol/Descriptors/Wrap/rdMolDescriptors.cpp`
- `Code/GraphMol/Descriptors/Wrap/benchmark_baseline.py`

What changed:

- Added the second batch overload: `CalcTPSA(mols, includeSandP=False)`.
- Extended `benchmark_baseline.py` to benchmark TPSA in the same serial-vs-batch style as ExactMolWt.

Reasoning:

This was the first confirmation that the design generalized beyond a single descriptor. `CalcTPSA` was a good second target because it has a real API option (`includeSandP`) and therefore exercised parameter passing through the new batch lambda path. It also validated that the wrapper can preserve scalar semantics while exposing the same Python function name for list inputs.

### 3. 2026-02-06 — `240059995` — Add dedicated CI-visible validation tests

Files changed:

- `Code/GraphMol/Descriptors/Wrap/CMakeLists.txt`
- `Code/GraphMol/Descriptors/Wrap/test_batch_descriptors.py`

What changed:

- Added a dedicated batch test file: `test_batch_descriptors.py`.
- Registered it in CTest as `pyBatchDescriptors`.
- Covered correctness vs scalar, empty inputs, `None -> NaN`, return type, and determinism.

Reasoning:

Once the wrapper pattern existed for more than one descriptor, correctness became more important than raw speed. The purpose of this step was to prevent regressions before broadening the API surface. The determinism checks were especially important because the code path was designed to run under OpenMP, which makes race-condition bugs harder to spot if there is no explicit repeated-call validation.

This commit also established the project's long-term testing philosophy: CI should validate correctness and stability, not enforce speed thresholds.

### 4. 2026-02-08 — `36cc509f1` — Enable real OpenMP execution in the build system

Files changed:

- `CMakeLists.txt`
- `.github/workflows/smoke_test.yml`
- `Code/GraphMol/Descriptors/Wrap/benchmarking/output.txt`

What changed:

- Added `RDK_BUILD_OPENMP` to the top-level CMake configuration.
- Linked `OpenMP::OpenMP_CXX` through `rdkit_base` so that OpenMP flags propagated to downstream targets, including the Python wrapper extension.
- Updated CI to build with `-DRDK_BUILD_OPENMP=ON`.
- Saved benchmark output showing actual scaling under different `OMP_NUM_THREADS` values.

Reasoning:

This was the commit that turned the design into actual multi-core behavior. Before this point, the project had the right source structure, but not the full build plumbing. Without `-fopenmp`, the compiler never activated the OpenMP path guarded by `_OPENMP`, so the code could still run effectively as a single-threaded C++ loop.

The reasoning here was straightforward: if the project claims OpenMP parallelization, that capability must be enabled centrally in the build, not assumed from local compiler behavior.

This is also the point where the benchmark story became more trustworthy. After this commit, speedups were no longer just a result of moving the loop from Python into C++; they also reflected true OpenMP parallel execution.

### 5. 2026-02-08 — `2a4561d89` — Expand from `2` to `10` descriptors, switch to NumPy, add multi-descriptor batching

Files changed:

- `Code/GraphMol/Descriptors/Wrap/rdMolDescriptors.cpp`
- `Code/GraphMol/Descriptors/Wrap/test_batch_descriptors.py`
- `Code/GraphMol/Descriptors/Wrap/benchmark_all_descriptors.py`
- `Code/GraphMol/Descriptors/Wrap/benchmarking/README.md`
- `Code/GraphMol/Descriptors/Wrap/benchmarking/NEW_BATCH_API_REFERENCE.txt`
- `Code/GraphMol/Descriptors/Wrap/benchmarking/benchmark_all_output.txt`
- `Code/GraphMol/Descriptors/Wrap/benchmarking/benchmark_all_results.json`

What changed:

- Expanded the batchable set from `2` descriptors to `10`.
- Added batch support for:
  - `CalcClogP`
  - `CalcMR`
  - `CalcNumHBD`
  - `CalcNumHBA`
  - `CalcNumRotatableBonds`
  - `CalcFractionCSP3`
  - `CalcLabuteASA`
  - `CalcNumHeavyAtoms`
- Added new scalar helpers `CalcClogP(mol)` and `CalcMR(mol)` so the Crippen pair could be accessed individually.
- Switched batch return values from Python lists to `numpy.ndarray` with `dtype=float64`.
- Added `CalcDescriptorsBatch(mols, descriptors)`.
- Added `GetBatchDescriptorNames()`.
- Expanded tests and added user-facing docs and benchmark artifacts.

Reasoning:

This was the biggest conceptual step in the project.

The first reason for this change was API usefulness. A batch API with only ExactMolWt and TPSA proves the mechanism, but it is too limited for real RDKit descriptor workflows.

The second reason was the Python object bottleneck. Returning Python lists scales poorly because every result becomes a heap-allocated Python object created under the GIL. For large batches, that serial Python object construction becomes the next bottleneck even if the descriptor math itself is parallel. Switching to `numpy.ndarray` solved that problem by replacing millions of Python object allocations with one contiguous allocation plus a bulk memory copy.

The third reason was workflow efficiency. `CalcDescriptorsBatch(...)` made it possible to compute several descriptors in one high-level Python call, which is much closer to how descriptor generation is used in machine learning and screening pipelines.

At this point the pushed GitHub state became a coherent Phase 1 product:

- `10` batch overloads
- `numpy.ndarray` outputs
- a registry-driven multi-descriptor API
- correctness tests in CI
- documentation and benchmark scripts

### 6. 2026-02-08 — `58c62c05a` — Fix duplicate-pointer OpenMP safety bug

Files changed:

- `Code/GraphMol/Descriptors/Wrap/BatchUtils.h`
- `Code/GraphMol/Descriptors/Wrap/rdMolDescriptors.cpp`

What changed:

- Reworked `extractMolPtrs(...)` to return a `MolBatch` structure instead of only a pointer vector.
- Added deep-copy handling for duplicate `ROMol*` references in the same input list.
- Preserved ownership of copied molecules until batch execution completes.

Reasoning:

This was the key safety fix of the first pushed phase.

The bug was subtle: if the same Python molecule object appeared multiple times in the input list, different OpenMP threads could end up calling descriptor functions against the same underlying `ROMol`. Some RDKit descriptors lazily initialize internal state or cache properties even through `const` references, so shared object reuse can still produce data races or heap corruption.

The solution was to deep-copy only duplicate references, not every molecule. That preserved correctness without destroying the performance model for normal inputs.

This commit is currently the tip of `origin/master`, so it is the last descriptor-parallelization change that is definitely on GitHub right now.

### 7. 2026-03-17 — `589e317ca` — Expand Phase 1 with `15` count and ring descriptors

Files changed:

- `Code/GraphMol/Descriptors/Wrap/rdMolDescriptors.cpp`
- `Code/GraphMol/Descriptors/Wrap/test_batch_descriptors.py`
- `Code/GraphMol/Descriptors/Wrap/benchmark_all_descriptors.py`

What changed:

- Added batch overloads for:
  - `CalcNumAromaticRings`
  - `CalcNumSaturatedRings`
  - `CalcNumAliphaticRings`
  - `CalcNumHeterocycles`
  - `CalcNumAromaticHeterocycles`
  - `CalcNumSaturatedHeterocycles`
  - `CalcNumAliphaticHeterocycles`
  - `CalcNumAromaticCarbocycles`
  - `CalcNumSaturatedCarbocycles`
  - `CalcNumAliphaticCarbocycles`
  - `CalcNumHeteroatoms`
  - `CalcNumAmideBonds`
  - `CalcNumAtoms`
  - `CalcNumSpiroAtoms`
  - `CalcNumBridgeheadAtoms`
- Extended tests and benchmarking coverage accordingly.

Reasoning:

This was the first major Phase 1 expansion beyond the original `10` descriptors. The reasoning was strong: these descriptors are mathematically simple scalar outputs and fit the existing `runBatch<double>` pattern almost perfectly. That means they provide large API coverage for relatively low implementation complexity.

This step reflects the project's stated Phase 1 philosophy well: maximize practical descriptor coverage first using the existing scalar-like template path, before taking on more complex vector-valued or fingerprint outputs.

### 8. 2026-03-17 — `6468dd61d` — Complete the local `40`-descriptor Phase 1 set and harden benchmarking

Files changed:

- `Code/GraphMol/Descriptors/Wrap/rdMolDescriptors.cpp`
- `Code/GraphMol/Descriptors/Wrap/test_batch_descriptors.py`
- `Code/GraphMol/Descriptors/Wrap/benchmark_all_descriptors.py`
- `Code/GraphMol/Descriptors/Wrap/benchmarking/README.md`

What changed:

- Added the remaining `15` scalar-like Phase 1 descriptors locally:
  - `_CalcMolWt`
  - `CalcKappa1`
  - `CalcKappa2`
  - `CalcKappa3`
  - `CalcChi0v`
  - `CalcChi1v`
  - `CalcChi2v`
  - `CalcChi3v`
  - `CalcChi4v`
  - `CalcChi0n`
  - `CalcChi1n`
  - `CalcChi2n`
  - `CalcChi3n`
  - `CalcChi4n`
  - `CalcHallKierAlpha`
- Changed benchmark molecule loading to re-parse SMILES strings and therefore produce unique underlying `ROMol` pointers.
- Added `--scale` and `--no-validate` to `benchmark_all_descriptors.py`.
- Fixed tests that still hard-coded the earlier `10`-descriptor count.

Reasoning:

This commit completed the current local `40`-descriptor Phase 1 set.

The benchmarking change was just as important as the API expansion. Once duplicate references were made safe through deep-copying, a naive benchmark that reused the same underlying molecule pointers would start measuring safety-copy overhead as much as descriptor throughput. That would distort performance conclusions. Re-parsing SMILES to create unique C++ molecule objects made the benchmark closer to real data-ingestion pipelines and much fairer for evaluating OpenMP scaling.

The addition of `--scale` and `--no-validate` also split benchmark usage into two clearer modes:

- small or medium validated runs for correctness-plus-performance
- large server runs where correctness has already been established and timing is the primary goal

### 9. 2026-03-17 — `e29c43cb2` — Strengthen test correctness around the expanded local API

Files changed:

- `Code/GraphMol/Descriptors/Wrap/test_batch_descriptors.py`

What changed:

- Updated tests to reflect the `40`-descriptor registry.
- Changed `_load_mols(...)` in the tests to re-parse SMILES so the test inputs use unique underlying C++ molecule pointers.
- Replaced a silent `continue` path in `test_all_descriptors_correctness` with an explicit `hasattr(...)` assertion.
- Tightened float checking for newly added scalar-like descriptors.

Reasoning:

This was a correctness hardening pass after the API surface grew substantially. The reasoning was to prevent two classes of subtle failures:

- false confidence from tests that silently skip missing Python bindings
- misleading behavior caused by test fixtures that reuse the same underlying `ROMol` pointer too often

This commit also made the local test suite better aligned with how the benchmark harness was already being corrected: unique-pointer inputs are safer and clearer when validating multi-threaded batch execution.

### 10. 2026-03-17 — `d0906dabb` — Fix benchmark JSON loader bug

Files changed:

- `Code/GraphMol/Descriptors/Wrap/benchmark_all_descriptors.py`

What changed:

- Fixed an unterminated string literal bug in the multi-process JSON reader.

Reasoning:

This was a reliability fix. Once the benchmark script started using subprocesses for different thread-count runs and collecting JSON back from stdout, the orchestration logic itself became part of the measurement pipeline. A string parsing bug here could invalidate the whole benchmark run even if the descriptor kernels were correct.

### 11. 2026-04-07 — `runBatchToNumpy` C++ Boilerplate Refactor

Files changed:

- `Code/GraphMol/Descriptors/Wrap/rdMolDescriptors.cpp`

What changed:

- Introduced a generic `runBatchToNumpy` template helper in the anonymous namespace of `rdMolDescriptors.cpp`.
- Refactored all 40 `Calc..._List` wrapper functions to use this single helper via C++ lambda closures.
- Eliminated over 300 lines of repetitive NumPy C-API allocation and memory copying boilerplate.

Reasoning:

As the Phase 1 descriptor surface expanded to 40 functions, `rdMolDescriptors.cpp` accumulated massive amounts of duplicated boilerplate (extracting pointers, running the batch, allocating a NumPy array, and copying memory). This duplication was a breeding ground for future bugs and made the codebase unnecessarily bloated.

By collapsing the 3-step pattern into a centralized `runBatchToNumpy` helper, the wrapper functions were reduced to clean, 3-line lambda closures that preserve individual parameter capturing (e.g., `onlyHeavy`, `includeSandP`) without duplicating array management. This drastically improves maintainability and sets a clean foundation for Phase 2 (Vector Descriptors) which will require similar array abstractions.

## Observed Results by Stage

### Early proof-of-concept stage

The first stage established that moving the loop into C++ was worthwhile even before the project broadened its API surface. The exact numbers varied across runs and environments, but the important result was that batch execution already matched scalar correctness while reducing Python-side overhead.

### OpenMP-enabled benchmark milestone on 2026-02-08

The saved `benchmarking/output.txt` artifact captures the first fully OpenMP-enabled benchmark milestone on `1,000,000` molecules.

Key results recorded there:

- `CalcExactMolWt`: about `2.7s` serial vs `0.64s` at `4` threads and `0.57s` at `6` threads
- `CalcTPSA`: about `3.1s` serial vs `0.80s` at `4` threads and `0.64s` at `6` threads
- all benchmarked outputs validated successfully against scalar execution

Why this mattered:

This was the first strong evidence that the project was no longer just removing Python loop overhead. After the build-system change, the benchmark results showed real OpenMP scaling on top of the wrapper-level optimization.

### Expanded local benchmark milestone on 2026-03-17

The saved `benchmarking/benchmark_all_output.txt` artifact captures the later local benchmark state for the expanded `40`-descriptor API on `10,000` molecules across `1`, `2`, `4`, and `6` threads.

The most important outcomes from that artifact are:

- every descriptor in the run validated with `[PASS]`
- heavier descriptors scaled well as thread count increased
- examples of strong scaling include `CalcNumRotatableBonds`, `CalcKappa3`, and several `Chi` descriptors

Why this mattered:

The project had moved from a narrow proof-of-concept into a broader Phase 1 descriptor suite, and the benchmark harness was now measuring a much more representative cross-section of descriptor types instead of only the initial `2` functions.

### Validation growth over time

The validation story also grew in clear steps:

- the initial February test suite covered the first `2` descriptors and the basic safety cases
- the `2026-02-08` expansion commit explicitly recorded `67` tests for the `10`-descriptor milestone
- the saved `benchmarking/validation_results.txt` artifact records `70` passing tests for the later local `40`-descriptor state

Why this mattered:

The project did not just grow the API surface. It also grew its correctness guarantees, which is what made the later expansion credible.

## What the Project Has Achieved

As of the committed 2026 work on the local branch, the descriptor-parallelization project has delivered:

- Reusable batch execution infrastructure in `BatchUtils.h`
- GIL-safe extraction of molecule pointers before parallel execution
- Batch overloads on the same Python function names used for scalar descriptors
- OpenMP-backed execution enabled through the build system
- `numpy.ndarray` outputs for batch overloads
- `NaN` handling for `None` or failed molecules
- A registry-driven `CalcDescriptorsBatch(...)` API
- `GetBatchDescriptorNames()` for discovery and registry-order validation
- A committed local Phase 1 surface of `40` batch descriptors
- Benchmark scripts for both simple and expanded profiling runs
- Dedicated correctness tests for the batch APIs
- A duplicate-pointer safety fix for repeated molecule references

## Current Descriptor Count by Milestone

The descriptor-parallelization work progressed in three waves:

1. First wave, `2` descriptors on `2026-02-06`:
`CalcExactMolWt`, `CalcTPSA`

2. Second wave, `10` descriptors on `2026-02-08`:
`CalcExactMolWt`, `CalcTPSA`, `CalcClogP`, `CalcMR`, `CalcNumHBD`, `CalcNumHBA`, `CalcNumRotatableBonds`, `CalcFractionCSP3`, `CalcLabuteASA`, `CalcNumHeavyAtoms`

3. Third wave, `40` descriptors on the local branch by `2026-03-17`:
the original `10`, plus `15` count/ring descriptors, plus `15` Kappa/Chi/HallKier/average-mass descriptors.

## GitHub State vs Local State

This distinction matters because the descriptor-parallelization work has moved beyond what is currently pushed.

`origin/master` currently includes:

- batch infrastructure
- OpenMP build support
- the initial `10` descriptor batch API
- `CalcDescriptorsBatch(...)`
- `GetBatchDescriptorNames()`
- duplicate-pointer safety fix

The local branch additionally includes:

- the expansion from `10` to `40` descriptors
- stronger test coverage for the larger registry
- a more realistic benchmark loader using unique pointers
- `--scale` and `--no-validate` benchmark modes
- the benchmark JSON loader fix

In other words, GitHub currently reflects the first stable public milestone, while the local branch reflects the broader committed Phase 1 expansion.

## Why the Major Design Decisions Were Correct

### Keep the scalar API unchanged

Using the same Python function name for single molecules and lists preserved backward compatibility and made adoption easy. Existing code keeps working, and new code gets batch execution by passing a list.

### Extract pointers before entering the parallel loop

This was required for correctness, not just optimization. Python objects cannot be safely manipulated inside OpenMP worker threads without the GIL.

### Return `numpy.ndarray` instead of Python lists

This removed a large serial bottleneck caused by creating millions of Python float objects under the GIL. It also made the batch API immediately usable in downstream NumPy-based pipelines.

### Add dedicated validation tests before broadening the API

Parallel code can fail in ways that are not obvious from a single benchmark run. The dedicated correctness suite made the project safer to extend.

### Separate correctness validation from heavy performance benchmarking

This kept CI stable while still allowing large local and server benchmark runs. The later addition of `--no-validate` in the benchmark harness reflects that separation clearly.

## Current Documentation Caveat

The files `benchmarking/README.md` and `benchmarking/NEW_BATCH_API_REFERENCE.txt` were introduced at the `2026-02-08` `10`-descriptor milestone. They are still useful, but their counts reflect that earlier stage.

So today:

- those docs are accurate about the original API design and usage model
- they are no longer the best source for the full local `40`-descriptor scope
- this `project.md` should be read as the timeline and status document for the broader 2026 project history

## Current Status Summary

The 2026 descriptor-parallelization project is in a strong Phase 1 state.

Completed in committed history:

- generic batch infrastructure
- OpenMP build integration
- dedicated correctness tests
- `numpy.ndarray` batch outputs
- multi-descriptor API
- duplicate-pointer safety fix
- local committed expansion to `40` descriptors
- improved benchmark harness for realistic local/server profiling

Not yet delivered by the 2026 commits in this branch:

- vector-valued batch descriptors such as `CalcMQNs` or `CalcAUTOCORR2D`
- conformer-dependent batch 3D descriptors
- dense NumPy fingerprint matrix APIs
- sanitizer-based CI for leak checking

That means the project successfully completed a broad scalar-descriptor Phase 1, but has not yet moved into the more complex vector, 3D, or fingerprint phases.
