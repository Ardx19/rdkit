# RDKit Batch Descriptor Optimization Status Log

**Total Optimized Batch Descriptors:** 53
**Date:** 2026-05-16

This log details the 53 RDKit descriptors that have been ported to the high-performance parallel batch execution architecture. The optimizations utilize C++ OpenMP with Python GIL release to eliminate loop overhead and enable parallel computation.

The implementations are grouped into four phases based on their return type and computation requirements.

---

## Phase 1: Scalar Descriptors (40 Descriptors)
**Status:** ✅ Optimized and Validated. 
**Integration:** These 40 descriptors are fully integrated into the `rdMD.GetBatchDescriptorNames()` registry and can be computed concurrently using the multi-descriptor API: `rdMD.CalcDescriptorsBatch(mols, "all")`. They all return 1D NumPy arrays of `float64`.

1. `CalcExactMolWt`
2. `_CalcMolWt` (Average Molecular Weight)
3. `CalcTPSA`
4. `CalcClogP` (Wildman-Crippen LogP)
5. `CalcMR` (Wildman-Crippen MR)
6. `CalcNumHBD`
7. `CalcNumHBA`
8. `CalcNumRotatableBonds`
9. `CalcFractionCSP3`
10. `CalcLabuteASA`
11. `CalcNumHeavyAtoms`
12. `CalcNumAromaticRings`
13. `CalcNumSaturatedRings`
14. `CalcNumAliphaticRings`
15. `CalcNumHeterocycles`
16. `CalcNumAromaticHeterocycles`
17. `CalcNumSaturatedHeterocycles`
18. `CalcNumAliphaticHeterocycles`
19. `CalcNumAromaticCarbocycles`
20. `CalcNumSaturatedCarbocycles`
21. `CalcNumAliphaticCarbocycles`
22. `CalcNumHeteroatoms`
23. `CalcNumAmideBonds`
24. `CalcNumAtoms`
25. `CalcNumSpiroAtoms`
26. `CalcNumBridgeheadAtoms`
27. `CalcKappa1`
28. `CalcKappa2`
29. `CalcKappa3`
30. `CalcChi0v`
31. `CalcChi1v`
32. `CalcChi2v`
33. `CalcChi3v`
34. `CalcChi4v`
35. `CalcChi0n`
36. `CalcChi1n`
37. `CalcChi2n`
38. `CalcChi3n`
39. `CalcChi4n`
40. `CalcHallKierAlpha`

---

## Phase 2: Vector Descriptors (2 Descriptors)
**Status:** ✅ Optimized and Validated.
**Integration:** These compute molecular descriptors that return arrays of values per molecule. They are excluded from the `CalcDescriptorsBatch("all")` API because they return 2D arrays, but they have individual batch overloads.

41. `CalcAUTOCORR2D` (Returns Autocorrelogram vector)
42. `CalcMQNs` (Returns Molecular Quantum Numbers vector)

---

## Phase 3: 3D Descriptors (8 Descriptors)
**Status:** ✅ Optimized and Validated.
**Integration:** These conformer-dependent descriptors are successfully ported to the batch architecture. They are separated from Phase 1 because they require molecules to have 3D coordinates (via `AllChem.EmbedMolecule`). If passed 2D molecules, they correctly return `NaN` arrays without throwing exceptions to Python.

43. `CalcAsphericity`
44. `CalcEccentricity`
45. `CalcPBF` (Plane of Best Fit)
46. `CalcPMI1` (Principal Moment of Inertia 1)
47. `CalcPMI2` (Principal Moment of Inertia 2)
48. `CalcPMI3` (Principal Moment of Inertia 3)
49. `CalcRadiusOfGyration`
50. `CalcSpherocityIndex`

---

## Phase 4: Fingerprints (3 Descriptors)
**Status:** ✅ Optimized and Validated (Yielding highest speedups: ~24-28x).
**Integration:** Ported using a specialized template that skips the Python `ExplicitBitVect` initialization entirely, streaming bits directly to C++ NumPy uint8/uint32 memory allocations. 

51. `GetMorganFingerprintAsBitVect`
52. `GetMACCSKeysFingerprint` (Wrapped in C++ as `GetMACCSKeysFingerprintAsBitVect_List`)
53. `GetHashedTopologicalTorsionFingerprintAsBitVect`

---
*End of Log*