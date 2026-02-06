"""Tests for parallel batch descriptor computation (CalcExactMolWt, CalcTPSA).

Exercises the OpenMP batch codepath in BatchUtils.h::runBatch<T>() which
releases the GIL and parallelises descriptor computation over a list of
molecules.  Registered as CTest 'pyBatchDescriptors' so that
``ctest -R Descriptor`` (used in the CI smoke test with OMP_NUM_THREADS=4)
picks it up automatically.
"""

import math
import os
import unittest

from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolDescriptors as rdMD


def _load_mols(replicate=1):
    """Load test molecules from PBF_egfr.sdf, optionally replicated."""
    sdf = os.path.join(
        RDConfig.RDBaseDir,
        "Code", "GraphMol", "Descriptors", "test_data", "PBF_egfr.sdf",
    )
    suppl = Chem.SDMolSupplier(sdf)
    base = [m for m in suppl if m is not None]
    if not base:
        raise RuntimeError(f"No molecules loaded from {sdf}")
    mols = base * replicate
    return mols


class TestBatchExactMolWt(unittest.TestCase):
    """Batch CalcExactMolWt(list) vs serial CalcExactMolWt(mol)."""

    def setUp(self):
        # ~100 mols replicated 10x -> ~1000 molecules.
        # Large enough to exercise OpenMP scheduling, small enough for CI.
        self.mols = _load_mols(replicate=10)

    def test_correctness(self):
        """Batch results must match serial single-molecule calls."""
        serial = [rdMD.CalcExactMolWt(m) for m in self.mols]
        batch = rdMD.CalcExactMolWt(self.mols)
        self.assertEqual(len(serial), len(batch))
        for i, (s, b) in enumerate(zip(serial, batch)):
            self.assertAlmostEqual(s, b, places=4,
                                   msg=f"Mismatch at index {i}")

    def test_result_length(self):
        """Output length must equal input length."""
        result = rdMD.CalcExactMolWt(self.mols)
        self.assertEqual(len(result), len(self.mols))

    def test_empty_list(self):
        """Empty input must return an empty list."""
        result = rdMD.CalcExactMolWt([])
        self.assertEqual(len(result), 0)

    def test_none_entries(self):
        """None entries must produce NaN (null-molecule path in runBatch)."""
        mols_with_none = [self.mols[0], None, self.mols[1], None]
        result = rdMD.CalcExactMolWt(mols_with_none)
        self.assertEqual(len(result), 4)
        # Valid molecules must produce finite values
        self.assertFalse(math.isnan(result[0]))
        self.assertFalse(math.isnan(result[2]))
        # None molecules must produce NaN
        self.assertTrue(math.isnan(result[1]),
                        "Expected NaN for None molecule at index 1")
        self.assertTrue(math.isnan(result[3]),
                        "Expected NaN for None molecule at index 3")

    def test_determinism(self):
        """Two consecutive batch calls must return identical results.

        Guards against race conditions in the OpenMP parallel-for path.
        """
        result1 = rdMD.CalcExactMolWt(self.mols)
        result2 = rdMD.CalcExactMolWt(self.mols)
        self.assertEqual(len(result1), len(result2))
        for i, (a, b) in enumerate(zip(result1, result2)):
            self.assertEqual(a, b,
                             msg=f"Non-deterministic result at index {i}")


class TestBatchTPSA(unittest.TestCase):
    """Batch CalcTPSA(list) vs serial CalcTPSA(mol)."""

    def setUp(self):
        self.mols = _load_mols(replicate=10)

    def test_correctness(self):
        """Batch results must match serial single-molecule calls."""
        serial = [rdMD.CalcTPSA(m) for m in self.mols]
        batch = rdMD.CalcTPSA(self.mols)
        self.assertEqual(len(serial), len(batch))
        for i, (s, b) in enumerate(zip(serial, batch)):
            self.assertAlmostEqual(s, b, places=4,
                                   msg=f"Mismatch at index {i}")

    def test_result_length(self):
        """Output length must equal input length."""
        result = rdMD.CalcTPSA(self.mols)
        self.assertEqual(len(result), len(self.mols))

    def test_empty_list(self):
        """Empty input must return an empty list."""
        result = rdMD.CalcTPSA([])
        self.assertEqual(len(result), 0)

    def test_none_entries(self):
        """None entries must produce NaN."""
        mols_with_none = [self.mols[0], None, self.mols[1], None]
        result = rdMD.CalcTPSA(mols_with_none)
        self.assertEqual(len(result), 4)
        self.assertFalse(math.isnan(result[0]))
        self.assertFalse(math.isnan(result[2]))
        self.assertTrue(math.isnan(result[1]),
                        "Expected NaN for None molecule at index 1")
        self.assertTrue(math.isnan(result[3]),
                        "Expected NaN for None molecule at index 3")

    def test_determinism(self):
        """Two consecutive batch calls must return identical results."""
        result1 = rdMD.CalcTPSA(self.mols)
        result2 = rdMD.CalcTPSA(self.mols)
        self.assertEqual(len(result1), len(result2))
        for i, (a, b) in enumerate(zip(result1, result2)):
            self.assertEqual(a, b,
                             msg=f"Non-deterministic result at index {i}")


if __name__ == "__main__":
    unittest.main()
