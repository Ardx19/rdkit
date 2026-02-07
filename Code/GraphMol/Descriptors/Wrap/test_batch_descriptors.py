"""Tests for parallel batch descriptor computation.

Exercises the OpenMP batch codepath in BatchUtils.h::runBatch<T>() which
releases the GIL and parallelises descriptor computation over a list of
molecules.  Batch descriptor APIs return numpy.ndarray (dtype=float64) for
performance; scalar APIs remain unchanged.

Descriptors tested (individual batch overloads):
  CalcExactMolWt, CalcTPSA, CalcClogP, CalcMR, CalcNumHBD, CalcNumHBA,
  CalcNumRotatableBonds, CalcFractionCSP3, CalcLabuteASA, CalcNumHeavyAtoms

Multi-descriptor batch API:
  CalcDescriptorsBatch(mols, descriptors) — compute multiple descriptors at once
  GetBatchDescriptorNames() — list valid descriptor names

Registered as CTest 'pyBatchDescriptors' so that
``ctest -R Descriptor`` (used in the CI smoke test with OMP_NUM_THREADS=4)
picks it up automatically.
"""

import math
import os
import unittest

import numpy as np

from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolDescriptors as rdMD


def _load_mols(replicate=1):
    """Load test molecules from PBF_egfr.sdf, optionally replicated."""
    # Derive path from this file's location (Code/GraphMol/Descriptors/Wrap/)
    # so tests work both under ctest and standalone pytest.
    test_data = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "test_data", "PBF_egfr.sdf",
    )
    if not os.path.exists(test_data):
        # Fallback: use RDConfig.RDBaseDir if available (ctest environment)
        test_data = os.path.join(
            RDConfig.RDBaseDir,
            "Code", "GraphMol", "Descriptors", "test_data", "PBF_egfr.sdf",
        )
    suppl = Chem.SDMolSupplier(test_data)
    base = [m for m in suppl if m is not None]
    if not base:
        raise RuntimeError(f"No molecules loaded from {test_data}")
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

    def test_return_type(self):
        """Batch must return numpy.ndarray with dtype float64."""
        result = rdMD.CalcExactMolWt(self.mols)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

    def test_result_length(self):
        """Output length must equal input length."""
        result = rdMD.CalcExactMolWt(self.mols)
        self.assertEqual(len(result), len(self.mols))

    def test_empty_list(self):
        """Empty input must return an empty numpy array."""
        result = rdMD.CalcExactMolWt([])
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, np.ndarray)

    def test_none_entries(self):
        """None entries must produce NaN (null-molecule path in runBatch)."""
        mols_with_none = [self.mols[0], None, self.mols[1], None]
        result = rdMD.CalcExactMolWt(mols_with_none)
        self.assertEqual(len(result), 4)
        # Valid molecules must produce finite values
        self.assertFalse(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[2]))
        # None molecules must produce NaN
        self.assertTrue(np.isnan(result[1]),
                        "Expected NaN for None molecule at index 1")
        self.assertTrue(np.isnan(result[3]),
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

    def test_return_type(self):
        """Batch must return numpy.ndarray with dtype float64."""
        result = rdMD.CalcTPSA(self.mols)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

    def test_result_length(self):
        """Output length must equal input length."""
        result = rdMD.CalcTPSA(self.mols)
        self.assertEqual(len(result), len(self.mols))

    def test_empty_list(self):
        """Empty input must return an empty numpy array."""
        result = rdMD.CalcTPSA([])
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, np.ndarray)

    def test_none_entries(self):
        """None entries must produce NaN."""
        mols_with_none = [self.mols[0], None, self.mols[1], None]
        result = rdMD.CalcTPSA(mols_with_none)
        self.assertEqual(len(result), 4)
        self.assertFalse(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[1]),
                        "Expected NaN for None molecule at index 1")
        self.assertTrue(np.isnan(result[3]),
                        "Expected NaN for None molecule at index 3")

    def test_determinism(self):
        """Two consecutive batch calls must return identical results."""
        result1 = rdMD.CalcTPSA(self.mols)
        result2 = rdMD.CalcTPSA(self.mols)
        self.assertEqual(len(result1), len(result2))
        for i, (a, b) in enumerate(zip(result1, result2)):
            self.assertEqual(a, b,
                             msg=f"Non-deterministic result at index {i}")


class TestBatchClogP(unittest.TestCase):
    """Batch CalcClogP(list) vs serial CalcCrippenDescriptors(mol)[0]."""

    def setUp(self):
        self.mols = _load_mols(replicate=10)

    def test_correctness(self):
        """Batch results must match serial CalcCrippenDescriptors logP."""
        serial = [rdMD.CalcCrippenDescriptors(m)[0] for m in self.mols]
        batch = rdMD.CalcClogP(self.mols)
        self.assertEqual(len(serial), len(batch))
        for i, (s, b) in enumerate(zip(serial, batch)):
            self.assertAlmostEqual(s, b, places=4,
                                   msg=f"Mismatch at index {i}")

    def test_return_type(self):
        """Batch must return numpy.ndarray with dtype float64."""
        result = rdMD.CalcClogP(self.mols)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

    def test_result_length(self):
        """Output length must equal input length."""
        result = rdMD.CalcClogP(self.mols)
        self.assertEqual(len(result), len(self.mols))

    def test_empty_list(self):
        """Empty input must return an empty numpy array."""
        result = rdMD.CalcClogP([])
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, np.ndarray)

    def test_none_entries(self):
        """None entries must produce NaN."""
        mols_with_none = [self.mols[0], None, self.mols[1], None]
        result = rdMD.CalcClogP(mols_with_none)
        self.assertEqual(len(result), 4)
        self.assertFalse(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[3]))


class TestBatchMR(unittest.TestCase):
    """Batch CalcMR(list) vs serial CalcCrippenDescriptors(mol)[1]."""

    def setUp(self):
        self.mols = _load_mols(replicate=10)

    def test_correctness(self):
        """Batch results must match serial CalcCrippenDescriptors MR."""
        serial = [rdMD.CalcCrippenDescriptors(m)[1] for m in self.mols]
        batch = rdMD.CalcMR(self.mols)
        self.assertEqual(len(serial), len(batch))
        for i, (s, b) in enumerate(zip(serial, batch)):
            self.assertAlmostEqual(s, b, places=4,
                                   msg=f"Mismatch at index {i}")

    def test_return_type(self):
        """Batch must return numpy.ndarray with dtype float64."""
        result = rdMD.CalcMR(self.mols)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

    def test_result_length(self):
        """Output length must equal input length."""
        result = rdMD.CalcMR(self.mols)
        self.assertEqual(len(result), len(self.mols))

    def test_empty_list(self):
        """Empty input must return an empty numpy array."""
        result = rdMD.CalcMR([])
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, np.ndarray)

    def test_none_entries(self):
        """None entries must produce NaN."""
        mols_with_none = [self.mols[0], None, self.mols[1], None]
        result = rdMD.CalcMR(mols_with_none)
        self.assertEqual(len(result), 4)
        self.assertFalse(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[3]))


class TestBatchNumHBD(unittest.TestCase):
    """Batch CalcNumHBD(list) vs serial CalcNumHBD(mol)."""

    def setUp(self):
        self.mols = _load_mols(replicate=10)

    def test_correctness(self):
        """Batch results must match serial single-molecule calls."""
        serial = [rdMD.CalcNumHBD(m) for m in self.mols]
        batch = rdMD.CalcNumHBD(self.mols)
        self.assertEqual(len(serial), len(batch))
        for i, (s, b) in enumerate(zip(serial, batch)):
            self.assertAlmostEqual(float(s), b, places=4,
                                   msg=f"Mismatch at index {i}")

    def test_return_type(self):
        """Batch must return numpy.ndarray with dtype float64."""
        result = rdMD.CalcNumHBD(self.mols)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

    def test_result_length(self):
        """Output length must equal input length."""
        result = rdMD.CalcNumHBD(self.mols)
        self.assertEqual(len(result), len(self.mols))

    def test_empty_list(self):
        """Empty input must return an empty numpy array."""
        result = rdMD.CalcNumHBD([])
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, np.ndarray)

    def test_none_entries(self):
        """None entries must produce NaN."""
        mols_with_none = [self.mols[0], None, self.mols[1], None]
        result = rdMD.CalcNumHBD(mols_with_none)
        self.assertEqual(len(result), 4)
        self.assertFalse(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[3]))


class TestBatchNumHBA(unittest.TestCase):
    """Batch CalcNumHBA(list) vs serial CalcNumHBA(mol)."""

    def setUp(self):
        self.mols = _load_mols(replicate=10)

    def test_correctness(self):
        """Batch results must match serial single-molecule calls."""
        serial = [rdMD.CalcNumHBA(m) for m in self.mols]
        batch = rdMD.CalcNumHBA(self.mols)
        self.assertEqual(len(serial), len(batch))
        for i, (s, b) in enumerate(zip(serial, batch)):
            self.assertAlmostEqual(float(s), b, places=4,
                                   msg=f"Mismatch at index {i}")

    def test_return_type(self):
        """Batch must return numpy.ndarray with dtype float64."""
        result = rdMD.CalcNumHBA(self.mols)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

    def test_result_length(self):
        """Output length must equal input length."""
        result = rdMD.CalcNumHBA(self.mols)
        self.assertEqual(len(result), len(self.mols))

    def test_empty_list(self):
        """Empty input must return an empty numpy array."""
        result = rdMD.CalcNumHBA([])
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, np.ndarray)

    def test_none_entries(self):
        """None entries must produce NaN."""
        mols_with_none = [self.mols[0], None, self.mols[1], None]
        result = rdMD.CalcNumHBA(mols_with_none)
        self.assertEqual(len(result), 4)
        self.assertFalse(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[3]))


class TestBatchNumRotatableBonds(unittest.TestCase):
    """Batch CalcNumRotatableBonds(list) vs serial CalcNumRotatableBonds(mol)."""

    def setUp(self):
        self.mols = _load_mols(replicate=10)

    def test_correctness(self):
        """Batch results must match serial single-molecule calls."""
        serial = [rdMD.CalcNumRotatableBonds(m) for m in self.mols]
        batch = rdMD.CalcNumRotatableBonds(self.mols)
        self.assertEqual(len(serial), len(batch))
        for i, (s, b) in enumerate(zip(serial, batch)):
            self.assertAlmostEqual(float(s), b, places=4,
                                   msg=f"Mismatch at index {i}")

    def test_return_type(self):
        """Batch must return numpy.ndarray with dtype float64."""
        result = rdMD.CalcNumRotatableBonds(self.mols)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

    def test_result_length(self):
        """Output length must equal input length."""
        result = rdMD.CalcNumRotatableBonds(self.mols)
        self.assertEqual(len(result), len(self.mols))

    def test_empty_list(self):
        """Empty input must return an empty numpy array."""
        result = rdMD.CalcNumRotatableBonds([])
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, np.ndarray)

    def test_none_entries(self):
        """None entries must produce NaN."""
        mols_with_none = [self.mols[0], None, self.mols[1], None]
        result = rdMD.CalcNumRotatableBonds(mols_with_none)
        self.assertEqual(len(result), 4)
        self.assertFalse(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[3]))


class TestBatchFractionCSP3(unittest.TestCase):
    """Batch CalcFractionCSP3(list) vs serial CalcFractionCSP3(mol)."""

    def setUp(self):
        self.mols = _load_mols(replicate=10)

    def test_correctness(self):
        """Batch results must match serial single-molecule calls."""
        serial = [rdMD.CalcFractionCSP3(m) for m in self.mols]
        batch = rdMD.CalcFractionCSP3(self.mols)
        self.assertEqual(len(serial), len(batch))
        for i, (s, b) in enumerate(zip(serial, batch)):
            self.assertAlmostEqual(s, b, places=4,
                                   msg=f"Mismatch at index {i}")

    def test_return_type(self):
        """Batch must return numpy.ndarray with dtype float64."""
        result = rdMD.CalcFractionCSP3(self.mols)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

    def test_result_length(self):
        """Output length must equal input length."""
        result = rdMD.CalcFractionCSP3(self.mols)
        self.assertEqual(len(result), len(self.mols))

    def test_empty_list(self):
        """Empty input must return an empty numpy array."""
        result = rdMD.CalcFractionCSP3([])
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, np.ndarray)

    def test_none_entries(self):
        """None entries must produce NaN."""
        mols_with_none = [self.mols[0], None, self.mols[1], None]
        result = rdMD.CalcFractionCSP3(mols_with_none)
        self.assertEqual(len(result), 4)
        self.assertFalse(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[3]))


class TestBatchLabuteASA(unittest.TestCase):
    """Batch CalcLabuteASA(list) vs serial CalcLabuteASA(mol)."""

    def setUp(self):
        self.mols = _load_mols(replicate=10)

    def test_correctness(self):
        """Batch results must match serial single-molecule calls."""
        serial = [rdMD.CalcLabuteASA(m) for m in self.mols]
        batch = rdMD.CalcLabuteASA(self.mols)
        self.assertEqual(len(serial), len(batch))
        for i, (s, b) in enumerate(zip(serial, batch)):
            self.assertAlmostEqual(s, b, places=4,
                                   msg=f"Mismatch at index {i}")

    def test_return_type(self):
        """Batch must return numpy.ndarray with dtype float64."""
        result = rdMD.CalcLabuteASA(self.mols)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

    def test_result_length(self):
        """Output length must equal input length."""
        result = rdMD.CalcLabuteASA(self.mols)
        self.assertEqual(len(result), len(self.mols))

    def test_empty_list(self):
        """Empty input must return an empty numpy array."""
        result = rdMD.CalcLabuteASA([])
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, np.ndarray)

    def test_none_entries(self):
        """None entries must produce NaN."""
        mols_with_none = [self.mols[0], None, self.mols[1], None]
        result = rdMD.CalcLabuteASA(mols_with_none)
        self.assertEqual(len(result), 4)
        self.assertFalse(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[3]))


class TestBatchNumHeavyAtoms(unittest.TestCase):
    """Batch CalcNumHeavyAtoms(list) vs serial CalcNumHeavyAtoms(mol)."""

    def setUp(self):
        self.mols = _load_mols(replicate=10)

    def test_correctness(self):
        """Batch results must match serial single-molecule calls."""
        serial = [rdMD.CalcNumHeavyAtoms(m) for m in self.mols]
        batch = rdMD.CalcNumHeavyAtoms(self.mols)
        self.assertEqual(len(serial), len(batch))
        for i, (s, b) in enumerate(zip(serial, batch)):
            self.assertAlmostEqual(float(s), b, places=4,
                                   msg=f"Mismatch at index {i}")

    def test_return_type(self):
        """Batch must return numpy.ndarray with dtype float64."""
        result = rdMD.CalcNumHeavyAtoms(self.mols)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)

    def test_result_length(self):
        """Output length must equal input length."""
        result = rdMD.CalcNumHeavyAtoms(self.mols)
        self.assertEqual(len(result), len(self.mols))

    def test_empty_list(self):
        """Empty input must return an empty numpy array."""
        result = rdMD.CalcNumHeavyAtoms([])
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, np.ndarray)

    def test_none_entries(self):
        """None entries must produce NaN."""
        mols_with_none = [self.mols[0], None, self.mols[1], None]
        result = rdMD.CalcNumHeavyAtoms(mols_with_none)
        self.assertEqual(len(result), 4)
        self.assertFalse(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[3]))


# ============================================================================
# Multi-descriptor batch API tests
# ============================================================================

class TestCalcDescriptorsBatch(unittest.TestCase):
    """Tests for CalcDescriptorsBatch(mols, descriptors)."""

    def setUp(self):
        self.mols = _load_mols(replicate=10)
        self.all_names = rdMD.GetBatchDescriptorNames()

    def test_correctness(self):
        """Each column must match the corresponding individual batch call."""
        names = ["CalcExactMolWt", "CalcTPSA", "CalcClogP", "CalcNumHBD"]
        result = rdMD.CalcDescriptorsBatch(self.mols, names)
        individual = {
            "CalcExactMolWt": rdMD.CalcExactMolWt(self.mols),
            "CalcTPSA": rdMD.CalcTPSA(self.mols),
            "CalcClogP": rdMD.CalcClogP(self.mols),
            "CalcNumHBD": rdMD.CalcNumHBD(self.mols),
        }
        for j, name in enumerate(names):
            for i in range(len(self.mols)):
                self.assertAlmostEqual(
                    result[i, j], individual[name][i], places=4,
                    msg=f"Mismatch for {name} at mol index {i}")

    def test_return_type(self):
        """Must return a 2D numpy.ndarray with dtype float64."""
        result = rdMD.CalcDescriptorsBatch(self.mols, ["CalcTPSA"])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(result.ndim, 2)

    def test_shape(self):
        """Shape must be (N_mols, N_descriptors)."""
        names = ["CalcExactMolWt", "CalcClogP", "CalcMR"]
        result = rdMD.CalcDescriptorsBatch(self.mols, names)
        self.assertEqual(result.shape, (len(self.mols), len(names)))

    def test_all_shortcut(self):
        """Passing "all" must return all 10 descriptors in registry order."""
        result_all = rdMD.CalcDescriptorsBatch(self.mols, "all")
        result_explicit = rdMD.CalcDescriptorsBatch(self.mols, self.all_names)
        self.assertEqual(result_all.shape, result_explicit.shape)
        self.assertEqual(result_all.shape[1], 10)
        np.testing.assert_array_almost_equal(result_all, result_explicit,
                                             decimal=10)

    def test_empty_list(self):
        """Empty molecule list must return shape (0, D)."""
        names = ["CalcTPSA", "CalcClogP"]
        result = rdMD.CalcDescriptorsBatch([], names)
        self.assertEqual(result.shape, (0, 2))
        self.assertIsInstance(result, np.ndarray)

    def test_none_entries(self):
        """None molecules must produce NaN for the entire row."""
        mols_with_none = [self.mols[0], None, self.mols[1], None]
        names = ["CalcExactMolWt", "CalcTPSA"]
        result = rdMD.CalcDescriptorsBatch(mols_with_none, names)
        self.assertEqual(result.shape, (4, 2))
        # Valid molecules: no NaN
        self.assertFalse(np.any(np.isnan(result[0, :])))
        self.assertFalse(np.any(np.isnan(result[2, :])))
        # None molecules: all NaN
        self.assertTrue(np.all(np.isnan(result[1, :])),
                        "Expected all NaN for None molecule at row 1")
        self.assertTrue(np.all(np.isnan(result[3, :])),
                        "Expected all NaN for None molecule at row 3")

    def test_invalid_name(self):
        """Unknown descriptor name must raise ValueError."""
        with self.assertRaises(ValueError):
            rdMD.CalcDescriptorsBatch(self.mols, ["CalcBogusDescriptor"])

    def test_invalid_shortcut(self):
        """Unknown string shortcut must raise ValueError."""
        with self.assertRaises(ValueError):
            rdMD.CalcDescriptorsBatch(self.mols, "bogus")

    def test_single_descriptor(self):
        """Single-element list must work, shape (N, 1)."""
        result = rdMD.CalcDescriptorsBatch(self.mols, ["CalcFractionCSP3"])
        self.assertEqual(result.shape, (len(self.mols), 1))
        individual = rdMD.CalcFractionCSP3(self.mols)
        for i in range(len(self.mols)):
            self.assertAlmostEqual(result[i, 0], individual[i], places=4)

    def test_all_descriptors_correctness(self):
        """Every descriptor in 'all' must match its individual batch call."""
        result = rdMD.CalcDescriptorsBatch(self.mols, "all")
        individual_calls = [
            rdMD.CalcExactMolWt(self.mols),
            rdMD.CalcTPSA(self.mols),
            rdMD.CalcClogP(self.mols),
            rdMD.CalcMR(self.mols),
            rdMD.CalcNumHBD(self.mols),
            rdMD.CalcNumHBA(self.mols),
            rdMD.CalcNumRotatableBonds(self.mols),
            rdMD.CalcFractionCSP3(self.mols),
            rdMD.CalcLabuteASA(self.mols),
            rdMD.CalcNumHeavyAtoms(self.mols),
        ]
        for j, col in enumerate(individual_calls):
            for i in range(len(self.mols)):
                self.assertAlmostEqual(
                    result[i, j], col[i], places=4,
                    msg=f"Mismatch for descriptor {j} at mol index {i}")

    def test_determinism(self):
        """Two consecutive calls must return identical results."""
        names = ["CalcClogP", "CalcTPSA", "CalcNumHBA"]
        r1 = rdMD.CalcDescriptorsBatch(self.mols, names)
        r2 = rdMD.CalcDescriptorsBatch(self.mols, names)
        np.testing.assert_array_equal(r1, r2)


class TestGetBatchDescriptorNames(unittest.TestCase):
    """Tests for GetBatchDescriptorNames()."""

    def test_returns_list(self):
        """Must return a Python list of strings."""
        names = rdMD.GetBatchDescriptorNames()
        self.assertIsInstance(names, list)
        for name in names:
            self.assertIsInstance(name, str)

    def test_count(self):
        """Must return exactly 10 descriptor names."""
        names = rdMD.GetBatchDescriptorNames()
        self.assertEqual(len(names), 10)

    def test_known_names(self):
        """All 10 expected descriptor names must be present."""
        names = rdMD.GetBatchDescriptorNames()
        expected = [
            "CalcExactMolWt", "CalcTPSA", "CalcClogP", "CalcMR",
            "CalcNumHBD", "CalcNumHBA", "CalcNumRotatableBonds",
            "CalcFractionCSP3", "CalcLabuteASA", "CalcNumHeavyAtoms",
        ]
        for e in expected:
            self.assertIn(e, names, f"Missing descriptor name: {e}")

    def test_order_matches_all(self):
        """Names must be in the same order as columns from 'all'."""
        names = rdMD.GetBatchDescriptorNames()
        expected_order = [
            "CalcExactMolWt", "CalcTPSA", "CalcClogP", "CalcMR",
            "CalcNumHBD", "CalcNumHBA", "CalcNumRotatableBonds",
            "CalcFractionCSP3", "CalcLabuteASA", "CalcNumHeavyAtoms",
        ]
        self.assertEqual(names, expected_order)


if __name__ == "__main__":
    unittest.main()
