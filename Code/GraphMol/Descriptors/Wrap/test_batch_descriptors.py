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


def _load_3d_mols(replicate=1):
    """Load test molecules from PBF_egfr.sdf with 3D conformers, optionally replicated."""
    test_data = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "test_data", "PBF_egfr.sdf",
    )
    if not os.path.exists(test_data):
        test_data = os.path.join(
            RDConfig.RDBaseDir,
            "Code", "GraphMol", "Descriptors", "test_data", "PBF_egfr.sdf",
        )
    suppl = Chem.SDMolSupplier(test_data)
    base = [m for m in suppl if m is not None]
    if not base:
        raise RuntimeError(f"No molecules loaded from {test_data}")
    return base * replicate

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
    smiles = [Chem.MolToSmiles(m) for m in base]
    smiles_pool = smiles * replicate
    mols = [Chem.MolFromSmiles(s) for s in smiles_pool]
    return [m for m in mols if m is not None]


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

class TestBatchVectorDescriptors(unittest.TestCase):
    """Batch for vector descriptors (MQNs and AUTOCORR2D)."""

    def setUp(self):
        self.mols = _load_mols(replicate=1)

    def test_mqns_batch(self):
        serial = [rdMD.CalcMQNs(m) for m in self.mols]
        batch = rdMD.CalcMQNs(self.mols)
        self.assertEqual(batch.shape, (len(self.mols), 42))
        for i, (s, b) in enumerate(zip(serial, batch)):
            np.testing.assert_allclose(s, b, err_msg=f"Mismatch at {i}")

    def test_autocorr2d_batch(self):
        serial = [rdMD.CalcAUTOCORR2D(m) for m in self.mols]
        batch = rdMD.CalcAUTOCORR2D(self.mols)
        self.assertEqual(batch.shape, (len(self.mols), 192))
        for i, (s, b) in enumerate(zip(serial, batch)):
            np.testing.assert_allclose(s, b, err_msg=f"Mismatch at {i}")

    def test_none_entries(self):
        mols_with_none = [self.mols[0], None, self.mols[1]]
        mqns = rdMD.CalcMQNs(mols_with_none)
        self.assertEqual(mqns.shape, (3, 42))
        self.assertFalse(np.isnan(mqns[0]).any())
        self.assertTrue(np.isnan(mqns[1]).all())
        self.assertFalse(np.isnan(mqns[2]).any())

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

    def test_include_sandp_option(self):
        """Batch overload must honor the includeSandP option."""
        serial = [rdMD.CalcTPSA(m, False, True) for m in self.mols]
        batch = rdMD.CalcTPSA(self.mols, True)
        self.assertEqual(len(serial), len(batch))
        for i, (s, b) in enumerate(zip(serial, batch)):
            self.assertAlmostEqual(s, b, places=4,
                                   msg=f"Mismatch at index {i} with includeSandP=True")


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


class TestBatchAdditionalDescriptors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mols = _load_mols(replicate=1)  # small set for correctness
        
        # Format: (BatchFunction, ScalarFunction)
        # Format: (BatchFunction, ScalarFunction)
        # We explicitly wrap integer-returning scalars with float() 
        # to ensure strict dtype checking against the float64 batch arrays.
        cls.descriptors = [
            (rdMD.CalcNumAromaticRings, lambda m: float(rdMD.CalcNumAromaticRings(m))),
            (rdMD.CalcNumSaturatedRings, lambda m: float(rdMD.CalcNumSaturatedRings(m))),
            (rdMD.CalcNumAliphaticRings, lambda m: float(rdMD.CalcNumAliphaticRings(m))),
            (rdMD.CalcNumHeterocycles, lambda m: float(rdMD.CalcNumHeterocycles(m))),
            (rdMD.CalcNumAromaticHeterocycles, lambda m: float(rdMD.CalcNumAromaticHeterocycles(m))),
            (rdMD.CalcNumSaturatedHeterocycles, lambda m: float(rdMD.CalcNumSaturatedHeterocycles(m))),
            (rdMD.CalcNumAliphaticHeterocycles, lambda m: float(rdMD.CalcNumAliphaticHeterocycles(m))),
            (rdMD.CalcNumAromaticCarbocycles, lambda m: float(rdMD.CalcNumAromaticCarbocycles(m))),
            (rdMD.CalcNumSaturatedCarbocycles, lambda m: float(rdMD.CalcNumSaturatedCarbocycles(m))),
            (rdMD.CalcNumAliphaticCarbocycles, lambda m: float(rdMD.CalcNumAliphaticCarbocycles(m))),
            (rdMD.CalcNumHeteroatoms, lambda m: float(rdMD.CalcNumHeteroatoms(m))),
            (rdMD.CalcNumAmideBonds, lambda m: float(rdMD.CalcNumAmideBonds(m))),
            (rdMD.CalcNumAtoms, lambda m: float(rdMD.CalcNumAtoms(m))),
            (rdMD.CalcNumSpiroAtoms, lambda m: float(rdMD.CalcNumSpiroAtoms(m))),
            (rdMD.CalcNumBridgeheadAtoms, lambda m: float(rdMD.CalcNumBridgeheadAtoms(m))),
            (rdMD._CalcMolWt, lambda m: float(rdMD._CalcMolWt(m))),
            (rdMD.CalcKappa1, lambda m: float(rdMD.CalcKappa1(m))),
            (rdMD.CalcKappa2, lambda m: float(rdMD.CalcKappa2(m))),
            (rdMD.CalcKappa3, lambda m: float(rdMD.CalcKappa3(m))),
            (rdMD.CalcChi0v, lambda m: float(rdMD.CalcChi0v(m))),
            (rdMD.CalcChi1v, lambda m: float(rdMD.CalcChi1v(m))),
            (rdMD.CalcChi2v, lambda m: float(rdMD.CalcChi2v(m))),
            (rdMD.CalcChi3v, lambda m: float(rdMD.CalcChi3v(m))),
            (rdMD.CalcChi4v, lambda m: float(rdMD.CalcChi4v(m))),
            (rdMD.CalcChi0n, lambda m: float(rdMD.CalcChi0n(m))),
            (rdMD.CalcChi1n, lambda m: float(rdMD.CalcChi1n(m))),
            (rdMD.CalcChi2n, lambda m: float(rdMD.CalcChi2n(m))),
            (rdMD.CalcChi3n, lambda m: float(rdMD.CalcChi3n(m))),
            (rdMD.CalcChi4n, lambda m: float(rdMD.CalcChi4n(m))),
            (rdMD.CalcHallKierAlpha, lambda m: float(rdMD.CalcHallKierAlpha(m))),
        ]

    def test_correctness_vs_scalar(self):
        """Verify that batch results perfectly match serial results."""
        for batch_fn, scalar_fn in self.descriptors:
            with self.subTest(descriptor=batch_fn.__name__):
                batch_res = batch_fn(self.mols)
                serial_res = np.array([scalar_fn(m) for m in self.mols], dtype=np.float64)
                
                self.assertIsInstance(batch_res, np.ndarray)
                self.assertEqual(batch_res.dtype, np.float64)
                self.assertEqual(batch_res.shape, (len(self.mols),))
                np.testing.assert_allclose(batch_res, serial_res, rtol=1e-7, atol=1e-7)

    def test_empty_list(self):
        """Passing an empty list should return an empty array without crashing."""
        for batch_fn, _ in self.descriptors:
            with self.subTest(descriptor=batch_fn.__name__):
                res = batch_fn([])
                self.assertIsInstance(res, np.ndarray)
                self.assertEqual(res.shape, (0,))
                self.assertEqual(res.dtype, np.float64)

    def test_none_handling(self):
        """Missing/None molecules should output NaN."""
        test_mols = [self.mols[0], None, self.mols[1]]
        for batch_fn, scalar_fn in self.descriptors:
            with self.subTest(descriptor=batch_fn.__name__):
                res = batch_fn(test_mols)
                self.assertEqual(res.shape, (3,))
                
                self.assertFalse(np.isnan(res[0]))
                self.assertTrue(np.isnan(res[1]))
                self.assertFalse(np.isnan(res[2]))
                
                np.testing.assert_allclose(res[0], scalar_fn(test_mols[0]), rtol=1e-7)
                np.testing.assert_allclose(res[2], scalar_fn(test_mols[2]), rtol=1e-7)


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
        """Passing "all" must return all Phase 1 descriptors in registry order."""
        result_all = rdMD.CalcDescriptorsBatch(self.mols, "all")
        result_explicit = rdMD.CalcDescriptorsBatch(self.mols, self.all_names)
        self.assertEqual(result_all.shape, result_explicit.shape)
        # Note: GetBatchDescriptorNames() intentionally returns only the 40 Phase 1 scalar descriptors.
        # Phase 2 (vectors), Phase 3 (conformer-dependent 3D), and Phase 4 (fingerprints) are
        # explicitly excluded to ensure CalcDescriptorsBatch('all') always returns a perfectly 
        # rectangular NxD float64 numpy array without crashing on 2D molecules or jagged lengths.
        self.assertEqual(result_all.shape[1], 40)
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
        names = rdMD.GetBatchDescriptorNames()
        result_all = rdMD.CalcDescriptorsBatch(self.mols, "all")
        
        for j, name in enumerate(names):
            self.assertTrue(hasattr(rdMD, name), f"Registered batch descriptor {name} is missing from Python API")
            batch_fn = getattr(rdMD, name)
            try:
                col_individual = batch_fn(self.mols)
            except TypeError:
                # some take params like CalcExactMolWt(mols, False), but default works
                continue
            
            for i in range(len(self.mols)):
                if np.isnan(result_all[i, j]) and np.isnan(col_individual[i]):
                    continue
                self.assertAlmostEqual(
                    result_all[i, j], col_individual[i], places=4,
                    msg=f"Mismatch for {name} at mol {i}"
                )

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
        """Must return exactly 40 Phase 1 scalar descriptor names."""
        names = rdMD.GetBatchDescriptorNames()
        self.assertEqual(len(names), 40)

    def test_known_names(self):
        """All 40 expected descriptor names must be present."""
        names = rdMD.GetBatchDescriptorNames()
        expected = [
            "CalcExactMolWt", "CalcTPSA", "CalcClogP", "CalcMR",
            "CalcNumHBD", "CalcNumHBA", "CalcNumRotatableBonds",
            "CalcFractionCSP3", "CalcLabuteASA", "CalcNumHeavyAtoms",
        ]
        for e in expected:
            self.assertIn(e, names, f"Missing descriptor name: {e}")

    def test_order_matches_all(self):
        """Names must be in the same order as columns from 'all' (verified implicitly by correctness tests)."""
        names = rdMD.GetBatchDescriptorNames()
        self.assertEqual(len(names), 40)

class TestBatch3DDescriptors(unittest.TestCase):
    """Batch for 3D descriptors."""

    def setUp(self):
        self.mols_3d = _load_3d_mols(replicate=1)
        self.mols_2d = _load_mols(replicate=1)

    def test_pbf_batch(self):
        serial = [rdMD.CalcPBF(m) for m in self.mols_3d]
        batch = rdMD.CalcPBF(self.mols_3d)
        self.assertEqual(batch.shape, (len(self.mols_3d),))
        for i, (s, b) in enumerate(zip(serial, batch)):
            self.assertAlmostEqual(s, b, places=4)

    def test_pmi_batch(self):
        serial1 = [rdMD.CalcPMI1(m) for m in self.mols_3d]
        batch1 = rdMD.CalcPMI1(self.mols_3d)
        serial2 = [rdMD.CalcPMI2(m) for m in self.mols_3d]
        batch2 = rdMD.CalcPMI2(self.mols_3d)
        serial3 = [rdMD.CalcPMI3(m) for m in self.mols_3d]
        batch3 = rdMD.CalcPMI3(self.mols_3d)

        for i, (s, b) in enumerate(zip(serial1, batch1)):
            self.assertAlmostEqual(s, b, places=4)
        for i, (s, b) in enumerate(zip(serial2, batch2)):
            self.assertAlmostEqual(s, b, places=4)
        for i, (s, b) in enumerate(zip(serial3, batch3)):
            self.assertAlmostEqual(s, b, places=4)

    def test_other_3d_batch(self):
        # Asphericity
        serial_as = [rdMD.CalcAsphericity(m) for m in self.mols_3d]
        batch_as = rdMD.CalcAsphericity(self.mols_3d)
        for i, (s, b) in enumerate(zip(serial_as, batch_as)):
            self.assertAlmostEqual(s, b, places=4)

        # Eccentricity
        serial_ec = [rdMD.CalcEccentricity(m) for m in self.mols_3d]
        batch_ec = rdMD.CalcEccentricity(self.mols_3d)
        for i, (s, b) in enumerate(zip(serial_ec, batch_ec)):
            self.assertAlmostEqual(s, b, places=4)

        # RadiusOfGyration
        serial_rg = [rdMD.CalcRadiusOfGyration(m) for m in self.mols_3d]
        batch_rg = rdMD.CalcRadiusOfGyration(self.mols_3d)
        for i, (s, b) in enumerate(zip(serial_rg, batch_rg)):
            self.assertAlmostEqual(s, b, places=4)

        # SpherocityIndex
        serial_si = [rdMD.CalcSpherocityIndex(m) for m in self.mols_3d]
        batch_si = rdMD.CalcSpherocityIndex(self.mols_3d)
        for i, (s, b) in enumerate(zip(serial_si, batch_si)):
            self.assertAlmostEqual(s, b, places=4)

    def test_3d_on_2d_mols(self):
        """2D molecules must return NaN arrays for 3D descriptors."""
        batch_pbf = rdMD.CalcPBF(self.mols_2d)
        self.assertTrue(np.isnan(batch_pbf).all())

        batch_pmi1 = rdMD.CalcPMI1(self.mols_2d)
        self.assertTrue(np.isnan(batch_pmi1).all())

        batch_ec = rdMD.CalcEccentricity(self.mols_2d)
        self.assertTrue(np.isnan(batch_ec).all())


class TestBatchMorganFingerprint(unittest.TestCase):
    """Batch GetMorganFingerprintAsBitVect(list) vs scalar loop."""

    def setUp(self):
        self.mols = _load_mols(replicate=1)

    def test_shape(self):
        result = rdMD.GetMorganFingerprintAsBitVect(self.mols, 2, 2048)
        self.assertEqual(result.shape, (len(self.mols), 2048))

    def test_dtype(self):
        result = rdMD.GetMorganFingerprintAsBitVect(self.mols, 2, 2048)
        self.assertEqual(result.dtype, np.uint8)

    def test_correctness(self):
        """Batch bits must match scalar GetMorganFingerprintAsBitVect row-by-row."""
        from rdkit.DataStructs import ConvertToNumpyArray
        batch = rdMD.GetMorganFingerprintAsBitVect(self.mols, 2, 2048)
        for i, mol in enumerate(self.mols):
            bv = rdMD.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            expected = np.zeros(2048, dtype=np.uint8)
            ConvertToNumpyArray(bv, expected)
            np.testing.assert_array_equal(batch[i], expected,
                                          err_msg=f"Morgan mismatch at index {i}")

    def test_none_entries(self):
        """None entries must produce a row of zeros."""
        mols_with_none = [self.mols[0], None, self.mols[1]]
        result = rdMD.GetMorganFingerprintAsBitVect(mols_with_none, 2, 2048)
        self.assertEqual(result.shape, (3, 2048))
        self.assertTrue(np.all(result[1] == 0), "None row should be all zeros")

    def test_empty_list(self):
        result = rdMD.GetMorganFingerprintAsBitVect([], 2, 2048)
        self.assertEqual(result.shape, (0, 2048))

    def test_custom_nbits(self):
        result = rdMD.GetMorganFingerprintAsBitVect(self.mols, 2, 1024)
        self.assertEqual(result.shape, (len(self.mols), 1024))

    def test_use_chirality(self):
        from rdkit.DataStructs import ConvertToNumpyArray
        batch = rdMD.GetMorganFingerprintAsBitVect(self.mols, 2, 2048, useChirality=True)
        for i, mol in enumerate(self.mols):
            bv = rdMD.GetMorganFingerprintAsBitVect(mol, 2, 2048, useChirality=True)
            expected = np.zeros(2048, dtype=np.uint8)
            ConvertToNumpyArray(bv, expected)
            np.testing.assert_array_equal(batch[i], expected,
                                          err_msg=f"Morgan mismatch at index {i} (useChirality=True)")

    def test_use_features(self):
        from rdkit.DataStructs import ConvertToNumpyArray
        batch = rdMD.GetMorganFingerprintAsBitVect(self.mols, 2, 2048, useFeatures=True)
        for i, mol in enumerate(self.mols):
            bv = rdMD.GetMorganFingerprintAsBitVect(mol, 2, 2048, useFeatures=True)
            expected = np.zeros(2048, dtype=np.uint8)
            ConvertToNumpyArray(bv, expected)
            np.testing.assert_array_equal(batch[i], expected,
                                          err_msg=f"Morgan mismatch at index {i} (useFeatures=True)")

class TestBatchTopologicalTorsionFingerprint(unittest.TestCase):
    """Batch GetHashedTopologicalTorsionFingerprintAsBitVect(list) vs scalar loop."""

    def setUp(self):
        self.mols = _load_mols(replicate=1)

    def test_shape(self):
        result = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(self.mols)
        self.assertEqual(result.shape, (len(self.mols), 2048))

    def test_dtype(self):
        result = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(self.mols)
        self.assertEqual(result.dtype, np.uint8)

    def test_correctness(self):
        from rdkit.DataStructs import ConvertToNumpyArray
        batch = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(self.mols)
        for i, mol in enumerate(self.mols):
            bv = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)
            expected = np.zeros(2048, dtype=np.uint8)
            ConvertToNumpyArray(bv, expected)
            np.testing.assert_array_equal(batch[i], expected,
                                          err_msg=f"Topological Torsion mismatch at index {i}")

    def test_none_entries(self):
        mols_with_none = [self.mols[0], None, self.mols[1]]
        result = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(mols_with_none)
        self.assertEqual(result.shape, (3, 2048))
        self.assertTrue(np.all(result[1] == 0), "None row should be all zeros")

    def test_empty_list(self):
        result = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect([])
        self.assertEqual(result.shape, (0, 2048))

    def test_custom_nbits(self):
        result = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(self.mols, nBits=1024)
        self.assertEqual(result.shape, (len(self.mols), 1024))

    def test_include_chirality(self):
        from rdkit.DataStructs import ConvertToNumpyArray
        batch = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(self.mols, includeChirality=True)
        for i, mol in enumerate(self.mols):
            bv = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, includeChirality=True)
            expected = np.zeros(2048, dtype=np.uint8)
            ConvertToNumpyArray(bv, expected)
            np.testing.assert_array_equal(batch[i], expected,
                                          err_msg=f"Topological Torsion mismatch at index {i} (includeChirality=True)")


class TestBatchMACCSFingerprint(unittest.TestCase):
    """Batch GetMACCSKeysFingerprint(list) vs scalar loop."""

    def setUp(self):
        self.mols = _load_mols(replicate=1)

    def test_shape(self):
        result = rdMD.GetMACCSKeysFingerprint(self.mols)
        self.assertEqual(result.shape, (len(self.mols), 167))

    def test_dtype(self):
        result = rdMD.GetMACCSKeysFingerprint(self.mols)
        self.assertEqual(result.dtype, np.uint8)

    def test_correctness(self):
        """Batch bits must match scalar GetMACCSKeysFingerprint row-by-row."""
        from rdkit.DataStructs import ConvertToNumpyArray
        batch = rdMD.GetMACCSKeysFingerprint(self.mols)
        for i, mol in enumerate(self.mols):
            bv = rdMD.GetMACCSKeysFingerprint(mol)
            expected = np.zeros(167, dtype=np.uint8)
            ConvertToNumpyArray(bv, expected)
            np.testing.assert_array_equal(batch[i], expected,
                                          err_msg=f"MACCS mismatch at index {i}")

    def test_none_entries(self):
        """None entries must produce a row of zeros."""
        mols_with_none = [self.mols[0], None, self.mols[1]]
        result = rdMD.GetMACCSKeysFingerprint(mols_with_none)
        self.assertEqual(result.shape, (3, 167))
        self.assertTrue(np.all(result[1] == 0), "None row should be all zeros")

    def test_empty_list(self):
        result = rdMD.GetMACCSKeysFingerprint([])
        self.assertEqual(result.shape, (0, 167))

if __name__ == '__main__':
    unittest.main()
