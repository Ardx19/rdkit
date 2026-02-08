#ifndef BATCH_UTILS_H
#define BATCH_UTILS_H

#include <memory>
#include <unordered_set>
#include <vector>
#include <limits>
#include <RDBoost/Wrap.h>
#include <GraphMol/GraphMol.h>
#include <Python.h> // Required for Py_BEGIN_ALLOW_THREADS

#ifdef _OPENMP
#include <omp.h>
#endif

namespace python = boost::python;

namespace RDKit {
namespace Descriptors {

// Helper RAII class to release the GIL
class NOGIL {
public:
  inline NOGIL() {
    _save = PyEval_SaveThread();
  }
  inline ~NOGIL() {
    PyEval_RestoreThread(_save);
  }
private:
  PyThreadState *_save;
};

// Holds the molecule pointer array and owns any deep-copied molecules.
// When the input Python list contains duplicate references to the same
// molecule object, the duplicates are deep-copied so that every pointer
// in `ptrs` is unique.  This prevents data races when OpenMP threads
// concurrently call descriptor functions that mutate molecule state
// through const references (lazy ring-info init, property-dict caching).
struct MolBatch {
  std::vector<const ROMol *> ptrs;
  // Lifetime storage for deep-copied duplicates; destroyed when
  // MolBatch goes out of scope (after runBatch completes).
  std::vector<std::unique_ptr<ROMol>> owned;
};

inline MolBatch extractMolPtrs(python::object mols) {
  MolBatch batch;
  python::list molList = python::extract<python::list>(mols);
  unsigned int n = python::len(molList);
  batch.ptrs.resize(n);

  std::unordered_set<const ROMol *> seen;

  for (unsigned int i = 0; i < n; ++i) {
    if (molList[i] == python::object()) {
      batch.ptrs[i] = nullptr;
    } else {
      const ROMol *mol = python::extract<const ROMol *>(molList[i]);
      if (mol && !seen.insert(mol).second) {
        // Duplicate pointer — deep-copy for thread safety
        batch.owned.emplace_back(new ROMol(*mol));
        batch.ptrs[i] = batch.owned.back().get();
      } else {
        batch.ptrs[i] = mol;
      }
    }
  }
  return batch;
}

// Swapped template args: T first so it can be explicit <double>, Func deduced
template <typename T, typename Func>
std::vector<T> runBatch(const std::vector<const ROMol *> &mols,
                        Func descriptorFn) {
  size_t n = mols.size();
  std::vector<T> results(n);

  {
    // Release GIL
    NOGIL gil;
    
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < n; ++i) {
      try {
        if (mols[i]) {
          results[i] = descriptorFn(*mols[i]);
        } else {
          results[i] = std::numeric_limits<T>::quiet_NaN();
        }
      } catch (...) {
        results[i] = std::numeric_limits<T>::quiet_NaN();
      }
    }
  }
  return results;
}

}  // namespace Descriptors
}  // namespace RDKit

#endif
