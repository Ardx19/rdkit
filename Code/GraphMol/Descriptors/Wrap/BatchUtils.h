#ifndef BATCH_UTILS_H
#define BATCH_UTILS_H

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

inline std::vector<const ROMol *> extractMolPtrs(python::object mols) {
  python::list molList = python::extract<python::list>(mols);
  unsigned int n = python::len(molList);
  std::vector<const ROMol *> molPtrs(n);
  for (unsigned int i = 0; i < n; ++i) {
    if (molList[i] == python::object()) {
      molPtrs[i] = nullptr;
    } else {
      molPtrs[i] = python::extract<const ROMol *>(molList[i]);
    }
  }
  return molPtrs;
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
