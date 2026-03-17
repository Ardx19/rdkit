//
//  Copyright (C) 2005-2021 Greg Landrum and other RDKit contributors
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//
#include <RDGeneral/Invariant.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/Descriptors/MolData3Ddescriptors.h>
#include "MolDescriptors.h"
#include <map>
#include <list>
#include <algorithm>
#include <sstream>
#include <cmath>

namespace RDKit {
namespace Descriptors {

const std::string amwVersion = "1.0.0";
double calcAMW(const ROMol &mol, bool onlyHeavy) {
  return MolOps::getAvgMolWt(mol, onlyHeavy);
}

const std::string NumHeavyAtomsVersion = "1.0.0";
unsigned int calcNumHeavyAtoms(const ROMol &mol) {
  return mol.getNumHeavyAtoms();
}

const std::string NumAtomsVersion = "1.0.0";
unsigned int calcNumAtoms(const ROMol &mol) {
  bool onlyExplicit = false;
  return mol.getNumAtoms(onlyExplicit);
}

const std::string exactmwVersion = "1.1.0";
double calcExactMW(const ROMol &mol, bool onlyHeavy) {
  return MolOps::getExactMolWt(mol, onlyHeavy);
}

static std::string _molFormulaVersion = "1.3.0";
std::string calcMolFormula(const ROMol &mol, bool separateIsotopes,
                           bool abbreviateHIsotopes) {
  return MolOps::getMolFormula(mol, separateIsotopes, abbreviateHIsotopes);
}

const std::string numValenceElectronsVersion = "1.1.0";
unsigned int calcNumValenceElectrons(const ROMol &mol) {
  const PeriodicTable *tbl = PeriodicTable::getTable();
  unsigned int res = 0;
  for (const auto atom : mol.atoms()) {
    res += tbl->getNouterElecs(atom->getAtomicNum()) - atom->getFormalCharge() +
           atom->getTotalNumHs();
  }
  return res;
}

const std::string numRadicalElectronsVersion = "1.1.0";
unsigned int calcNumRadicalElectrons(const ROMol &mol) {
  unsigned int res = 0;
  for (const auto atom : mol.atoms()) {
    res += atom->getNumRadicalElectrons();
  }
  return res;
}

const std::string heavyAtomMolWtVersion = "1.0.0";
double calcHeavyAtomMolWt(const ROMol &mol) {
  return calcAMW(mol, true);
}

const std::string chi0Version = "1.0.0";
double calcChi0(const ROMol &mol) {
  double res = 0.0;
  for (const auto atom : mol.atoms()) {
    int degree = atom->getDegree();
    if (degree > 0) {
      res += std::sqrt(1.0 / static_cast<double>(degree));
    }
  }
  return res;
}

const std::string chi1Version = "1.0.0";
double calcChi1(const ROMol &mol) {
  double res = 0.0;
  for (const auto bond : mol.bonds()) {
    int deg1 = bond->getBeginAtom()->getDegree();
    int deg2 = bond->getEndAtom()->getDegree();
    int prod = deg1 * deg2;
    if (prod > 0) {
      res += std::sqrt(1.0 / static_cast<double>(prod));
    }
  }
  return res;
}

namespace {
// Helper to compute min/max EState efficiently
void calcEStateIndices(const ROMol &mol, double &maxVal, double &minVal, 
                       double &maxAbsVal, double &minAbsVal) {
  MolData3Ddescriptors mddd;
  std::vector<double> estate = mddd.GetEState(mol);
  if (estate.empty()) {
    maxVal = minVal = maxAbsVal = minAbsVal = 0.0;
    return;
  }
  
  auto [minIt, maxIt] = std::minmax_element(estate.begin(), estate.end());
  minVal = *minIt;
  maxVal = *maxIt;
  
  // Compute min/max absolute values efficiently in single pass
  maxAbsVal = 0.0;
  minAbsVal = std::numeric_limits<double>::max();
  for (double v : estate) {
    double absV = std::abs(v);
    if (absV > maxAbsVal) maxAbsVal = absV;
    if (absV < minAbsVal) minAbsVal = absV;
  }
  if (minAbsVal == std::numeric_limits<double>::max()) {
    minAbsVal = 0.0;
  }
}
}  // anonymous namespace

const std::string maxEStateIndexVersion = "1.0.0";
double calcMaxEStateIndex(const ROMol &mol) {
  double maxVal, minVal, maxAbsVal, minAbsVal;
  calcEStateIndices(mol, maxVal, minVal, maxAbsVal, minAbsVal);
  return maxVal;
}

const std::string minEStateIndexVersion = "1.0.0";
double calcMinEStateIndex(const ROMol &mol) {
  double maxVal, minVal, maxAbsVal, minAbsVal;
  calcEStateIndices(mol, maxVal, minVal, maxAbsVal, minAbsVal);
  return minVal;
}

const std::string maxAbsEStateIndexVersion = "1.0.0";
double calcMaxAbsEStateIndex(const ROMol &mol) {
  double maxVal, minVal, maxAbsVal, minAbsVal;
  calcEStateIndices(mol, maxVal, minVal, maxAbsVal, minAbsVal);
  return maxAbsVal;
}

const std::string minAbsEStateIndexVersion = "1.0.0";
double calcMinAbsEStateIndex(const ROMol &mol) {
  double maxVal, minVal, maxAbsVal, minAbsVal;
  calcEStateIndices(mol, maxVal, minVal, maxAbsVal, minAbsVal);
  return minAbsVal;
}

}  // end of namespace Descriptors
}  // end of namespace RDKit
