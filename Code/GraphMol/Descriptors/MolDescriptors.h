//
//  Copyright (C) 2004-2021 Greg Landrum and other RDKit contributors
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//

#include <RDGeneral/export.h>
#ifndef RD_MOLDESCRIPTORS_H
#define RD_MOLDESCRIPTORS_H

#include <GraphMol/Descriptors/Crippen.h>
#include <GraphMol/Descriptors/MolSurf.h>
#include <GraphMol/Descriptors/Lipinski.h>
#include <GraphMol/Descriptors/ConnectivityDescriptors.h>
#include <GraphMol/Descriptors/MQN.h>
#include <GraphMol/Descriptors/AUTOCORR2D.h>

namespace RDKit {
class ROMol;
namespace Descriptors {
/*!
  Calculates a molecule's average molecular weight

  \param mol        the molecule of interest
  \param onlyHeavy  (optional) if this is true (the default is false),
      only heavy atoms will be included in the MW calculation

  \return the AMW
*/
RDKIT_DESCRIPTORS_EXPORT extern const std::string amwVersion;
RDKIT_DESCRIPTORS_EXPORT double calcAMW(const ROMol &mol,
                                        bool onlyHeavy = false);
/*!
  Calculates a molecule's number of heavy (non-hydrogen) atoms

  \param mol        the molecule of interest

  \return the number of heavy atoms
*/
RDKIT_DESCRIPTORS_EXPORT extern const std::string NumHeavyAtomsVersion;
RDKIT_DESCRIPTORS_EXPORT unsigned int calcNumHeavyAtoms(const ROMol &mol);
/*!
  Calculates a molecule's number of atoms

  \param mol        the molecule of interest

  \return the number of atoms
*/
RDKIT_DESCRIPTORS_EXPORT extern const std::string NumAtomsVersion;
RDKIT_DESCRIPTORS_EXPORT unsigned int calcNumAtoms(const ROMol &mol);
/*!
  Calculates a molecule's exact molecular weight

  \param mol        the molecule of interest
  \param onlyHeavy  (optional) if this is true (the default is false),
      only heavy atoms will be included in the MW calculation

  \return the exact MW
*/
RDKIT_DESCRIPTORS_EXPORT extern const std::string exactmwVersion;
RDKIT_DESCRIPTORS_EXPORT double calcExactMW(const ROMol &mol,
                                            bool onlyHeavy = false);
/*!
  Calculates a molecule's formula

  \param mol        the molecule of interest
  \param separateIsotopes  if true, isotopes will show up separately in the
     formula. So C[13CH2]O will give the formula: C[13C]H6O
  \param abbreviateHIsotopes  if true, 2H and 3H will be represented as
     D and T instead of [2H] and [3H]. This only applies if \c separateIsotopes
     is true

  \return the formula as a string
*/
RDKIT_DESCRIPTORS_EXPORT std::string calcMolFormula(
    const ROMol &mol, bool separateIsotopes = false,
    bool abbreviateHIsotopes = true);
/*!
  Calculates the total number of valence electrons for a molecule

  	param mol        the molecule of interest

  \return the total number of valence electrons
*/
RDKIT_DESCRIPTORS_EXPORT extern const std::string numValenceElectronsVersion;
RDKIT_DESCRIPTORS_EXPORT unsigned int calcNumValenceElectrons(const ROMol &mol);
/*!
  Calculates the total number of radical electrons for a molecule

  \param mol        the molecule of interest

  \return the total number of radical electrons
*/
RDKIT_DESCRIPTORS_EXPORT extern const std::string numRadicalElectronsVersion;
RDKIT_DESCRIPTORS_EXPORT unsigned int calcNumRadicalElectrons(const ROMol &mol);
/*!
  Calculates the average molecular weight of only heavy atoms

  \param mol        the molecule of interest

  \return the heavy atom molecular weight
*/
RDKIT_DESCRIPTORS_EXPORT extern const std::string heavyAtomMolWtVersion;
RDKIT_DESCRIPTORS_EXPORT double calcHeavyAtomMolWt(const ROMol &mol);
/*!
  Calculates Chi0 molecular connectivity index

  \param mol        the molecule of interest

  \return the Chi0 index
*/
RDKIT_DESCRIPTORS_EXPORT extern const std::string chi0Version;
RDKIT_DESCRIPTORS_EXPORT double calcChi0(const ROMol &mol);
/*!
  Calculates Chi1 molecular connectivity index

  \param mol        the molecule of interest

  \return the Chi1 index
*/
RDKIT_DESCRIPTORS_EXPORT extern const std::string chi1Version;
RDKIT_DESCRIPTORS_EXPORT double calcChi1(const ROMol &mol);
/*!
  Calculates the maximum EState index

  \param mol        the molecule of interest

  \return the max EState value
*/
RDKIT_DESCRIPTORS_EXPORT extern const std::string maxEStateIndexVersion;
RDKIT_DESCRIPTORS_EXPORT double calcMaxEStateIndex(const ROMol &mol);
/*!
  Calculates the minimum EState index

  \param mol        the molecule of interest

  \return the min EState value
*/
RDKIT_DESCRIPTORS_EXPORT extern const std::string minEStateIndexVersion;
RDKIT_DESCRIPTORS_EXPORT double calcMinEStateIndex(const ROMol &mol);
/*!
  Calculates the maximum absolute EState index

  \param mol        the molecule of interest

  \return the max absolute EState value
*/
RDKIT_DESCRIPTORS_EXPORT extern const std::string maxAbsEStateIndexVersion;
RDKIT_DESCRIPTORS_EXPORT double calcMaxAbsEStateIndex(const ROMol &mol);
/*!
  Calculates the minimum absolute EState index

  \param mol        the molecule of interest

  \return the min absolute EState value
*/
RDKIT_DESCRIPTORS_EXPORT extern const std::string minAbsEStateIndexVersion;
RDKIT_DESCRIPTORS_EXPORT double calcMinAbsEStateIndex(const ROMol &mol);

}  // end of namespace Descriptors
}  // end of namespace RDKit

#endif
