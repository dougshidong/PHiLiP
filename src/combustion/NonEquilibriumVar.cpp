#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
// My libraires:
#include "NonEquilibriumVar.h"

namespace PHiLiP {
namespace RealGasConstants {

double Q_TransRot_Vib, VibEnergyReaction, EnergyFormation,
 Q_Trans_Elec, Q_Elec_Vib, ElectronicEnergyReaction, Q_eii,
  Q_eid, Q_hpci, Q_Trans_Elec_debug;

double Sp_TraceInitMoleFrac; // Rename and move to ReactiveVar later on

double RT_ee_old, bisecIntervalFactor = 0.25;
double RootFindingTolerance_Vib = 1.0e-4;
double RootFindingTolerance_Elec = 1.0e-4;
/* ElecEnergyMethod options:
 * - partitionFxn
 * - partitionFxnApprox
 */
std::string ElecEnergyMethod = "partitionFxnApprox";
/* VibEnergyMethod options:
 * - Harmonic Oscillator
 * - NASA CAP
 */
// std::string VibEnergyForm = "Harmonic Oscillator";
std::string VibEnergyForm = "NASA CAP";
/* ParkCorrectionForm options:
 * - 1984
 * - 1993
 */
std::string ParkCorrectionForm = "1993";

double DensityTolerance = 0.0; // should always be zero

int *Sp_TempIndex;
double *Sp_CharVibTemp, *Sp_EnthalpyFormation, *Sp_EnergyFormation,
 *Sp_TransRotCv, *Sp_VibEnergy, *Sp_TransRotEnergy, *Sp_ElectronicEnergy,
 *Sp_TotIntEnergy, *Sp_VibEnergy_Ref, *Sp_TransRotEnergy_Ref,
 *Sp_ElectronicEnergy_Ref, *Rxn_MacheretFridman_MassFraction;

int i_VibEnergy, i_TotEnergy, i_ElecEnergy;

/* Curve fit variables */
int ElecPolyDeg, NumElecPolys, k_ElecTempInt;
double *ElecTempIntUpper;
double ***Sp_ElecEnergyCoeff;

/* Debugging */
double global_time;
int debugDummyInt = 0;

} // RealGasConstants namespace
} // PHiLiP namespace
