#ifndef NONEQUILIBRIUMVAR_H
#define NONEQUILIBRIUMVAR_H

#include <iostream>
#include <fstream> // for writing to files
#include <string> // for strings
#include <stdlib.h>
#include <stdio.h>

namespace PHiLiP {
namespace RealGasConstants {

extern double Q_TransRot_Vib, VibEnergyReaction, EnergyFormation, 
 Q_Trans_Elec, Q_Elec_Vib, ElectronicEnergyReaction, Q_eii, Q_eid, Q_hpci,
 Q_Trans_Elec_debug;

extern double Sp_TraceInitMoleFrac;
extern double RT_ee_old, bisecIntervalFactor, RootFindingTolerance_Vib, RootFindingTolerance_Elec;

extern int *Sp_TempIndex;
extern double *Sp_CharVibTemp, *Sp_EnthalpyFormation, *Sp_EnergyFormation,
 *Sp_TransRotCv, *Sp_VibEnergy, *Sp_TransRotEnergy, *Sp_ElectronicEnergy,
 *Sp_TotIntEnergy, *Sp_VibEnergy_Ref, *Sp_TransRotEnergy_Ref,
 *Sp_ElectronicEnergy_Ref, *Rxn_MacheretFridman_MassFraction;

extern int i_VibEnergy, i_TotEnergy, i_ElecEnergy;

extern int ElecPolyDeg, NumElecPolys, k_ElecTempInt;
extern double *ElecTempIntUpper;
extern double ***Sp_ElecEnergyCoeff;
extern std::string ElecEnergyMethod, VibEnergyForm, ParkCorrectionForm;
extern double DensityTolerance;

/* Debugging */
extern double global_time;
extern int debugDummyInt;

} // RealGasConstants namespace
} // PHiLiP namespace

#endif