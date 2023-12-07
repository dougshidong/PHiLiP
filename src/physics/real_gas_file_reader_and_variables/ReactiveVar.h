#ifndef REACTIVEVAR_H
#define REACTIVEVAR_H

#include <iostream>
#include <fstream> // for writing to files
#include <string> // for strings
#include <stdlib.h>
#include <stdio.h>

namespace PHiLiP {
namespace RealGasConstants {

extern std::string namechem, EqConstForm, nameinitmixfrac, neutralDissociationRxnRateForm;

extern std::string *Sp_name;

extern std::string fuel, oxidizer, dilutant;

extern bool TROE, LIND;

extern int N_elements, N_species, N_mechs, NonEqFlag, nTEMP, Max_ElecLevels;

/* Indices of relevant species (combustion and nonequilibrium) */
// Common between combustion and nonequilibrium
extern int i_O2, i_N2;
// Combustion
extern int i_CH4, i_CH3, i_AR, i_CO2, i_H2O; // combustion
 // Nonequilibrium
extern int i_N, i_O, i_NO, i_N_ion, i_O_ion, i_NO_ion, i_N2_ion, i_O2_ion, i_electron;
// Indices of fuel, oxidizer, dilutant (combustion)
extern int i_fuel, i_ox, i_dil, i_interm;

extern double cal2Joule;

extern int ***SpReactantIn, ***SpProductIn, **MechRPMat, **MechStoichMat,
 *MechMixtureFlag, *MechPressureFlag, *MechTroeFlag, *ReversibleFlag,
 *Sp_IonFlag, *Sp_ElecLevels, *Sp_eiiMech, *Sp_eidMech, *hpciReactionFlag,
 *MechEqConstCalcMethod, *BackwardFlag, *neutralDissociationReactionFlag,
 *dissocRxn_DissocSp, *dissocRxn_CollidingSp, *dissocRxn_DiatomMonatomFlag;

extern double *Sp_W, *Le_k, **NASACoeffsLowT, **NASACoeffsHighT, *VibEnergy_MacheretFridman_mech,
 **TemperatureLimits, *omega_dot, **omega_dot_mech, *H0, *S0, *Cp0, *H_T,
  *S_T, *Cp_T, **MechParametersArr, **MechParametersLowPressure,
   **MechParametersTroe, **MechMixtureCoeffs, *K_f, *K_r,
    *MechDeltaH_RT, *MechDeltaS_R, *q_mech, **MechTempExps,
     **MechEqConstCoeffs, **Sp_ElecDegeneracy, **Sp_CharElecTemp,
     **Sp_ElecNeutralSigmaConst, *Sp_FirstIonEnergy, *Sp_IonizationRate,
      *Sp_DissociationRate, *Sp_DissociationEnergy, **MechEqConstNumDensityRange,
       ***MechEqConstCoeffsNumDensity, *Sp_MilikanWhiteConstantA, *Sp_gRotElec,
       **NASACAPCoeffsLowT, **NASACAPCoeffsMidT, **NASACAPCoeffsHighT,
       **NASACAPTemperatureLimits, ***NASACAPCoeffs, *NASACAPSpeciesEnergyAtRefTemp;

extern double omega_dot_hpci;

extern double Hs_mean, Es_mean, S_mean, Cp_mean, Cv_mean;

extern int InitMixFrac_Mole_Flag, InitMixFrac_Mass_Flag, InitMixFrac_ERROR, SetInitEoS_ERROR;
extern double *InitialMixtureFractions;
extern int *TraceSpFlag;

} // RealGasConstants namespace
} // PHiLiP namespace

#endif