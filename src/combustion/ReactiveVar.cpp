#include <iostream>
#include <fstream> // for writing to files
#include <string> // for strings
#include <stdlib.h>
#include <stdio.h>
#include "ReactiveVar.h"

namespace PHiLiP {
namespace RealGasConstants {

/* Chemistry read file --> Dictates what physics are being simulated */
/* Initial mixture read file --> Dictates what the initial composition of the gas mixture is */

/* GRI-Mech Methane Combustion */
// std::string namechem = "./ReactionFiles/grimech30.kinetics";
// std::string nameinitmixfrac = "./InitialMixtureFiles/InitialMixtureFractions_grimech30.txt";

/* Clarey 2019 JTHT: */
// std::string namechem = "./ReactionFiles/airmech_clarey2019.kinetics";
// std::string nameinitmixfrac = "./InitialMixtureFiles/InitialMixtureFractions_Clarey2019.txt";

/* Clarey 'nitrogen': */
// std::string namechem = "./ReactionFiles/airmech_clarey2019_nitrogen.kinetics";
// std::string nameinitmixfrac = "./InitialMixtureFiles/InitialMixtureFractions_Clarey_nitrogen.txt";

/* Clarey '5species': */
// std::string namechem = "./ReactionFiles/airmech_clarey2019_5species.kinetics";
// std::string nameinitmixfrac = "./InitialMixtureFiles/InitialMixtureFractions_Clarey_5species.txt";

/* '11species': */
std::string namechem = "./ReactionFiles/airmech_clarey2019.kinetics";
std::string nameinitmixfrac = "./InitialMixtureFiles/InitialMixtureFractions_11species.txt";

/* Ibraguimova 'oxygen': */
// std::string namechem = "./ReactionFiles/airmech_clarey2019_oxygen.kinetics";
// std::string namechem = "./ReactionFiles/airmech_luo2018_oxygen.kinetics";
// std::string nameinitmixfrac = "./InitialMixtureFiles/InitialMixtureFractions_oxygen.txt";

/* Equilibrium constant formulation --> For nonequilibrium, must correspond to namechem */
// std::string EqConstForm = "park1985convergence";
// std::string EqConstForm = "gnoffo1989nasa";
// std::string EqConstForm = "NumberDensityVariation";
std::string EqConstForm = "GibbsFreeEnergy"; // NASA CAP Program

/* Neutral dissociation reaction rate formulation */
// std::string neutralDissociationRxnRateForm = "MacheretFridman";
std::string neutralDissociationRxnRateForm = "Arrhenius";

std::string *Sp_name;

std::string fuel, oxidizer, dilutant;

bool TROE=true, LIND=false;

int N_elements, N_species, N_mechs, NonEqFlag, nTEMP, Max_ElecLevels;

/* Indices of relevant species (combustion and nonequilibrium) */
// Common between combustion and nonequilibrium
int i_O2, i_N2;
// Combustion
int i_CH4, i_CH3, i_AR, i_CO2, i_H2O;
 // Nonequilibrium
int i_N, i_O, i_NO, i_N_ion, i_O_ion, i_NO_ion, i_N2_ion, i_O2_ion, i_electron;
// Indices of fuel, oxidizer, dilutant (combustion)
int i_fuel, i_ox, i_dil, i_interm;

double cal2Joule = 4.184; // [J/cal]

int ***SpReactantIn, ***SpProductIn, **MechRPMat, **MechStoichMat,
    *MechMixtureFlag, *MechPressureFlag, *MechTroeFlag, *ReversibleFlag,
    *Sp_IonFlag, *Sp_ElecLevels, *Sp_eiiMech, *Sp_eidMech, *hpciReactionFlag,
    *MechEqConstCalcMethod, *BackwardFlag, *neutralDissociationReactionFlag,
    *dissocRxn_DissocSp, *dissocRxn_CollidingSp, *dissocRxn_DiatomMonatomFlag;

double *Sp_W, *Le_k, **NASACoeffsLowT, **NASACoeffsHighT, *VibEnergy_MacheretFridman_mech,
       **TemperatureLimits, *omega_dot, **omega_dot_mech, *H0, *S0, *Cp0, *H_T,
       *S_T, *Cp_T, **MechParametersArr, **MechParametersLowPressure,
       **MechParametersTroe, **MechMixtureCoeffs, *K_f, *K_r,
       *MechDeltaH_RT, *MechDeltaS_R, *q_mech, **MechTempExps,
       **MechEqConstCoeffs, **Sp_ElecDegeneracy, **Sp_CharElecTemp,
       **Sp_ElecNeutralSigmaConst, *Sp_FirstIonEnergy, *Sp_IonizationRate,
       *Sp_DissociationEnergy, *Sp_DissociationRate, **MechEqConstNumDensityRange,
       ***MechEqConstCoeffsNumDensity, *Sp_MilikanWhiteConstantA, *Sp_gRotElec,
       **NASACAPCoeffsLowT, **NASACAPCoeffsMidT, **NASACAPCoeffsHighT,
       **NASACAPTemperatureLimits, ***NASACAPCoeffs, *NASACAPSpeciesEnergyAtRefTemp;

// Mass electron rate of production from HPCI reactions
double omega_dot_hpci;

double Hs_mean, Es_mean, S_mean, Cp_mean, Cv_mean;

/* For initialization of mixture fractions */
int InitMixFrac_Mole_Flag, InitMixFrac_Mass_Flag, InitMixFrac_ERROR, SetInitEoS_ERROR;
double *InitialMixtureFractions;
int *TraceSpFlag;

} // RealGasConstants namespace
} // PHiLiP namespace