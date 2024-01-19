#include <iostream>
#include <fstream> // for writing to files
#include <string> // for strings
#include <stdlib.h>
#include <stdio.h>
#include "var.h"

namespace PHiLiP {
namespace RealGasConstants {

/* Switches */
std::string debugRHS_Mode = "OFF";
std::string debugINT_Mode = "ON";

/* Setup parameters */
int nEQ;

/* Temporary variables for running Dr. Clarey's Sp. Prod. Souce Term (omega_dot) */
int ClareySpProd_Flag = 0;
int NOUT_Clarey;
double DELTA_T_CLAREY;
double **omega_dot_Clarey_store, *time_Clarey_store;

/* Physical and mathematical constants */
double R_universal = 8.314510;//4598; // [J/mol-K]  8.314 originally
double gammaGas = 1.4;
double P_atm = 101325.0; // [Pa]
double AvogadroConstant_without_OoM = 6.022140857; // [atoms/mol] e23
double BoltzmannConstant_without_OoM = 1.38064852;// [J/K] e-23
double AvogadroConstant_OoM = 23.0;
double PI = 3.14159265359;
double T_ref = 298.15; // [K] - reference temperature for formation enthalpy and energy (NASA CAP)
double P_ref = 1.0e5; // (1 bar) - NASA CAP Reference pressure -- for combustion: P_ref = P_atm
double ElecChargeConstant_without_OoM = 1.60207; // [C] e-19
double ElecChargeConstantESU_without_OoM = 4.80298; // [esu] e-10
double ElecChargeConstantESU_OoM = -10.0; // [esu] e-10
double ElecMass_without_OoM = 9.10938356; // [kg] e-31
double VacuumPermittivity_without_OoM = 8.854; // [C/(V*m)] e-12
// Sp_W of e is 0.0005485799039e-3 --> Change CAP ?? --> See McBride 1999

/* Equation of State */
double *MolarityVec, *MoleFractions, *MassFractions, *Sp_Density;
double W_mean, rho_mean, P_mean, NumDensity_mean;
double *H, *S, *Cp;

/* Combustion */
double *Cp_molar, *Cv_molar, *H_formation;
double T_dot;

/* PDE */
double *Qvec, *RT;
int i_Sp_Dominant, *Sp_Index_Qvec, N_species_Qvec;

/* Time advancement */
int NOUT;
double *t_store;

/* Post processing */
double **T_store, **MoleFractions_store, **MolarityVec_store, 
	**MassFractions_store, **Sp_Density_store, **Qvec_store, *y_store, *Q_eii_store,
	*Q_eid_store, *Q_hpci_store, *Q_TransRot_Vib_store, 
	*Q_Trans_Elec_store, *Q_Trans_Elec_store_debug, 
	*Q_Elec_Vib_store, *ElectronicEnergyReaction_store, 
	*NumDensity_mean_store, *P_mean_store, *rho_mean_store, 
	*W_mean_store, **omega_dot_store;

} // RealGasConstants namespace
} // PHiLiP namespace