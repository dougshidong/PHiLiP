#ifndef VAR_H
#define VAR_H

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>

namespace PHiLiP {
namespace RealGasConstants {

/* Switches */
extern std::string debugRHS_Mode, debugINT_Mode;

/* Setup parameters */
extern int nEQ;

/* Temporary variables for running Dr. Clarey's Sp. Prod. Souce Term (omega_dot) */
extern int ClareySpProd_Flag;
extern int NOUT_Clarey;
extern double DELTA_T_CLAREY;
extern double **omega_dot_Clarey_store, *time_Clarey_store;

/* Physical and mathematical constants */
extern double R_universal, gammaGas, P_atm,
 AvogadroConstant_without_OoM, BoltzmannConstant_without_OoM,
  PI, T_ref, P_ref, ElecChargeConstant_without_OoM, ElecMass_without_OoM,
   VacuumPermittivity_without_OoM, AvogadroConstant_OoM,
   ElecChargeConstantESU_without_OoM, ElecChargeConstantESU_OoM;

extern double *MolarityVec, *MoleFractions, *MassFractions, *Sp_Density;
extern double W_mean, rho_mean, P_mean, NumDensity_mean;
extern double *H, *S, *Cp;

/* Combustion */
extern double *Cp_molar, *Cv_molar, *H_formation;
extern double T_dot;

/* */
extern double *Qvec, *RT;
extern int i_Sp_Dominant, *Sp_Index_Qvec, N_species_Qvec;

/* Time advancement */
extern int NOUT;
extern double *t_store;

/* Post processing */
extern double **T_store, **MoleFractions_store, **MolarityVec_store, 
	**MassFractions_store, **Sp_Density_store, **Qvec_store, *y_store, *Q_eii_store,
	*Q_eid_store, *Q_hpci_store, *Q_TransRot_Vib_store, 
	*Q_Trans_Elec_store, *Q_Trans_Elec_store_debug, 
	*Q_Elec_Vib_store, *ElectronicEnergyReaction_store, 
	*NumDensity_mean_store, *P_mean_store, *rho_mean_store, 
	*W_mean_store, **omega_dot_store;

} // RealGasConstants namespace
} // PHiLiP namespace

#endif