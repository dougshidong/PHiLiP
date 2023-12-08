#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <math.h>
// My standard libraires:
#include "allocatePointersLib.h"
// My nonequilibrium libraries:
#include "NonEquilibriumVar.h"
#include "ReactiveVar.h"
#include "var.h"
#include "eos.h"
using namespace std;

namespace PHiLiP {
namespace RealGasConstants {

//==============================================
void GetNASACAP_TemperatureIndex(double T, int *Sp_TempIndex)
{
	double PercentOutRange;

	for(int i=0; i<N_species; i++)
	{
		Sp_TempIndex[i] = -1; // initialize
		if((T >= NASACAPTemperatureLimits[i][0]) && (T < NASACAPTemperatureLimits[i][1]))
		{
			Sp_TempIndex[i] = 0; // low temp
		}
		else if((T >= NASACAPTemperatureLimits[i][1]) && (T < NASACAPTemperatureLimits[i][2]))
		{
			Sp_TempIndex[i] = 1; // mid temp
		}
		else if((T >= NASACAPTemperatureLimits[i][2]) && (T <= NASACAPTemperatureLimits[i][3]))
		{
			Sp_TempIndex[i] = 2; // high temp
		}
		else
		{
			// Outside temperature range
			if(T < NASACAPTemperatureLimits[i][0])
			{
				PercentOutRange = 1.0 - T/NASACAPTemperatureLimits[i][0];
				PercentOutRange *= 100.0;
				if(PercentOutRange < 20.0)
				{
					Sp_TempIndex[i] = 0;
				}
				else
				{
					// Extrapolation -- Wilkhoit
					// NOTE: THE FOLLOWING IS TEMPORARY -- freeze temp at 20% out of range
					// T = 0.8*NASACAPTemperatureLimits[i][0]; // COMMENT TO UNFREEZE
					Sp_TempIndex[i] = 0;
				}
			}
			else if(T > NASACAPTemperatureLimits[i][3])
			{
				PercentOutRange = T/NASACAPTemperatureLimits[i][3] - 1.0;
				PercentOutRange *= 100.0;
				if(PercentOutRange < 20.0)
				{
					Sp_TempIndex[i] = 2;
				}
				else
				{
					// Extrapolation -- Wilkhoit
					// NOTE: THE FOLLOWING IS TEMPORARY -- freeze temp at 20% out of range
					// T = 1.2*NASACAPTemperatureLimits[i][3]; // COMMENT TO UNFREEZE
					Sp_TempIndex[i] = 2;
				}
			}
		}
	}
}
//==============================================
void CalcSpeciesInternalEnergy(double RT, double *H, double *Sp_TotIntEnergy)
{
	double T;
	T = RT/R_universal;
	
	for(int i=0; i<N_species; i++)
	{
		Sp_TotIntEnergy[i] = H[i] - Sp_EnthalpyFormation[i] - (T-T_ref)*R_universal; // [J/mol]
		Sp_TotIntEnergy[i] /= Sp_W[i]; // [J/kg]
		Sp_TotIntEnergy[i] += Sp_EnergyFormation[i]; // [J/kg]
	}
	/* Units: [J/kg] */
}
//===============================================
void NASACAP_GetHSCp(double *H, double *S, double *Cp_molar,
 double *Cv_molar, double RT)
{
	// return H, S, and Cp  (in J/mol-K)
	double T = RT/R_universal;

	GetNASACAP_TemperatureIndex(T, Sp_TempIndex);

	for(int i=0; i<N_species; i++)
	{
		if (Sp_TempIndex[i] != -1)
		{
			int T_Index = Sp_TempIndex[i];
			Cp_molar[i] = 0.0;
			S[i] = -0.5*NASACAPCoeffs[i][0][T_Index]*pow(T, -2.0) - NASACAPCoeffs[i][1][T_Index]/T + NASACAPCoeffs[i][2][T_Index]*log(T);
			H[i] = -NASACAPCoeffs[i][0][T_Index]*pow(T, -2.0) + NASACAPCoeffs[i][1][T_Index]*log(T)/T;

			for(int j=0; j<7; j++)
			{
				Cp_molar[i] += NASACAPCoeffs[i][j][T_Index]*pow(T, double(j-2));
				if(j > 1)
				{
					H[i] += NASACAPCoeffs[i][j][T_Index]*pow(T, double(j-2))/(double(j-1));
				}
				if(j > 2)
				{
					S[i] += NASACAPCoeffs[i][j][T_Index]*pow(T, double(j-2))/(double(j-2));	
				}
			}
			H[i] += NASACAPCoeffs[i][7][T_Index]/T;
			S[i] += NASACAPCoeffs[i][8][T_Index];
			
			Cp_molar[i] *= R_universal; // [J/mol-K]
			Cv_molar[i] = Cp_molar[i] - R_universal; // [J/mol-K] 
			H[i] *= RT; // [J/mol]
			S[i] *= R_universal; // [J/mol-K]
		}
	}
}
//===============================================
void GetHSCp(double *H, double *S, double *Cp_molar,
 double *Cv_molar, double RT)
{
	/* For Combustion */
	// return molar H, S, and Cp  (in cal, mol, K)
	double T;
	double *ThermalCoeffs = dvector(7);

	T = RT/R_universal;

	for(int i=0; i<N_species; i++)
	{
		if((T >= TemperatureLimits[i][0]) && (T < TemperatureLimits[i][2]))
		{
			for(int j=0; j<7; j++)
			{
				ThermalCoeffs[j] = NASACoeffsLowT[i][j];	
			}
		}
		else if(T >= TemperatureLimits[i][2])
		{
			for(int j=0; j<7; j++)
			{
				ThermalCoeffs[j] = NASACoeffsHighT[i][j];	
			}
		}
		else
		{
			for(int j=0; j<7; j++)
			{
				ThermalCoeffs[j] = NASACoeffsLowT[i][j];	
			}
		}
		Cp_molar[i] = 0.0;
		H[i] = 0.0;
		S[i] = ThermalCoeffs[0]*log(T);

		for(int j=0; j<5; j++)
		{
			Cp_molar[i] += ThermalCoeffs[j]*pow(T,double(j));
			H[i] += ThermalCoeffs[j]*pow(T,double(j))/double(j+1);
			if(j > 0)
			{
				S[i] += ThermalCoeffs[j]*pow(T,double(j))/double(j);
			}
		}

		Cp_molar[i] *= (R_universal/cal2Joule); // [cal/mol-K]
		Cv_molar[i] = Cp_molar[i] - (R_universal/cal2Joule); // [cal/mol-K]
		H[i] = (R_universal/cal2Joule)*T*(H[i] + ThermalCoeffs[5]/T); // [cal/mol]
		S[i] = (R_universal/cal2Joule)*(S[i] + ThermalCoeffs[6]); // [cal/mol-K]
		H_formation[i] = (R_universal/cal2Joule)*ThermalCoeffs[5]; // [cal/mol]
	}
	delete [] ThermalCoeffs;
}
//===============================================
double GetMeanMolecularWeight_MoleFractions(double *MoleFractions)
{
	double W_mean = 0.0;
	for(int i=0; i<N_species; i++)
	{
		W_mean += MoleFractions[i]*Sp_W[i];
		// mixture molecular weight [kg/mol]
	}
	return W_mean;
}
//===============================================
double GetMeanDensity_PTW(double P_mean,
 double RT, double W_mean)
{
	/* Only used for combustion */
	if(NonEqFlag == 1)
	{
		cout << "\nWARNING: Function GetMeanDensity_PTW() is only valid for combustion or neutral species flows.\n" << endl;
	}
	return P_mean*W_mean/(RT);
	// mixture density in [kg/m3]
}
//===============================================
void GetMolarity_MoleFractions(double *MolarityVec,
 double *MoleFractions, double W_mean, double rho_mean)
{
	for(int i=0; i<N_species; i++)
	{
		MolarityVec[i] = (1.0e-6)*MoleFractions[i]*rho_mean/W_mean;
		// molarity [mol/cm3]
	}
}
//===============================================
void GetMassFractions_MoleFractions(double *MassFractions,
 double *MoleFractions, double W_mean)
{
	for(int i=0; i<N_species; i++)
	{
		MassFractions[i] = MoleFractions[i]*Sp_W[i]/W_mean;
	}
}
//===============================================
double GetCpMean(double *MoleFractions, double *Cp_molar)
{
	double Cp_mean=0.0;

	for(int i=0; i<N_species; i++)
	{
		Cp_mean += Cp_molar[i]*MoleFractions[i];
	}
	return Cp_mean;
	// mixture Cp [cal/mol-K]
}
//===============================================
double GetCvMean(double *MoleFractions, double *Cv_molar)
{
	double Cv_mean=0.0;

	for(int i=0; i<N_species; i++)
	{
		Cv_mean += Cv_molar[i]*MoleFractions[i];
	}
	return Cv_mean;
	// mixture Cv [cal/mol-K]
}
//===============================================
void GetSpecieDensities_MassFractions(double *MassFractions,
 double rho_mean, double *Sp_Density)
{
	for(int i=0; i<N_species; i++)
	{
		Sp_Density[i] = MassFractions[i]*rho_mean;
	}
	// Units: [kg/m3]
}
//===============================================
double GetMeanPressure_SpecieDensities(double *RT, double *Sp_Density)
{
	/* Mixture pressure in [Pa] from Dalton's
	   law of partial pressures + ideal gas law */
	double P_mean = 0.0;

	for(int i=0; i<N_species; i++)
	{
		if((i == i_electron) && (nTEMP==3))
		{
			P_mean += Sp_Density[i]*RT[2]/Sp_W[i];
		}
		else
		{
			P_mean += Sp_Density[i]*RT[0]/Sp_W[i];
		}
	}
	return P_mean;
}
//===============================================
double GetMeanDensity_SpecieDensities(double *Sp_Density)
{
	double rho_mean = 0.0;

	for(int i=0; i<N_species; i++)
	{
		rho_mean += Sp_Density[i];
	}
	return rho_mean;
	// mixture density [kg/m3]
}
//===============================================
double GetMeanMolecularWeight_MeanRhoPT(double rho_mean,
 double RT, double P_mean)
{
	/* Can only be used for combustion */
	if(NonEqFlag == 1)
	{
		cout << "\nWARNING: Function GetMeanMolecularWeight_MeanRhoPT() is only valid for combustion or neutral species flows.\n" << endl;
	}
	return rho_mean*RT/P_mean;
	// mixture molecular weight [kg/mol]
}
//===============================================
void GetMoleFractions_Molarity(double *MoleFractions,
 double *MolarityVec, double W_mean, double rho_mean)
{
	for(int i=0; i<N_species; i++)
	{
		MoleFractions[i] = (1.0e6)*MolarityVec[i]*W_mean/rho_mean;
	}
}
//===============================================
void GetMolarity_rhoW(double *MolarityVec, double *Sp_Density)
{
	// W = MolecularWeight [Kg/mol]
	// rho = species density [Kg/m3]
	// MolarityVec = molar concentrations [mol/cm3]
	for(int i=0; i<N_species; i++)
	{
		MolarityVec[i] = (1.0e-6)*Sp_Density[i]/Sp_W[i];
	}
}
//===============================================
double GetMeanMolecularWeight_Molarity(double rho_mean,
 double *MolarityVec)
{
	double MolarityVec_Sum = 0.0;
	
	for(int i=0; i<N_species; i++)
	{
		MolarityVec_Sum += MolarityVec[i]; 
	}

	return rho_mean*(1.0e-6)/MolarityVec_Sum;
	// mixture molecular weight [kg/mol]
}
//===============================================
double GetMeanNumberDensity_Molarity(double *MolarityVec)
{
	double MolarityVec_Sum = 0.0;
	
	for(int i=0; i<N_species; i++)
	{
		MolarityVec_Sum += MolarityVec[i]; 
	}

	return MolarityVec_Sum*AvogadroConstant_without_OoM*pow(10.0, AvogadroConstant_OoM);
	// mixture number density [atoms/cm3]
}
//===============================================
void GetSpeciesDensity_Qvec(double *Qvec, double rho_mean, double *Sp_Density)
{
	if(N_species_Qvec != N_species)
	{
		Sp_Density[i_Sp_Dominant] = rho_mean;
	}

	// cout << " \n GetSpeciesDensity_Qvec() INPUTS: " << endl;
	// cout << " \t rho_mean = " << rho_mean << endl;
	// for(int i=0; i<N_species_Qvec; i++)
	// {
	// 	cout << "\t Qvec[" << i << "] = " << Qvec[i] << endl;
	// }

	for(int i=0; i<N_species_Qvec; i++)
	{
		Sp_Density[Sp_Index_Qvec[i]] = Qvec[i];
		
		if(N_species_Qvec != N_species)
		{
			Sp_Density[i_Sp_Dominant] -= Qvec[i];	
		}
	}

	// cout << " \n GetSpeciesDensity_Qvec() OUTPUTS: " << endl;
	// for(int i=0; i<N_species; i++)
	// {
	// 	cout << "\t Sp_Density[" << i << "] = " << Sp_Density[i] << endl;
	// }
	// species density [Kg/m3]
}
//===============================================
void GetSpeciesEnergyFormation(double *Sp_EnergyFormation) // TO DO: call this somewhere
{
	/* Species energy of formation at T_ref */
	for(int i=0; i<N_species; i++)
	{
		Sp_EnergyFormation[i] = (Sp_EnthalpyFormation[i] - R_universal*T_ref)/Sp_W[i];
	}
	/* Units: [J/kg] */
}
//===============================================
void GetSpeciesTransRotCv(double *Sp_TransRotCv)
{
	/* Species specific heat at constant volume for
	   translational-rotational mode */
	for(int i=0; i<N_species; i++)
	{	
		if((Sp_CharVibTemp[i] != 0.0) && (i != i_electron))
		{
			/* Diatomic species */
			Sp_TransRotCv[i] = 2.5*(R_universal/Sp_W[i]);
		}
		else
		{
			/* Monatomic species and electrons */
			Sp_TransRotCv[i] = 1.5*(R_universal/Sp_W[i]);
		}
	}
	// Units: [J/(kg-K)]
}
//===============================================
void CalcSpeciesTransRotEnergy(double *RT, double *e_trs)
{
	if(NonEqFlag == 1)
	{
		/* -- Nonequilibrium -- */
		double T_tr, T_ee;

		T_tr = RT[0]/R_universal;
		
		if(nTEMP == 3)
		{
			T_ee = RT[2]/R_universal;
		}
		
		for(int i=0; i<N_species; i++)
		{	
			if((i == i_electron) && (nTEMP == 3))
			{
				e_trs[i] = Sp_TransRotCv[i]*T_ee;
			}
			else
			{
				e_trs[i] = Sp_TransRotCv[i]*T_tr;
			}
			// species translational-rotational energy per unit mass [J/kg]
			/* Display trans-rot energies */
			// cout << "Species: " << Sp_name[i] << "\t e_trs [J/kg] = " << e_trs[i] << endl;
		}
	}
	else
	{
		cout << "ERROR: Function 'CalcSpeciesTransRotEnergy()' only valid for nonequilibrium thermodynamics." << endl;
	}
}
//===============================================
//===============================================
// NOTE: The following 3 functions are for 
//       the get temp root finding Newton method
//===============================================
double f_GetTemp_Newton(double *Qvec, double *Sp_Density, double RT_)
{
	double f_val = Qvec[i_VibEnergy]; // TO DO: MODIFY THIS
	double *e_s = dvector(N_species);
	
	NASACAP_GetHSCp(H, S, Cp_molar, Cv_molar, RT_);
	CalcSpeciesInternalEnergy(RT_, H, e_s);
	
	for(int i=0; i<N_species; i++)
	{
		f_val -= Sp_Density[i]*e_s[i];
	}

	/* Delete local pointers */
	delete [] e_s;

	return f_val;
}
//===============================================
void CalcInternalEnergyDerivative_NASACAP(double T, double *de_s)
{
	GetNASACAP_TemperatureIndex(T, Sp_TempIndex);

	for(int i=0; i<N_species; i++)
	{
		int T_Index = Sp_TempIndex[i];

		de_s[i] = 0.0;
		for(int j=0; j<7; j++)
		{
			de_s[i] += NASACAPCoeffs[i][j][T_Index]*pow(T, double(j-2));
		}
		de_s[i] *= R_universal; // [J/mol] dH/dT
		de_s[i] -= R_universal; // [J/mol] dH/dT - R
		de_s[i] /= Sp_W[i]; // [J/kg]
	}
}
//===============================================
double df_GetTemp_Newton(double *Sp_Density, double RT_)
{
	double *de_s = dvector(N_species);

	double T_ = RT_/R_universal;

	CalcInternalEnergyDerivative_NASACAP(T_,de_s);

	double df_val = 0.0;
	for(int i=0; i<N_species; i++)
	{
		df_val -= Sp_Density[i]*de_s[i];
	}

	/* Delete local pointers */
	delete [] de_s;

	return df_val;
}
//===============================================
double GetTemp(double *Qvec, double *Sp_Density, double r0)
{
	/* Newton's method for root finding */
	double r1 = r0;
	double err = 1.1;
	double dr;
	int i = 0;
	int maxIter = 5000;
	
	while(err > RootFindingTolerance_Vib)
	{
		r0 = r1;
		dr = f_GetTemp_Newton(Qvec, Sp_Density, r0)/df_GetTemp_Newton(Sp_Density, r0);
		r1 = r0 - dr;
		err = abs(r1-r0);
		i += 1;
		
		// if(i%100 == 0)
		// {
		// 	cout << "\t i = " << i << "\t error = " << err << endl;
		// }

		if(i>maxIter)
		{
			cout << "ERROR: GetTemp() reached maximum number of iterations." << endl;
			break;	
		}
	}
	return r1;
}
//===============================================
void GetEnergyFormation(double *Sp_Density)
{
	/* Remove this later on by incorporating it directly were needed */
	EnergyFormation = 0.0;
	/* Mixture energy of formation at T_ref */
	for(int i=0; i<N_species; i++)
	{
		EnergyFormation += Sp_Density[i]*Sp_EnergyFormation[i];
	}
	/* Units: [J/m3] */
}
//===============================================

} // RealGasConstants namespace
} // PHiLiP namespace