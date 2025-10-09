/* 
===============================================
McGill University
-----------------------------------------------
Julien Brillon
-----------------------------------------------
Includes:
- C++ script for reading ChemKin files
===============================================
*/
#include <iostream>
#include <fstream> // for writing to files
#include <string> // for strings
#include <stdlib.h>
#include <stdio.h>
#include "allocatePointersLib.h"
#include "strManipLib.h"
#include "readspecies.h"
#include "ReactiveVar.h"
#include "NonEquilibriumVar.h"

using namespace std;

namespace PHiLiP {
namespace RealGasConstants {
//===============================================
// string delSpaces(string str) 
// {
// 	str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
// 	return str;
// }
//===============================================
void getSpeciesIndex(int i, string sp)
{
	if(sp == "N")
	{
		i_N = i;
		// cout << "\t" << i_N;
	}
	if(sp == "N2")
	{
		i_N2 = i;
		// cout << "\t" << i_N2;
	}
	if(sp == "O")
	{
		i_O = i;
		// cout << "\t" << i_O;
	}
	if(sp == "O2")
	{
		i_O2 = i;
		// cout << "\t" << i_O2;
	}
	if(sp == "NO")
	{
		i_NO = i;
		// cout << "\t" << i_NO;
	}
	if(sp == "N+")
	{
		i_N_ion = i;
		// cout << "\t" << i_N_ion;
	}
	if(sp == "N2+")
	{
		i_N2_ion = i;
		// cout << "\t" << i_N2_ion;
	}
	if(sp == "O+")
	{
		i_O_ion = i;
		// cout << "\t" << i_O_ion;
	}
	if(sp == "O2+")
	{
		i_O2_ion = i;
		// cout << "\t" << i_O2_ion;
	}
	if(sp == "NO+")
	{
		i_NO_ion = i;
		// cout << "\t" << i_NO_ion;
	}
	if(sp == "e-")
	{
		i_electron = i;
		// cout << "\t" << i_electron;
	}
	if(sp == "CH4")
	{
		i_CH4 = i;
		// cout << "\t" << i_CH4;
	}
	if(sp == "CH3")
	{
		i_CH3 = i;
		// cout << "\t" << i_CH3;
	}
	if(sp == "AR")
	{
		i_AR = i;
		// cout << "\t" << i_AR;
	}
	if(sp == "CO2")
	{
		i_CO2 = i;
		// cout << "\t" << i_CO2;
	}
	if(sp == "H2O")
	{
		i_H2O = i;
		// cout << "\t" << i_H2O;
	}
}
//===============================================
void readspecies(std::string reactionFilename)
{
	int ii, jj;
    string line, dum_char;

	ifstream REACTF (reactionFilename);
	
	getline(REACTF, line);
	getline(REACTF, line);
	N_elements = (int)stof(line);
	getline(REACTF, line);
	getline(REACTF, line);
	N_species = (int)stof(line);
	getline(REACTF, line);
	getline(REACTF, line);
	N_mechs = (int)stof(line);
	getline(REACTF, line);
	getline(REACTF, line);
	NonEqFlag = (int)stof(line);
	
	if(NonEqFlag == 1)
	{
		getline(REACTF, line);
		getline(REACTF, line);
		nTEMP = (int)stof(line);

		if(nTEMP == 3)
		{
			getline(REACTF, line);
			getline(REACTF, line);
			Max_ElecLevels = (int)stof(line);
			getline(REACTF, line);
			getline(REACTF, line);
			ElecPolyDeg = (int)stof(line);
			getline(REACTF, line);
			getline(REACTF, line);
			NumElecPolys = (int)stof(line);
		}
	}
	else
	{
		/* Combustion */
		nTEMP = 1;
	}

	// cout << "Total chemical elements: " << N_elements << endl;
	// cout << "Total species: " << N_species << endl;
	// cout << "Total reaction mechanisms: " << N_mechs << endl;
	// cout << "NonEqFlag: " << NonEqFlag << endl;
	//===============================================
	// Allocate integer types
	SpReactantIn = iblock(N_species, N_mechs, 2); 
    SpProductIn = iblock(N_species, N_mechs, 2);
	MechRPMat = imatrix(N_mechs,6); 
    MechStoichMat = imatrix(N_mechs,6);
	MechMixtureFlag = ivector(N_mechs);
    MechPressureFlag = ivector(N_mechs); 
    MechTroeFlag = ivector(N_mechs); 
    ReversibleFlag = ivector(N_mechs);
    neutralDissociationReactionFlag = ivector(N_mechs);
    dissocRxn_DissocSp = ivector(N_mechs);
    dissocRxn_CollidingSp = ivector(N_mechs);
    dissocRxn_DiatomMonatomFlag = ivector(N_mechs);
    hpciReactionFlag = ivector(N_mechs);
    Sp_IonFlag = ivector(N_species);
    MechEqConstCalcMethod = ivector(N_mechs);
    BackwardFlag = ivector(N_mechs);
	/* 3T model specific */
	Sp_ElecLevels = ivector(N_species);
	Sp_eiiMech = ivector(N_species);
	Sp_eidMech = ivector(N_species);
    
    // Init
	for(int k=0; k<2; k++)
	{
		for(int i=0; i<N_species; i++)
		{
			for(int j=0; j<N_mechs; j++)
			{
				if(k==0)
				{
					SpReactantIn[i][j][k] = -1;
					SpProductIn[i][j][k] = -1;	
				}
				else if(k==1)
				{
					SpReactantIn[i][j][k] = 0;
					SpProductIn[i][j][k] = 0;	
				}
			}
		}
	}
    
	// Allocate double types
	Sp_W = dvector(N_species); 
	Le_k = dvector(N_species); 
	NASACoeffsLowT = dmatrix(N_species,7);
    NASACoeffsHighT = dmatrix(N_species,7); 
    TemperatureLimits = dmatrix(N_species,3);
    NASACAPCoeffs = dblock(N_species,9,3);
    NASACAPCoeffsLowT = dmatrix(N_species,9);
    NASACAPCoeffsMidT = dmatrix(N_species,9);
    NASACAPCoeffsHighT = dmatrix(N_species,9);
    NASACAPTemperatureLimits = dmatrix(N_species,4);
    omega_dot = dvector(N_species);
    omega_dot_mech = dmatrix(N_species,N_mechs);
    VibEnergy_MacheretFridman_mech = dvector(N_mechs);
    H0 = dvector(N_species); 
    S0 = dvector(N_species); 
    Cp0 = dvector(N_species); 
    H_T = dvector(N_species); 
    S_T = dvector(N_species); 
    Cp_T = dvector(N_species);
    MechParametersArr = dmatrix(N_mechs,3);
    MechParametersLowPressure = dmatrix(N_mechs,3); 
    MechParametersTroe = dmatrix(N_mechs,4); 
    MechMixtureCoeffs = dmatrix(N_mechs,N_species);
    K_f = dvector(N_mechs); 
    K_r = dvector(N_mechs); 
    MechDeltaH_RT = dvector(N_mechs); 
    MechDeltaS_R = dvector(N_mechs); 
    q_mech = dvector(N_mechs);
    Sp_CharVibTemp = dvector(N_species);
    Sp_EnthalpyFormation = dvector(N_species);
    Sp_MilikanWhiteConstantA = dvector(N_species);
    MechTempExps = dmatrix(N_mechs,2*nTEMP);
    MechEqConstCoeffs = dmatrix(N_mechs,5);
    MechEqConstNumDensityRange = dmatrix(N_mechs,6);
    MechEqConstCoeffsNumDensity = dblock(6,6,N_mechs);
	/* 3 Temperature model specific */
	Sp_ElecDegeneracy = dmatrix(N_species,Max_ElecLevels);
	Sp_CharElecTemp = dmatrix(N_species,Max_ElecLevels);
	Sp_ElecNeutralSigmaConst = dmatrix(N_species,3);
	Sp_DissociationEnergy = dvector(N_species);
	Sp_gRotElec = dvector(N_species);
	Sp_FirstIonEnergy = dvector(N_species);
	Sp_IonizationRate = dvector(N_species);
	Sp_DissociationRate = dvector(N_species);
	ElecTempIntUpper = dvector(NumElecPolys);
	Sp_ElecEnergyCoeff = dblock(N_species, ElecPolyDeg+1, NumElecPolys);

	string dummy_name;
	string::size_type sz1;

	getline(REACTF, line);
	getline(REACTF, line);
	if(nTEMP == 3)
	{
		/*-------------------------------------------
	 	 * 	Upper limit of temperature intervals
	 	 *  SECTION for electronic energies
	 	 *-------------------------------------------*/
		sz1 = 0;
		getline(REACTF, line);
		line = line.substr(11);
		for(int i=0; i<NumElecPolys; i++)
		{
			ElecTempIntUpper[i] = stod(line,&sz1);
			line = line.substr(sz1);
			sz1 = 0;
			// cout << "\t" << ElecTempIntUpper[i];
		}
		sz1 = 0;
		getline(REACTF, line);
	}
	// cout << endl;

    // Allocate string types
    Sp_name = svector(N_species);
    //===============================================
	/*-------------------------------------------
	 * 			 SPECIES SECTION
	 *-------------------------------------------*/
	for(int i=0; i<N_species; i++)
	{
		// Init
		sz1 = 0;
		getline(REACTF, line);
		dummy_name = line.substr(11,21);
		line = line.substr(32);
		Sp_name[i] = delSpaces(dummy_name);
		Sp_W[i] = stod(line,&sz1); // Species molecular weight [kg/mol]
		
		// cout << Sp_name[i] << "\t";
		// cout << Sp_W[i] << "\t";
		
		if(NonEqFlag == 1)
		{
			line = line.substr(sz1);
			sz1 = 0;
			Sp_CharVibTemp[i] = stod(line,&sz1);
			// cout << "\t" << Sp_CharVibTemp[i];
			line = line.substr(sz1);
			sz1 = 0;
			Sp_EnthalpyFormation[i] = stod(line,&sz1); // [J/mol]
			// cout << "\t" << Sp_EnthalpyFormation[i];
			line = line.substr(sz1);
			sz1 = 0;
			Sp_MilikanWhiteConstantA[i] = stod(line,&sz1);
			// cout << "\t" << Sp_MilikanWhiteConstantA[i];
			line = line.substr(sz1);
			sz1 = 0;
			Sp_IonFlag[i] = (int)stof(line,&sz1);
			// cout << "\t" << Sp_IonFlag[i];
			i_electron = -1; // initialized value
			
		}
		getSpeciesIndex(i, Sp_name[i]);
		// cout << endl;

		if(NonEqFlag == 1)
		{
			for(int j=0; j<4; j++)
			{
				line = line.substr(sz1);
				sz1 = 0;
				NASACAPTemperatureLimits[i][j] = stod(line,&sz1);
				// cout << "\t" << NASACAPTemperatureLimits[i][j];
			}
		}
		else
		{
			for(int j=0; j<3; j++)
			{
				line = line.substr(sz1);
				sz1 = 0;
				TemperatureLimits[i][j] = stod(line,&sz1);
				// cout << "\t" << TemperatureLimits[i][j];
			}
		}
		// cout << endl;

		// Init
		sz1 = 0;
		getline(REACTF, line);
		line = line.substr(11);
		if(NonEqFlag == 1)
		{
			for(int j=0; j<9; j++)
			{
				NASACAPCoeffs[i][j][0] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
			}
		}
		else
		{
			for(int j=0; j<7; j++)
			{
				NASACoeffsLowT[i][j] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
				// cout << "\t" << NASACoeffsLowT[i][j];
			}
		}
		// cout << endl;


		if(NonEqFlag == 1)
		{
			// Init
			sz1 = 0;
			getline(REACTF, line);
			line = line.substr(11);
			for(int j=0; j<9; j++)
			{
				NASACAPCoeffs[i][j][1] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
			}
		}
		
		// Init
		sz1 = 0;
		getline(REACTF, line);
		line = line.substr(11);
		if(NonEqFlag == 1)
		{
			for(int j=0; j<9; j++)
			{
				NASACAPCoeffs[i][j][2] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
				// cout << "\t" << NASACAPCoeffsHighT[i][j];
			}
		}
		else
		{
			for(int j=0; j<7; j++)
			{
				NASACoeffsHighT[i][j] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
				// cout << "\t" << NASACoeffsLowT[i][j];
			}
		}
		// cout << endl;

		if(nTEMP == 3)
		{
			sz1 = 0;
			getline(REACTF, line);
			line = line.substr(11);
			
			Sp_ElecLevels[i] = (int)stof(line,&sz1);
			line = line.substr(sz1);
			sz1 = 0;
			// cout << Sp_name[i] << "\t" << Sp_ElecLevels[i];
			
			/* Get electronic energy level degeneracies */			
			for(int j=0; j<Max_ElecLevels; j++)
			{
				Sp_ElecDegeneracy[i][j] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
				// cout << "\t" << Sp_ElecDegeneracy[i][j];
			}
			// cout << endl;

			/* Get characteristic electronic temperatures */
			sz1 = 0;
			getline(REACTF, line);
			line = line.substr(11);
			for(int j=0; j<Max_ElecLevels; j++)
			{
				Sp_CharElecTemp[i][j] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
				// cout << "\t" << Sp_CharElecTemp[i][j];
			}
			// cout << endl;

			/* Get species electron-neutral cross-section constants */
			// cout << Sp_name[i];
			sz1 = 0;
			getline(REACTF, line);
			line = line.substr(11);
			for(int j=0; j<3; j++)
			{
				Sp_ElecNeutralSigmaConst[i][j] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
				// cout << "\t" << Sp_ElecNeutralSigmaConst[i][j];
			}

			/* Get reaction dependant information for electron source terms */
			sz1 = 0;
			getline(REACTF, line);
			line = line.substr(11);
			Sp_eiiMech[i] = (int)stof(line,&sz1);
			// cout << "\t" << Sp_eiiMech[i];
			line = line.substr(sz1);
			sz1 = 0;
			Sp_FirstIonEnergy[i] = stod(line,&sz1);
			// cout << "\t" << Sp_FirstIonEnergy[i];
			line = line.substr(sz1);
			sz1 = 0;
			Sp_eidMech[i] = (int)stof(line,&sz1);
			line = line.substr(sz1);
			sz1 = 0;
			Sp_DissociationEnergy[i] = stod(line,&sz1);
			line = line.substr(sz1);
			sz1 = 0;
			Sp_gRotElec[i] = stod(line,&sz1);
			// if(Sp_eidMech[i] != -1)
			// {
			// 	cout << "EID " << Sp_name[i] << "\t" << Sp_eidMech[i] << "\t" << Sp_DissociationEnergy[i] << endl;
			// }

			// cout << Sp_name[i] << endl;
			for(int k=0; k<NumElecPolys; k++)
			{
				sz1 = 0;
				getline(REACTF, line);
				line = line.substr(11);
				for(int j=0; j<ElecPolyDeg+1; j++)
				{
					Sp_ElecEnergyCoeff[i][j][k] = stod(line,&sz1);
					line = line.substr(sz1);
					sz1 = 0;
					// cout << "  " << Sp_ElecEnergyCoeff[i][j][k];
				}
				// cout << endl;
			}
			// cout << endl;
		}
		// cout << endl;
	}
	// cout << endl;

	getline(REACTF, line);
	getline(REACTF, line);

	/*-------------------------------------------
	 * 			 MECHANISM SECTION
	 *-------------------------------------------*/
	for(int i=0; i<N_mechs; i++) // N_mechs
	{
		sz1 = 0;
		getline(REACTF, line);
		line = line.substr(11);
		MechMixtureFlag[i] = (int)stof(line,&sz1);
		line = line.substr(sz1);
		sz1 = 0;
		MechPressureFlag[i] = (int)stof(line,&sz1);
		line = line.substr(sz1);
		sz1 = 0;
		MechTroeFlag[i] = (int)stof(line,&sz1);
		line = line.substr(sz1);
		sz1 = 0;
		ReversibleFlag[i] = (int)stof(line,&sz1);
		
		if(NonEqFlag == 1)
		{
			line = line.substr(sz1);
			sz1 = 0;
			neutralDissociationReactionFlag[i] = (int)stof(line,&sz1);
			if(nTEMP == 3)
			{
				line = line.substr(sz1);
				sz1 = 0;
				hpciReactionFlag[i] = (int)stof(line,&sz1);
			}
		}
		// cout << (i+1) << "\t" << MechMixtureFlag[i] << "\t" << MechPressureFlag[i]
		 // << "\t" << MechTroeFlag[i] << "\t" << ReversibleFlag[i] << endl;
		
		if(NonEqFlag == 1)
		{
			sz1 = 0;
			getline(REACTF, line);
			line = line.substr(11);
			dissocRxn_DissocSp[i] = (int)stof(line,&sz1);
			line = line.substr(sz1);
			sz1 = 0;
			dissocRxn_CollidingSp[i] = (int)stof(line,&sz1);
			line = line.substr(sz1);
			sz1 = 0;
			dissocRxn_DiatomMonatomFlag[i] = (int)stof(line,&sz1);
			line = line.substr(sz1);
			sz1 = 0;
		}

		sz1 = 0;
		getline(REACTF, line);
		line = line.substr(11);
		for(int j=0; j<3; j++)
		{
			MechParametersArr[i][j] = stod(line,&sz1);
			line = line.substr(sz1);
			sz1 = 0;
			// cout << "\t" << MechParametersArr[i][j];
		}
		if(NonEqFlag == 1)
		{
			for(int j=0; j<(2*nTEMP); j++)
			{
				MechTempExps[i][j] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
				// cout << "\t" << MechTempExps[i][j];
			}
			for(int j=0; j<5; j++)
			{
				MechEqConstCoeffs[i][j] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
				// cout << "\t" << MechEqConstCoeffs[i][j];
			}
		}
		// cout << endl;

		// NEW SECTION FOR EQ CONST
		if(NonEqFlag == 1)
		{
			sz1 = 0;
			getline(REACTF, line);
			line = line.substr(11);

			MechEqConstCalcMethod[i] = (int)stof(line,&sz1);
			line = line.substr(sz1);
			sz1 = 0;

			BackwardFlag[i] = (int)stof(line,&sz1);
			line = line.substr(sz1);
			sz1 = 0;

			for(int j=0; j<6; j++)
			{
				MechEqConstNumDensityRange[i][j] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
			}

			for(int k=0; k<6; k++)
			{
				sz1 = 0;
				getline(REACTF, line);
				line = line.substr(11);
				for(int j=0; j<6; j++)
				{
					MechEqConstCoeffsNumDensity[k][j][i] = stod(line,&sz1);
					line = line.substr(sz1);
					sz1 = 0;
				}
			}
		}


		sz1 = 0;
		getline(REACTF, line);
		line = line.substr(11);
		for(int j=0; j<6; j++)
		{
			if(NonEqFlag == 1)
			{
				MechRPMat[i][j] = (int)stof(line,&sz1);
			}
			else // Temporary patch for Fortran to C++ indexing conversion
			{
				MechRPMat[i][j] = (int)stof(line,&sz1) - 1;
			}
			line = line.substr(sz1);
			sz1 = 0;
			// cout << "\t" << MechRPMat[i][j];
		}
		// cout << endl;

		sz1 = 0;
		getline(REACTF, line);
		line = line.substr(11);
		for(int j=0; j<6; j++)
		{
			MechStoichMat[i][j] = (int)stof(line,&sz1);
			line = line.substr(sz1);
			sz1 = 0;
			// cout << "\t" << MechStoichMat[i][j];
		}
		// cout << endl;

		if(MechPressureFlag[i] == 1)
		{
			sz1 = 0;
			getline(REACTF, line);
			line = line.substr(11);
			for(int j=0; j<3; j++)
			{
				MechParametersLowPressure[i][j] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
				// cout << "\t" << MechParametersLowPressure[i][j];
			}
			// cout << endl;
		}

		if(MechTroeFlag[i] == 1)
		{
			sz1 = 0;
			getline(REACTF, line);
			line = line.substr(11);
			for(int j=0; j<4; j++)
			{
				MechParametersTroe[i][j] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
				// cout << "\t" << MechParametersTroe[i][j];
			}
			// cout << endl;
		}
		
		if(MechMixtureFlag[i] == 1)
		{
			sz1 = 0;
			getline(REACTF, line);
			line = line.substr(11);
			for(int j=0; j<N_species; j++)
			{
				MechMixtureCoeffs[i][j] = stod(line,&sz1);
				line = line.substr(sz1);
				sz1 = 0;
				// cout << "\t" << MechMixtureCoeffs[i][j];
			}
			// cout << endl;
		}
	}

	REACTF.close();
	// construct SpReactantIn and SpProductIn matrices in the solver
	for(int i=0; i<N_mechs; i++)
	{
		for(int j=0; j<3; j++)
		{
			ii = MechStoichMat[i][j];
			if(ii != 0)
			{
				ii = MechRPMat[i][j];
				SpReactantIn[ii][i][0] = i;
				SpReactantIn[ii][i][1] += MechStoichMat[i][j];
			}
			jj = MechStoichMat[i][j+3];
			if(jj != 0)
			{
				jj = MechRPMat[i][j+3];
				SpProductIn[jj][i][0] = i;
				SpProductIn[jj][i][1] += MechStoichMat[i][j+3];
			}
		}
	}

	for(int i=0; i<N_species; i++)
	{
		Le_k[i] = 1.0;
	}
}

} // RealGasConstants namespace
} // PHiLiP namespace