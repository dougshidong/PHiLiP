#ifndef EOS_H
#define EOS_H

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>

namespace PHiLiP {
namespace RealGasConstants {

void GetNASACAP_TemperatureIndex(double, int*);
void CalcSpeciesInternalEnergy(double, double*, double*);
void NASACAP_GetHSCp(double*, double*, double*, double*, double);
void GetHSCp(double*, double*, double*, double*, double);
double GetMeanMolecularWeight_MoleFractions(double*);
double GetMeanDensity_PTW(double, double, double);
void GetMolarity_MoleFractions(double*, double*, double, double);
void GetMassFractions_MoleFractions(double*, double*, double);
double GetCpMean(double*, double*);
double GetCvMean(double*, double*);
void GetSpecieDensities_MassFractions(double*, double, double*);
double GetMeanPressure_SpecieDensities(double*, double*);
double GetMeanDensity_SpecieDensities(double*);
double GetMeanMolecularWeight_MeanRhoPT(double, double, double);
void GetMoleFractions_Molarity(double*, double*, double, double);
void GetMolarity_rhoW(double*, double*);
double GetMeanMolecularWeight_Molarity(double, double*);
double GetMeanNumberDensity_Molarity(double*);
void GetSpeciesDensity_Qvec(double*, double, double*);

} // RealGasConstants namespace
} // PHiLiP namespace

#endif