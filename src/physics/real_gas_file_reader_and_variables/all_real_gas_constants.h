#ifndef __REAL_GAS_CONSTANTS_H__
#define __REAL_GAS_CONSTANTS_H__

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include "readspecies.h"
// #include <string> // for strings

namespace PHiLiP {
namespace RealGasConstants {

/// Main parameter class that contains the various other sub-parameter classes.
class AllRealGasConstants
{
public:
    /// Constructor
    AllRealGasConstants();

    /// Destructor
    ~AllRealGasConstants() {};

    // Thermodynamic Nonequilibrium Flow Variables
    /*struct NonequilibriumVars
    {*/
        double Q_TransRot_Vib, VibEnergyReaction, EnergyFormation, 
         Q_Trans_Elec, Q_Elec_Vib, ElectronicEnergyReaction, Q_eii, Q_eid, Q_hpci,
         Q_Trans_Elec_debug;

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
        int debugDummyInt;
    /*}
    NonequilibriumVars noneq;*/

    // Reacting Flow Variable
    /*struct ReactiveVars
    {*/
        /* Chemistry read file --> Dictates what physics are being simulated */
        /* Initial mixture read file --> Dictates what the initial composition of the gas mixture is */

        /* GRI-Mech Methane Combustion */
        // std::string namechem = "/home/liki/Codes/NEQZD/ReactionFiles/grimech30.kinetics";
        // std::string nameinitmixfrac = "/home/liki/Codes/NEQZD/InitialMixtureFiles/InitialMixtureFractions_grimech30.txt";

        /* Clarey 2019 JTHT: */
        // std::string namechem = "./ReactionFiles/airmech_clarey2019.kinetics";
        // std::string nameinitmixfrac = "./InitialMixtureFiles/InitialMixtureFractions_Clarey2019.txt";

        /* Clarey 'nitrogen': */
        // std::string namechem = "/home/liki/Codes/NEQZD/ReactionFiles/airmech_clarey2019_nitrogen.kinetics";
        // std::string nameinitmixfrac = "/home/liki/Codes/NEQZD/InitialMixtureFractions_Clarey_nitrogen.txt";

        /* Clarey '5species': */
        // std::string namechem = "/home/liki/Codes/NEQZD/ReactionFiles/airmech_clarey2019_5species.kinetics";
        // std::string nameinitmixfrac = "/home/liki/Codes/NEQZD/InitialMixtureFiles/InitialMixtureFractions_Clarey_5species.txt";

        /* '11species': */
        // std::string namechem = "/home/liki/Codes/NEQZD/ReactionFiles/airmech_clarey2019.kinetics";
        // std::string nameinitmixfrac = "/home/liki/Codes/NEQZD/InitialMixtureFiles/InitialMixtureFractions_11species.txt";

        /* 'N2_O2': */
        std::string namechem = "/home/liki/Codes/NEQZD/ReactionFiles/N2_O2.kinetics";
        std::string nameinitmixfrac = "/home/liki/Codes/NEQZD/InitialMixtureFiles/InitialMixtureFractions_N2_O2.txt";

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
        int i_CH4, i_CH3, i_AR, i_CO2, i_H2O; // combustion
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
              *Sp_DissociationRate, *Sp_DissociationEnergy, **MechEqConstNumDensityRange,
               ***MechEqConstCoeffsNumDensity, *Sp_MilikanWhiteConstantA, *Sp_gRotElec,
               **NASACAPCoeffsLowT, **NASACAPCoeffsMidT, **NASACAPCoeffsHighT,
               **NASACAPTemperatureLimits, ***NASACAPCoeffs, *NASACAPSpeciesEnergyAtRefTemp;

        // Mass electron rate of production from HPCI reactions
        double omega_dot_hpci;

        double Hs_mean, Es_mean, S_mean, Cp_mean, Cv_mean;

        /* For initialization of mixture fractions */
        double Sp_TraceInitMoleFrac; // Rename
        int InitMixFrac_Mole_Flag, InitMixFrac_Mass_Flag, InitMixFrac_ERROR, SetInitEoS_ERROR;
        double *InitialMixtureFractions;
        int *TraceSpFlag;
    /*}
    ReactiveVars reactive;*/

protected:
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
    void getSpeciesIndex(int i, std::string sp);
    void readspecies(std::string reactionFilename);
    void InitEOSVariables();

};  

} // Parameters namespace
} // PHiLiP namespace

#endif

