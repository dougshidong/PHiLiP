#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include "all_real_gas_constants.h"
#include "readspecies.h"
#include "ReactiveVar.h"
#include "eos.h"
#include "var.h"
#include "NonEquilibriumVar.h"
#include "allocatePointersLib.h"

namespace PHiLiP {
namespace RealGasConstants {

AllRealGasConstants::AllRealGasConstants ()
    : //manufactured_convergence_study_param(ManufacturedConvergenceStudyParam())
    // , reactive_var(EulerParam())
    // , nonequilibrium_var(NavierStokesParam())
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{ 
    //-------------------------------------------
    //          READ SPECIES FILE
    //-------------------------------------------
    readspecies(namechem);
    
    /* Number of species in conserved quantities vector */
    N_species_Qvec = N_species-1;

    /* Number of equations for 0D */
    nEQ = N_species_Qvec + nTEMP;
    
    /* Allocate NonEquilibrium Global Variables + omega_dot + Qvec */
    omega_dot = dvector(N_species);
    omega_dot_mech = dmatrix(N_species,N_mechs);
    VibEnergy_MacheretFridman_mech = dvector(N_mechs);
    Sp_VibEnergy = dvector(N_species);
    Sp_TransRotEnergy = dvector(N_species);
    Sp_ElectronicEnergy = dvector(N_species);
    Sp_TransRotCv = dvector(N_species);
    Sp_TotIntEnergy = dvector(N_species);
    Sp_EnergyFormation = dvector(N_species);
    NASACAPSpeciesEnergyAtRefTemp = dvector(N_species);
    Sp_TempIndex = ivector(N_species);
    Qvec = dvector(nEQ);
    RT = dvector(nTEMP);
    MoleFractions = dvector(N_species);
    MolarityVec = dvector(N_species);
    MassFractions = dvector(N_species);
    Sp_Density = dvector(N_species);
    Cp = dvector(N_species);
    H = dvector(N_species);
    S = dvector(N_species);
    Sp_Index_Qvec = ivector(N_species_Qvec);
    InitialMixtureFractions = dvector(N_species);
    TraceSpFlag = ivector(N_species);
    Rxn_MacheretFridman_MassFraction = dvector(N_mechs);
    /* Reference Energies */
    Sp_VibEnergy_Ref = dvector(N_species);
    Sp_TransRotEnergy_Ref = dvector(N_species);
    Sp_ElectronicEnergy_Ref = dvector(N_species);
    /* Combustion */
    Cp_molar = dvector(N_species);
    Cv_molar = dvector(N_species);
    H_formation = dvector(N_species);
    
    /* Initialize Qvec as zeros */
    for(int i=0; i<nEQ; i++)
    {
        Qvec[i] = 0.0;
    }

    /* Index for Qvec */
    if(NonEqFlag == 1)
    {
        i_VibEnergy = nEQ - nTEMP;
        if(nTEMP == 3)
        {
            i_ElecEnergy = nEQ - 2;
        }
    }
    i_TotEnergy = nEQ - 1;
    
    /* One time thermodynamic property calculations */
    if(NonEqFlag==1)
    {
        /* Get species translational-rotational isochoric specific heat: Sp_TransRotCv */
        GetSpeciesTransRotCv(Sp_TransRotCv);
        /* Get species energy of formation at T_ref */
        GetSpeciesEnergyFormation(Sp_EnergyFormation);
    }
}

} // RealGasConstants namespace
} // PHiLiP namespace
