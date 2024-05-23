#ifndef __POSITIVITY_TESTS_GRID_H__
#define __POSITIVITY_TESTS_GRID_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>
#include "parameters/all_parameters.h"

namespace PHiLiP::Grids {

/// Create mesh for each specific case 
/// Boundary IDs are assigned for limiter_convergence_tests
/// Unassigned otherwise
template<int dim, typename TriangulationType>
void shock_tube_1D_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input);

template<int dim, typename TriangulationType>
void explosion_problem_grid(
    TriangulationType& grid,
    const Parameters::AllParameters* const parameters_input);

template<int dim, typename TriangulationType>
void double_mach_reflection_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input);

template<int dim, typename TriangulationType>
void sedov_blast_wave_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input);

template<int dim, typename TriangulationType>
void mach_3_wind_tunnel_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input);

template<int dim, typename TriangulationType>
void shock_diffraction_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input);

template<int dim, typename TriangulationType>
void astrophysical_jet_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input);
} 

// namespace PHiLiP::Grids
#endif