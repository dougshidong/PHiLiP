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
    const Parameters::FlowSolverParam *const flow_solver_param);

template<int dim, typename TriangulationType>
void double_mach_reflection_grid(
    TriangulationType&  grid,
    const Parameters::FlowSolverParam *const flow_solver_param);

template<int dim, typename TriangulationType>
void shock_diffraction_grid(
    TriangulationType&  grid,
    const Parameters::FlowSolverParam *const flow_solver_param);

template<int dim, typename TriangulationType>
void astrophysical_jet_grid(
    TriangulationType&  grid,
    const Parameters::FlowSolverParam *const flow_solver_param);

template<int dim, typename TriangulationType>
void svsw_grid(
    TriangulationType&  grid,
    const Parameters::FlowSolverParam *const flow_solver_param);
} 

// namespace PHiLiP::Grids
#endif