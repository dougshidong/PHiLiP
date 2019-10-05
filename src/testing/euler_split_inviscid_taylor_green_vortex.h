#ifndef __EULER_SPLIT_TAYLOR_GREEN_H__
#define __EULER_SPLIT_TAYLOR_GREEN_H__


#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/fe/mapping_q.h>
#include "tests.h"


#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "numerical_flux/numerical_flux.h"
#include "physics/physics_factory.h"
#include "physics/physics.h"
#include "dg/dg.h"
#include "ode_solver/ode_solver.h"

#include<fenv.h>

//using PDEType  = PHiLiP::Parameters::AllParameters::PartialDifferentialEquation;
//using ConvType = PHiLiP::Parameters::AllParameters::ConvectiveNumericalFlux;
//using DissType = PHiLiP::Parameters::AllParameters::DissipativeNumericalFlux;
//
//
//const double TOLERANCE = 1E-12;
namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
class EulerTaylorGreen : public TestsBase
{
public:
	EulerTaylorGreen() = delete;
	EulerTaylorGreen(const Parameters::AllParameters *const parameters_input);
	int run_test() const override;

private:
	double compute_kinetic_energy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const;
	//double compute_quadrature_kinetic_energy(std::array<double,nstate> soln_at_q) const ;
    //const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
    const MPI_Comm mpi_communicator;
    dealii::ConditionalOStream pcout;
};


} //Tests
} //PHiLiP

#endif
