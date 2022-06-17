#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>
#include <math.h>
#include <iostream>
#include <stdlib.h>

#include <ctime>

#include <deal.II/distributed/solution_transfer.h>

#include "testing/tests.h"

#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/meshworker/dof_info.h>

// Finally, we take our exact solution from the library as well as volume_quadrature
// and additional tools.
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_fe_field.h> 

#include <deal.II/fe/fe_bernstein.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "operators/operators.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"

const double TOLERANCE = 1E-6;
using namespace std;

int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    using real = double;
    using namespace PHiLiP;
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = 1;
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);
    all_parameters_new.nstate = nstate;
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    using FR_enum = Parameters::AllParameters::Flux_Reconstruction;
    all_parameters_new.flux_reconstruction_type = FR_enum::cHU;
    // all_parameters_new.use_collocated_nodes=true;
    all_parameters_new.overintegration = 0;

    for(unsigned int poly_degree=2; poly_degree<3; poly_degree++){

        // OPERATOR::OperatorsBase<dim,real> operators(&all_parameters_new, nstate, poly_degree, poly_degree, poly_degree); 
        OPERATOR::OperatorsBaseState<dim,real,nstate,2*dim> operators(&all_parameters_new, poly_degree, poly_degree);

        const unsigned int n_dofs = operators.fe_collection_basis[poly_degree].dofs_per_cell;
        // const unsigned int n_dofs_flux = operators.fe_collection_flux_basis[poly_degree].dofs_per_cell;
        const unsigned int n_quad_pts = operators.volume_quadrature_collection[poly_degree].size();
        std::vector<real> u(n_quad_pts);
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            u[iquad] = exp(operators.volume_quadrature_collection[poly_degree].point(iquad)[0]);
        }
        dealii::Vector<real> four_pt_flux(n_quad_pts);
        dealii::Vector<real> deriv_F(n_quad_pts);
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            four_pt_flux[iquad] = 0.0;
            deriv_F[iquad] = 0.0;
            for(unsigned int flux_basis=0; flux_basis<n_quad_pts; flux_basis++){
                const dealii::Point<dim> qpoint_L  = operators.volume_quadrature_collection[poly_degree].point(iquad);
                double weight_L = (1.0/std::sqrt(qpoint_L[0]*(1.0-qpoint_L[0]))); 
                const dealii::Point<dim> qpoint_R  = operators.volume_quadrature_collection[poly_degree].point(flux_basis);
                double weight_R = (1.0/std::sqrt(qpoint_R[0]*(1.0-qpoint_R[0]))); 
                four_pt_flux[iquad] += 2.0 * (1.0/6.0 *(u[iquad]*u[iquad] + u[iquad]*u[flux_basis] + u[flux_basis]*u[flux_basis]) )
                                                        *  (weight_L + weight_R)
                                                        *  operators.gradient_flux_basis[poly_degree][0][0][iquad][flux_basis];

                deriv_F[iquad] += 1.0/3.0* pow(u[flux_basis],3)
                //                * (1.0/std::sqrt(qpoint_L[0]*(1.0-qpoint_L[0]))) 
                                *  operators.gradient_flux_basis[poly_degree][0][0][iquad][flux_basis];
            }
        }
        dealii::Vector<real> vol_int(n_dofs);
        pcout<<" volume integral "<<std::endl;
        for(unsigned int idof=0; idof<n_dofs; idof++){
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                const dealii::Point<dim> qpoint = operators.volume_quadrature_collection[poly_degree].point(iquad);
                vol_int[idof] += operators.vol_integral_basis[poly_degree][iquad][idof]
                                * four_pt_flux[iquad]
                                / (1.0/std::sqrt(qpoint[0]*(1.0-qpoint[0]))); 
            }
            pcout<<vol_int[idof]<<std::endl;
        }
        double energy =0.0;
        double energy2 =0.0;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            const dealii::Point<dim> qpoint = operators.volume_quadrature_collection[poly_degree].point(iquad);
            energy += operators.volume_quadrature_collection[poly_degree].weight(iquad)
                            * u[iquad]
                            / (1.0/std::sqrt(qpoint[0]*(1.0-qpoint[0]))) 
                            * four_pt_flux[iquad];
            energy2 +=operators.volume_quadrature_collection[poly_degree].weight(iquad)
             //           / (1.0/std::sqrt(qpoint[0]*(1.0-qpoint[0]))) 
                            * deriv_F[iquad];
           // energy2 +=operators.volume_quadrature_collection[poly_degree].weight(iquad)
           //             * 1.0/3.0 *pow(u[iquad],3)
           //             * ((2.0*qpoint[0]-1.0)/(pow(qpoint[0]*(1.0-qpoint[0]), 3.0/2.0)*2.0))
           //             / (1.0/std::sqrt(qpoint[0]*(1.0-qpoint[0]))); 
        }
        pcout<<" energy "<<energy<<" energy twooo "<<energy2<<std::endl;
    }//end of poly_degree loop

    return 0;
}//end of main
