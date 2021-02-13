#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <fstream>

#include "dg/dg_factory.hpp"
#include "parameters/parameters.h"
#include "physics/physics.h"
#include "numerical_flux/convective_numerical_flux.hpp"

using PDEType  = PHiLiP::Parameters::AllParameters::PartialDifferentialEquation;
using ConvType = PHiLiP::Parameters::AllParameters::ConvectiveNumericalFlux;
using DissType = PHiLiP::Parameters::AllParameters::DissipativeNumericalFlux;

const bool   COMPARE_MATRICES = true;//false;
const bool   PRODUCE_TESTS    = !COMPARE_MATRICES;
const double TOLERANCE = 1E-12;

int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    using namespace dealii;
    using namespace PHiLiP;
    const int dim = PHILIP_DIM;
    int error = 0;
    int success_bool = true;

    ParameterHandler parameter_handler;
    Parameters::AllParameters::declare_parameters (parameter_handler);

    Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);
    std::vector<PDEType> pde_type {
        PDEType::advection,
        PDEType::diffusion,
        PDEType::convection_diffusion,
        PDEType::advection_vector
    };

    for (auto pde = pde_type.begin(); pde != pde_type.end() && error == 0; pde++) {
        for (unsigned int poly_degree=1; poly_degree<3; ++poly_degree) {
            for (unsigned int igrid=2; igrid<5; ++igrid) {
                // Generate grids
#if PHILIP_DIM==1
                using Triangulation = dealii::Triangulation<dim>;
#else
                using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif
                std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
#if PHILIP_DIM!=1
                    MPI_COMM_WORLD,
#endif
                    typename dealii::Triangulation<dim>::MeshSmoothing(
                        dealii::Triangulation<dim>::smoothing_on_refinement |
                        dealii::Triangulation<dim>::smoothing_on_coarsening));
                GridGenerator::subdivided_hyper_cube(*grid, igrid);

                // Assemble Jacobian
                all_parameters.pde_type = *pde;
                std::shared_ptr < DGBase<PHILIP_DIM, double> > dg = DGFactory<PHILIP_DIM,double>::create_discontinuous_galerkin(&all_parameters, poly_degree, grid);
                dg->allocate_system ();

                dg->solution *= 0.0;

                dg->assemble_residual(true);

                const int nrows = dg->system_matrix.m();

                // Copy stuff into SparseMatrix since it has the function block_write and block_read
                SparseMatrix<double> sparse_mat;
                SparsityPattern sparsity_pattern;
                sparsity_pattern.copy_from(dg->sparsity_pattern);
                sparse_mat.reinit(sparsity_pattern);
                std::cout << sparse_mat.m() << std::endl;
                std::cout << dg->system_matrix.m() << std::endl;
                sparse_mat.copy_from(dg->system_matrix);

                // Define filename
                std::string pde_string, poly_string = std::to_string(poly_degree), grid_string = std::to_string(igrid);
                if (*pde == PDEType::advection) pde_string = "advection";
                if (*pde == PDEType::diffusion) pde_string = "diffusion";
                if (*pde == PDEType::convection_diffusion) pde_string = "convection_diffusion";
                if (*pde == PDEType::advection_vector) pde_string = "advection_vector";

                std::string filename = std::to_string(dim) + "d_" + pde_string + "_poly_" + poly_string + "_gridsize_" + grid_string + ".mat";
                std::string path = "matrix_data/" + filename;

                if (COMPARE_MATRICES) {
                    // Load up matrix from file
                    std::cout << "Reading matrix from: "<< path << std::endl;
                    std::ifstream infile (path,std::ifstream::binary);
                    SparseMatrix<double> sparse_mat_from_file;
                    sparse_mat_from_file.reinit(sparsity_pattern);
                    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) sparse_mat_from_file.block_read(infile);
                    infile.close();

                    // Compare the matrices and evaluate the relative error in the Frobenius norm
                    double matrix_norm = sparse_mat_from_file.frobenius_norm();
                    double sum = 0.0;
                    auto coeff2 = sparse_mat.begin();
                    for (auto coeff1 = sparse_mat_from_file.begin(); coeff1 < sparse_mat_from_file.end(); ++coeff1) {
                        double diff = coeff1->value() - coeff2->value();
                        sum += std::pow(diff, 2);
                        //std::cout << coeff1->value() << " " << coeff2->value() << " " << diff << sum << std::endl;
                        ++coeff2;
                    }
                    sum = sqrt(sum);
                    double rel_err = std::abs(sum/matrix_norm);

                    success_bool = success_bool && (rel_err < TOLERANCE);

                    std::cout << filename << ", Relative error: " << rel_err << std::endl;
                    if (!success_bool) {
                        error = 1;
                        std::cout << "Previous matrix given by "<< path << "  did not match current one." << std::endl;

                        if (nrows < 15) {
                            FullMatrix<double> fullA(nrows);
                            fullA.copy_from(dg->system_matrix);
                            std::cout<<"CURRENT MATRIX:"<<std::endl;
                            fullA.print_formatted(std::cout, 3, true, 10, "0", 1., 0.);
                            std::cout<<std::endl;
                            std::cout<<std::endl;

                            FullMatrix<double> fullB(nrows);
                            fullB.copy_from(sparse_mat_from_file);
                            std::cout<<"MATRIX FROM FILE:"<<std::endl;
                            fullB.print_formatted(std::cout, 3, true, 10, "0", 1., 0.);
                            std::cout<<std::endl;
                            std::cout<<std::endl;
                        }
                    }
                }


                if (PRODUCE_TESTS) {
                    std::cout << "Creating matrix "<< path << "..." << std::endl;
                    std::ofstream outfile (path,std::ofstream::binary);
                    sparse_mat.block_write(outfile);
                    outfile.close();
                }


            }
        }
    }

    if(PRODUCE_TESTS) return 1;
    return error;
}

