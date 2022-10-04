#include "optimization/design_parameterization/ffd_parameterization.hpp"
#include <deal.II/grid/grid_generator.h>
#include "dg/dg_factory.hpp"
#include "mesh/free_form_deformation.h"
#include "mesh/grids/gaussian_bump.h"

const int dim = 2;
const int nstate = 4;
const int POLY_DEGREE = 2;
const int MESH_DEGREE = POLY_DEGREE+1;
const double BUMP_HEIGHT = 0.0625;
const double CHANNEL_LENGTH = 3.0;
const double CHANNEL_HEIGHT = 0.8;
const unsigned int NY_CELL = 3;
const unsigned int NX_CELL = 5*NY_CELL;
const unsigned int nx_ffd = 10;

int main (int argc, char * argv[])
{
    using namespace PHiLiP;
    using VectorType = typename dealii::LinearAlgebra::distributed::Vector<double>;
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix;
    using MeshType = typename dealii::parallel::distributed::Triangulation<dim>; // dim is 2 for this test.
    
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);    
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);
    
    dealii::ParameterHandler parameter_handler; // Using default parameters. 
    Parameters::AllParameters::declare_parameters (parameter_handler);
    Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);

    const int dim = PHILIP_DIM;
    AssertDimension(dim, 2);

    // Create grid
    std::vector<unsigned int> n_subdivisions(dim);

    n_subdivisions[1] = NY_CELL;
    n_subdivisions[0] = NX_CELL;
    using MeshType = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<MeshType> grid = std::make_shared<MeshType>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
    Grids::gaussian_bump(*grid, n_subdivisions, CHANNEL_LENGTH, CHANNEL_HEIGHT, 0.5*BUMP_HEIGHT);
    

    // Create DG object
    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters, POLY_DEGREE, POLY_DEGREE, MESH_DEGREE, grid);
    dg->allocate_system ();
    VectorType initial_vol_nodes = dg->high_order_grid->volume_nodes;
    
    // Create FFD and get ffd_design_variables_indices_dim.
    const dealii::Point<dim> ffd_origin(-1.4,-0.1);
    const std::array<double,dim> ffd_rectangle_lengths = {{2.8,0.6}};
    const std::array<unsigned int,dim> ffd_ndim_control_pts = {{nx_ffd,2}};
    FreeFormDeformation<dim> ffd( ffd_origin, ffd_rectangle_lengths, ffd_ndim_control_pts);

    unsigned int n_design_variables = 0;
    // Vector of ijk indices and dimension.
    // Each entry in the vector points to a design variable's ijk ctl point and its acting dimension.
    std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;
    for (unsigned int i_ctl = 0; i_ctl < ffd.n_control_pts; ++i_ctl) {

        const std::array<unsigned int,dim> ijk = ffd.global_to_grid ( i_ctl );
        for (unsigned int d_ffd = 0; d_ffd < dim; ++d_ffd) {

            if (   ijk[0] == 0 // Constrain first column of FFD points.
                || ijk[0] == ffd_ndim_control_pts[0] - 1  // Constrain last column of FFD points.
                || ijk[1] == 0 // Constrain first row of FFD points.
                || d_ffd == 0 // Constrain x-direction of FFD points.
               ) {
                continue;
            }
            ++n_design_variables;
            ffd_design_variables_indices_dim.push_back(std::make_pair(i_ctl, d_ffd));
        }
    }

    // Create design parameterization
    std::unique_ptr<BaseParameterization<dim>> design_parameterization = 
                        std::make_unique<FreeFormDeformationParameterization<dim>>(dg->high_order_grid, ffd, ffd_design_variables_indices_dim);


    VectorType initial_design_var;
    design_parameterization->initialize_design_variables(initial_design_var); //Initializes design variables with control points
    pcout<<"Initialized design variables."<<std::endl;
    
    MatrixType dXv_dXp;
    design_parameterization->compute_dXv_dXp(dXv_dXp);
    pcout<<"Computed dXv_dXp."<<std::endl;

    VectorType design_var_updated = initial_design_var;
    const double change_in_val = 1.0e-3;
    design_var_updated.add(change_in_val);
    design_var_updated.update_ghost_values();

    // Update mesh using design parameterization class.
    const bool is_mesh_updated = design_parameterization->update_mesh_from_design_variables(dXv_dXp, design_var_updated); // Expected to update the mesh according to ffd parameterization.
    if(! is_mesh_updated) {return 1;}
    pcout<<"Updated mesh."<<std::endl;

    VectorType volume_nodes_from_design_parameterization = dg->high_order_grid->volume_nodes; 

    // Now reset everything to initial state.
    dg->high_order_grid->volume_nodes = initial_vol_nodes;
    ffd.set_design_variables(ffd_design_variables_indices_dim, initial_design_var);

    // Update mesh with meshmover linear elasticity.
    ffd.set_design_variables(ffd_design_variables_indices_dim, design_var_updated); // updates control points
    ffd.deform_mesh(*(dg->high_order_grid)); // Deforms mesh using updated control points.
    VectorType volume_nodes_from_meshmover =  dg->high_order_grid->volume_nodes;

    // Check if the two resulting meshes are equal.
    VectorType vol_nodes_diff = volume_nodes_from_design_parameterization;
    vol_nodes_diff -= volume_nodes_from_meshmover;

    if(vol_nodes_diff.l2_norm() > 1.0e-14)
    {
        pcout<<"Volume nodes from design parameterization and meshmover linearelasticity are different."<<std::endl;
        return 1;
    }

    pcout<<"Design parameterizaion and meshmover linearelasticity gave the same change in volume nodes."<<std::endl;
    return 0;
}
