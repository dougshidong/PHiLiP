#include "optimization/design_parameterization/identity_parameterization.hpp"
#include <deal.II/grid/grid_generator.h>
#include "dg/dg_factory.hpp"


int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);    
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    const int dim = PHILIP_DIM;
    AssertDimension(dim, 2);

    using namespace PHiLiP;   
    
    using VectorType = typename dealii::LinearAlgebra::distributed::Vector<double>;
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix;
    using MeshType = typename dealii::parallel::distributed::Triangulation<dim>; // dim is 2 for this test.
    
    // Create grid and dg. 
    std::shared_ptr<MeshType> grid = std::make_shared<MeshType>(MPI_COMM_WORLD);
    unsigned int grid_refinement_val = 4;
    dealii::GridGenerator::hyper_cube(*grid);
    grid->refine_global(grid_refinement_val);

    dealii::ParameterHandler parameter_handler; // Using default parameters. 
    Parameters::AllParameters::declare_parameters (parameter_handler);
    Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);
    const unsigned int poly_degree = 2;
    const unsigned int grid_degree = 1;

    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim, double>::create_discontinuous_galerkin(&all_parameters, poly_degree,poly_degree, grid_degree, grid);
    dg->allocate_system();


    VectorType initial_volume_nodes = dg->high_order_grid->volume_nodes; 
    const dealii::IndexSet &volume_range = dg->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    const unsigned int n_global_vol_nodes = dg->high_order_grid->volume_nodes.size();


    std::unique_ptr<BaseParameterization<dim>> design_parameterization = 
                        std::make_unique<IdentityParameterization<dim>>(dg->high_order_grid);

    VectorType design_var;
    design_parameterization->initialize_design_variables(design_var); // expected to get volume nodes.
    const dealii::IndexSet &design_var_range = design_var.get_partitioner()->locally_owned_range();
    for(unsigned int i=0; i<n_global_vol_nodes; ++i)
    {
        if(volume_range.is_element(i))
        {
            if(! design_var_range.is_element(i)) 
            {
                pcout<<"Design variable's parallel distribution is not the same as volume node's distribution"<<std::endl;
                return 1;
            }

            if(dg->high_order_grid->volume_nodes(i) != design_var(i))
            {
                pcout<<"Initialization isn't done well."<<std::endl;
                return 1;
            }

        }
    }
    pcout<<"Initialized design variables."<<std::endl;
    MatrixType dXv_dXp;
    design_parameterization->compute_dXv_dXp(dXv_dXp);
    pcout<<"Computed dXv_dXp."<<std::endl;

    VectorType design_var_updated = design_var;
    const double change_in_val = 1.0e-3;
    design_var_updated.add(change_in_val);


    const bool is_mesh_updated = design_parameterization->update_mesh_from_design_variables(dXv_dXp, design_var_updated); // Expected to update all volume nodes by change_in_val.
    if(! is_mesh_updated) {return 1;}
    pcout<<"Updated mesh."<<std::endl;

    VectorType diff = dg->high_order_grid->volume_nodes;
    diff -= initial_volume_nodes;

    pcout<<"Now checking if mesh has been updated as expected..."<<std::endl;
    for(unsigned int i=0; i<n_global_vol_nodes; ++i)
    {
        if(!volume_range.is_element(i)) continue;
        
        if((diff(i) - change_in_val) > 1.0e-16) {
            pcout<<"Volume node hasn't changed as expected. Should have changed by change_in_val."<<std::endl;
            return 1;
        }
    }
   
    pcout<<"Mesh has been updated as expected."<<std::endl;
    return 0; // Test passed
}
