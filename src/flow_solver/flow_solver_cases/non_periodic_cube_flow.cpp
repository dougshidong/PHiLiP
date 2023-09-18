#include "non_periodic_cube_flow.h"
#include "mesh/grids/non_periodic_cube_flow.h"
#include <deal.II/grid/grid_generator.h>
#include "physics/physics_factory.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
NonPeriodicCubeFlow<dim, nstate>::NonPeriodicCubeFlow(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : FlowSolverCaseBase<dim, nstate>(parameters_input)
{
    //create the Physics object
    this->pde_physics = std::dynamic_pointer_cast<Physics::PhysicsBase<dim,nstate,double>>(
                Physics::PhysicsFactory<dim,nstate,double>::create_Physics(parameters_input));
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> NonPeriodicCubeFlow<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
    #if PHILIP_DIM!=1
                this->mpi_communicator
    #endif
        );

    bool use_number_mesh_refinements = false;
    if(this->all_param.flow_solver_param.number_of_mesh_refinements>0)
        use_number_mesh_refinements = true;
    
    const unsigned int number_of_refinements = use_number_mesh_refinements ? this->all_param.flow_solver_param.number_of_mesh_refinements 
                                                                           : log2(this->all_param.flow_solver_param.number_of_grid_elements_per_dimension);

    const double domain_left = this->all_param.flow_solver_param.grid_left_bound;
    const double domain_right = this->all_param.flow_solver_param.grid_right_bound;
    const bool colorize = true;
    
    bool shock_tube = false;
    bool shu_osher = false;
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case_type = this->all_param.flow_solver_param.flow_case_type;

    if (flow_case_type == flow_case_enum::sod_shock_tube
        || flow_case_type == flow_case_enum::leblanc_shock_tube) {
        shock_tube = true;
    } else if (flow_case_type == flow_case_enum::shu_osher_problem) {
        shu_osher = true;
    }


    Grids::non_periodic_cube_flow<dim>(*grid, domain_left, domain_right, colorize, shock_tube, shu_osher);
    grid->refine_global(number_of_refinements);

    return grid;
}

template <int dim, int nstate>
void NonPeriodicCubeFlow<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
}

template <int dim, int nstate>
double NonPeriodicCubeFlow<dim,nstate>::get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> dg) const
{
    // compute time step based on advection speed (i.e. maximum local wave speed)
    const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
    const double approximate_grid_spacing = (this->all_param.flow_solver_param.grid_right_bound-this->all_param.flow_solver_param.grid_left_bound)/pow(number_of_degrees_of_freedom_per_state,(1.0/dim));
    const double cfl_number = this->all_param.flow_solver_param.courant_friedrichs_lewy_number;
    const double time_step = cfl_number * approximate_grid_spacing / this->maximum_local_wave_speed;
    
    return time_step;
}

template <int dim, int nstate>
double NonPeriodicCubeFlow<dim,nstate>::get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim,double>> dg)
{
    // initialize the maximum local wave speed
    update_maximum_local_wave_speed(*dg);
    // compute time step based on advection speed (i.e. maximum local wave speed)
    const double time_step = get_adaptive_time_step(dg);
    return time_step;
}

template<int dim, int nstate>
void NonPeriodicCubeFlow<dim, nstate>::update_maximum_local_wave_speed(DGBase<dim, double> &dg)
{    
    // Initialize the maximum local wave speed to zero
    this->maximum_local_wave_speed = 0.0;

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(dg.max_degree+1+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg.high_order_grid->mapping_fe_field), dg.fe_collection[dg.max_degree], quad_extra,
                                              dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);

    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    for (auto cell = dg.dof_handler.begin_active(); cell!=dg.dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }

            // Update the maximum local wave speed (i.e. convective eigenvalue)
            const double local_wave_speed = this->pde_physics->max_convective_eigenvalue(soln_at_q);
            if(local_wave_speed > this->maximum_local_wave_speed) this->maximum_local_wave_speed = local_wave_speed;
        }
    }
    this->maximum_local_wave_speed = dealii::Utilities::MPI::max(this->maximum_local_wave_speed, this->mpi_communicator);
}

#if PHILIP_DIM==2
    template class NonPeriodicCubeFlow<PHILIP_DIM, 1>;
#else
    template class NonPeriodicCubeFlow <PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // FlowSolver namespace
} // PHiLiP namespace