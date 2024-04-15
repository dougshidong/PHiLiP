#include "airfoil_2D.h"
#include <deal.II/base/function.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include <deal.II/base/table_handler.h>
#include "mesh/grids/naca_airfoil_grid.hpp"
#include "mesh/gmsh_reader.hpp"
#include "functional/lift_drag.hpp"
#include "functional/extraction_functional.hpp"
#include "functional/amiet_model.hpp"

namespace PHiLiP {
namespace FlowSolver {
//=========================================================
// Airfoil 2D
//=========================================================
template <int dim, int nstate>
Airfoil2D<dim, nstate>::Airfoil2D(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , airfoil_length(this->all_param.flow_solver_param.airfoil_length)
        , height(this->all_param.flow_solver_param.height)
        , length_b2(this->all_param.flow_solver_param.length_b2)
        , incline_factor(this->all_param.flow_solver_param.incline_factor)
        , bias_factor(this->all_param.flow_solver_param.bias_factor)
        , refinements(this->all_param.flow_solver_param.refinements)
        , n_subdivision_x_0(this->all_param.flow_solver_param.n_subdivision_x_0)
        , n_subdivision_x_1(this->all_param.flow_solver_param.n_subdivision_x_1)
        , n_subdivision_x_2(this->all_param.flow_solver_param.n_subdivision_x_2)
        , n_subdivision_y(this->all_param.flow_solver_param.n_subdivision_y)
        , airfoil_sampling_factor(this->all_param.flow_solver_param.airfoil_sampling_factor)
{}

template <int dim, int nstate>
void Airfoil2D<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    using Model_enum = Parameters::AllParameters::ModelType;
    const PDE_enum pde_type = this->all_param.pde_type;
    const Model_enum model_type = this->all_param.model_type;
    if (pde_type == PDE_enum::navier_stokes || (pde_type == PDE_enum::physics_model && model_type == Model_enum::reynolds_averaged_navier_stokes)){
        this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
    }
    this->pcout << "- - Courant-Friedrich-Lewy number: " << this->all_param.flow_solver_param.courant_friedrich_lewy_number << std::endl;
    this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    const double pi = atan(1.0) * 4.0;
    this->pcout << "- - Angle of attack [deg]: " << this->all_param.euler_param.angle_of_attack*180/pi << std::endl;
    this->pcout << "- - Side-slip angle [deg]: " << this->all_param.euler_param.side_slip_angle*180/pi << std::endl;
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> Airfoil2D<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
    this->mpi_communicator
#endif
    );

#if PHILIP_DIM==2
    dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
    airfoil_data.airfoil_type = "NACA";
    airfoil_data.naca_id      = "0012";
    airfoil_data.airfoil_length = airfoil_length;
    airfoil_data.height         = height;
    airfoil_data.length_b2      = length_b2;
    airfoil_data.incline_factor = incline_factor;
    airfoil_data.bias_factor    = bias_factor; 
    airfoil_data.refinements    = refinements;

    airfoil_data.n_subdivision_x_0 = n_subdivision_x_0;
    airfoil_data.n_subdivision_x_1 = n_subdivision_x_1;
    airfoil_data.n_subdivision_x_2 = n_subdivision_x_2;
    airfoil_data.n_subdivision_y = n_subdivision_y;
    airfoil_data.airfoil_sampling_factor = airfoil_sampling_factor; 

    dealii::GridGenerator::Airfoil::create_triangulation(*grid, airfoil_data);

//    // Assign a manifold to have curved geometry
//    unsigned int manifold_id = 0;
//    grid->reset_all_manifolds();
//    grid->set_all_manifold_ids(manifold_id);
//    // Set Flat manifold on the domain, but not on the boundary.
//    grid->set_manifold(manifold_id, dealii::FlatManifold<2>());
//
//    manifold_id = 1;
//    bool is_upper = true;
//    const Grids::NACAManifold<2,1> upper_naca(airfoil_data.naca_id, is_upper);
//    grid->set_all_manifold_ids_on_boundary(2,manifold_id); // upper airfoil side
//    grid->set_manifold(manifold_id, upper_naca);
//
//    is_upper = false;
//    const Grids::NACAManifold<2,1> lower_naca(airfoil_data.naca_id, is_upper);
//    manifold_id = 2;
//    grid->set_all_manifold_ids_on_boundary(3,manifold_id); // lower airfoil side
//    grid->set_manifold(manifold_id, lower_naca); 

    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0 || current_id == 1 || current_id == 4 || current_id == 5) {
                    cell->face(face)->set_boundary_id (1005); // farfield
                } else {
                    cell->face(face)->set_boundary_id (1001); // wall
                }
            }
        }
    }
#endif
    return grid;
}

template <int dim, int nstate>
void Airfoil2D<dim,nstate>::steady_state_postprocessing(std::shared_ptr<DGBase<dim, double>> dg) const
{
    if constexpr(nstate!=1){
        dealii::Point<dim,double> extraction_point;
        if constexpr(dim==2){
            extraction_point[0] = this->all_param.boundary_layer_extraction_param.extraction_point_x;
            extraction_point[1] = this->all_param.boundary_layer_extraction_param.extraction_point_y;
        } else if constexpr(dim==3){
            extraction_point[0] = this->all_param.boundary_layer_extraction_param.extraction_point_x;
            extraction_point[1] = this->all_param.boundary_layer_extraction_param.extraction_point_y;
            extraction_point[2] = this->all_param.boundary_layer_extraction_param.extraction_point_z;
        }
        int number_of_sampling = this->all_param.boundary_layer_extraction_param.number_of_sampling;
    
        ExtractionFunctional<dim,nstate,double,Triangulation> boundary_layer_extraction(dg, extraction_point, number_of_sampling);

        dealii::Point<3,double> observer_coord_ref;
        observer_coord_ref[0] = this->all_param.amiet_param.observer_coord_ref_x;
        observer_coord_ref[1] = this->all_param.amiet_param.observer_coord_ref_y;
        observer_coord_ref[2] = this->all_param.amiet_param.observer_coord_ref_z;

        AmietModelFunctional<dim,nstate,double,Triangulation> amiet_acoustic_response(dg,boundary_layer_extraction,observer_coord_ref);

        real OASPL_airfoil_2D;
        OASPL_airfoil_2D = amiet_acoustic_response.evaluate_functional(true,false,false);
    }
}

#if PHILIP_DIM==2
    template class Airfoil2D <PHILIP_DIM,1>;
    template class Airfoil2D <PHILIP_DIM,PHILIP_DIM+2>;
    template class Airfoil2D <PHILIP_DIM,PHILIP_DIM+3>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace