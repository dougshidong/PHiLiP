#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "positivity_preserving_tests_grid.h"

namespace PHiLiP::Grids {
template<int dim, typename TriangulationType>
void shock_tube_1D_grid(
    TriangulationType&  grid,
    const Parameters::FlowSolverParam *const flow_solver_param)
{
    const double xmax = flow_solver_param->grid_xmax;
    const double xmin = flow_solver_param->grid_xmin;
    const unsigned int n_subdivisions_x = flow_solver_param->number_of_grid_elements_x;

    dealii::GridGenerator::subdivided_hyper_cube(grid, n_subdivisions_x, xmin, xmax, true);

    int left_boundary_id = 9999;
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case_type = flow_solver_param->flow_case_type;

    if (flow_case_type == flow_case_enum::sod_shock_tube
        || flow_case_type == flow_case_enum::leblanc_shock_tube) {
        left_boundary_id = 1001; // x_left, wall bc
    } else if (flow_case_type == flow_case_enum::shu_osher_problem) {
        left_boundary_id = 1008; // x_left, custom inflow (set in prm file)
    } 

    if (left_boundary_id != 9999 && dim == 1) {
        for (auto cell = grid.begin_active(); cell != grid.end(); ++cell) {
            // Set a dummy material ID
            cell->set_material_id(9002);
            if (cell->face(0)->at_boundary()) cell->face(0)->set_boundary_id(left_boundary_id);
            if (cell->face(1)->at_boundary()) cell->face(1)->set_boundary_id(1001);
        }
    }
}

template<int dim, typename TriangulationType>
void double_mach_reflection_grid(
    TriangulationType&  grid,
    const Parameters::FlowSolverParam *const flow_solver_param) 
{
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = flow_solver_param->grid_xmin; p1[1] = flow_solver_param->grid_ymin;
    p2[0] = flow_solver_param->grid_xmax; p2[1] = flow_solver_param->grid_ymax;
    
    std::vector<unsigned int> n_subdivisions(2);

    n_subdivisions[0] = flow_solver_param->number_of_grid_elements_x;
    n_subdivisions[1] = flow_solver_param->number_of_grid_elements_y;

    dealii::GridGenerator::subdivided_hyper_rectangle(grid, n_subdivisions, p1, p2, true);

    double bottom_x = 0.0;

    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0) {
                    cell->face(face)->set_boundary_id(1008); // x_left, Post Shock (custom bc set in prm file)
                }
                else if (current_id == 1) {
                    cell->face(face)->set_boundary_id(1007); // x_right, Do Nothing Outflow 
                }
                else if (current_id == 2) {
                    if (bottom_x < (1.0 / 6.0)) {
                        bottom_x += cell->extent_in_direction(0);
                        cell->face(face)->set_boundary_id(1008); // y_bottom, Post Shock (custom bc set in prm file)
                    }
                    else {
                        cell->face(face)->set_boundary_id(1001); // y_bottom, Symmetry/Wall
                    }
                }
                else if (current_id == 3) {
                    // currently set to 
                    cell->face(face)->set_boundary_id(1001); // y_top, Symmetry/Wall
                }
            }
        }
    } 
}


template<int dim, typename TriangulationType>
void shock_diffraction_grid(
    TriangulationType&  grid,
    const Parameters::FlowSolverParam *const flow_solver_param) 
{
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = flow_solver_param->grid_xmin; p1[1] = flow_solver_param->grid_ymin;
    p2[0] = flow_solver_param->grid_xmax; p2[1] = flow_solver_param->grid_ymax;
    
    std::vector<unsigned int> n_subdivisions(2);

    n_subdivisions[0] = flow_solver_param->number_of_grid_elements_x;
    n_subdivisions[1] = flow_solver_param->number_of_grid_elements_y;

    std::vector<int> n_cells_remove(2);
    n_cells_remove[0] = (1.0/13.0)*n_subdivisions[0];
    n_cells_remove[1] = (6.0/11.0)*n_subdivisions[1];

    dealii::GridGenerator::subdivided_hyper_L(grid, n_subdivisions, p1, p2, n_cells_remove);


    // Set boundary type and design type
    double left_y = 0.0;
    double bottom_x = 0.0;
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                if (face == 0) {
                    if (left_y < 6.0) {
                        left_y += cell->extent_in_direction(1);
                        cell->face(face)->set_boundary_id(1001); // x_left, Symmetry/Wall
                    }
                    else {
                        cell->face(face)->set_boundary_id(1008); // x_left, Post Shock (custom bc set in prm file)
                    }
                }
                else if (face == 1) {
                    cell->face(face)->set_boundary_id(1007); // x_right, Do Nothing Outflow 
                }
                else if (face == 2) {
                    if (left_y >= 6.0 && bottom_x < 1.0) {
                        bottom_x += cell->extent_in_direction(1);
                        cell->face(face)->set_boundary_id(1001); // y_bottom, Symmetry/Wall
                    }
                    else {
                        cell->face(face)->set_boundary_id(1009); // y_bottom, Do Nothing Outflow 
                    }
                }
                else if (face == 3) {
                        cell->face(face)->set_boundary_id(1001); // y_bottom, Symmetry/Wall
                }
            }
        }
    }

}

template<int dim, typename TriangulationType>
void astrophysical_jet_grid(
    TriangulationType&  grid,
    const Parameters::FlowSolverParam *const flow_solver_param) 
{
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = flow_solver_param->grid_xmin; p1[1] = flow_solver_param->grid_ymin;
    p2[0] = flow_solver_param->grid_xmax; p2[1] = flow_solver_param->grid_ymax;
    
    std::vector<unsigned int> n_subdivisions(2);

    n_subdivisions[0] = flow_solver_param->number_of_grid_elements_x;
    n_subdivisions[1] = flow_solver_param->number_of_grid_elements_y;

    std::vector<int> n_cells_remove(2);
    n_cells_remove[0] = (1.0/13.0)*n_subdivisions[0];
    n_cells_remove[1] = (6.0/11.0)*n_subdivisions[1];

    dealii::GridGenerator::subdivided_hyper_rectangle(grid, n_subdivisions, p1, p2, true);

    double left_y = 0.0;
    const double dy = (p2[1]-p2[0])/n_subdivisions[1];

    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0) {
                    if (left_y >= 0.45 && left_y <=0.55-dy) {
                        left_y += cell->extent_in_direction(1);
                        cell->face(face)->set_boundary_id(1008); // x_left, Post Shock (custom bc set in prm file)
                    }
                    else {
                        left_y += cell->extent_in_direction(1);
                        cell->face(face)->set_boundary_id(1009); // x_left, Astrophysical Jet Inflow 
                    }
                }
                else if (current_id == 1) {
                    cell->face(face)->set_boundary_id(1007); // x_right, Do Nothing Outflow 
                }
                else if (current_id == 2) {
                    cell->face(face)->set_boundary_id(1007); // y_bottom, Do Nothing Outflow 
                }
                else if (current_id == 3) {
                    cell->face(face)->set_boundary_id(1007); // y_top,  Do Nothing Outflow 
                }
            }
        }
    }
}

template<int dim, typename TriangulationType>
void svsw_grid(
    TriangulationType&  grid,
    const Parameters::FlowSolverParam *const flow_solver_param) 
{
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = flow_solver_param->grid_xmin; p1[1] = flow_solver_param->grid_ymin;
    p2[0] = flow_solver_param->grid_xmax; p2[1] = flow_solver_param->grid_ymax;
    
    std::vector<unsigned int> n_subdivisions(2);

    n_subdivisions[0] = flow_solver_param->number_of_grid_elements_x;
    n_subdivisions[1] = flow_solver_param->number_of_grid_elements_y;

    std::vector<int> n_cells_remove(2);
    n_cells_remove[0] = (1.0/13.0)*n_subdivisions[0];
    n_cells_remove[1] = (6.0/11.0)*n_subdivisions[1];


    dealii::GridGenerator::subdivided_hyper_rectangle(grid, n_subdivisions, p1, p2, true);

    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0) {
                    cell->face(face)->set_boundary_id(1008); // x_left, Post Shock (custom bc set in prm file)
                }
                else if (current_id == 1) {
                    cell->face(face)->set_boundary_id(1007); // x_right,  Do Nothing Outflow 
                }
                else if (current_id == 2) {
                    cell->face(face)->set_boundary_id(1001); // y_top, Symmetry/Wall
                }
                else if (current_id == 3) {
                    cell->face(face)->set_boundary_id(1001); // y_bottom, Symmetry/Wall
                }
            }
        }
    }
}

#if PHILIP_DIM==1
template void shock_tube_1D_grid<1, dealii::Triangulation<1>>(
    dealii::Triangulation<1>&   grid,
    const Parameters::FlowSolverParam *const flow_solver_param);
#else
template void double_mach_reflection_grid<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    const Parameters::FlowSolverParam *const flow_solver_param);
template void shock_diffraction_grid<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    const Parameters::FlowSolverParam *const flow_solver_param);
template void astrophysical_jet_grid<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    const Parameters::FlowSolverParam *const flow_solver_param);
template void svsw_grid<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    const Parameters::FlowSolverParam *const flow_solver_param);
#endif
} // namespace PHiLiP::Grids