#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "positivity_preserving_tests_grid.h"

namespace PHiLiP::Grids {
template<int dim, typename TriangulationType>
void shock_tube_1D_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input)
{
    double xmax = parameters_input->flow_solver_param.grid_xmax;
    double xmin = parameters_input->flow_solver_param.grid_xmin;
    unsigned int n_subdivisions_x = parameters_input->flow_solver_param.number_of_grid_elements_x;

    dealii::GridGenerator::subdivided_hyper_cube(grid, n_subdivisions_x, xmin, xmax, true);

    int left_boundary_id = 9999;
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case_type = parameters_input->flow_solver_param.flow_case_type;

    if (flow_case_type == flow_case_enum::sod_shock_tube
        || flow_case_type == flow_case_enum::leblanc_shock_tube) {
        left_boundary_id = 1001;
    } else if (flow_case_type == flow_case_enum::shu_osher_problem) {
        left_boundary_id = 1007;
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
void nonsmooth_case_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input)
{
    double xmax = parameters_input->flow_solver_param.grid_xmax;
    double xmin = parameters_input->flow_solver_param.grid_xmin;
    unsigned int n_subdivisions_x = parameters_input->flow_solver_param.number_of_grid_elements_x;

    dealii::GridGenerator::subdivided_hyper_cube(grid, n_subdivisions_x, xmin, xmax, true);

    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face = 0; face < dealii::GeometryInfo<PHILIP_DIM>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                cell->face(face)->set_boundary_id(1000);
            }
        }
    }
}

template<int dim, typename TriangulationType>
void explosion_problem_grid(
    TriangulationType& grid,
    const Parameters::AllParameters* const parameters_input)
{
    double xmax = parameters_input->flow_solver_param.grid_xmax;
    double xmin = parameters_input->flow_solver_param.grid_xmin;
    double ymax = parameters_input->flow_solver_param.grid_ymax;
    double ymin = parameters_input->flow_solver_param.grid_ymin;
    double zmax = parameters_input->flow_solver_param.grid_zmax;
    double zmin = parameters_input->flow_solver_param.grid_zmin;

    unsigned int n_subdivisions_x = parameters_input->flow_solver_param.number_of_grid_elements_x;
    unsigned int n_subdivisions_y = parameters_input->flow_solver_param.number_of_grid_elements_y;
    unsigned int n_subdivisions_z = parameters_input->flow_solver_param.number_of_grid_elements_z;
    
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = xmin; p1[1] = ymin;
    p2[0] = xmax; p2[1] = ymax;

    if(dim == 3) {
        p1[2] = zmin;
        p2[2] = zmax;
    }
    
    std::vector<unsigned int> n_subdivisions(dim);

    n_subdivisions[0] = n_subdivisions_x;
    n_subdivisions[1] = n_subdivisions_y;

    if(dim == 3)
        n_subdivisions[2] = n_subdivisions_z;


    dealii::GridGenerator::subdivided_hyper_rectangle(grid, n_subdivisions, p1, p2, true);
    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face = 0; face < dealii::GeometryInfo<PHILIP_DIM>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                cell->face(face)->set_boundary_id(1001);
            }
        }
    }
}

template<int dim, typename TriangulationType>
void double_mach_reflection_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input) 
{
    double xmax = parameters_input->flow_solver_param.grid_xmax;
    double xmin = parameters_input->flow_solver_param.grid_xmin;
    double ymax = parameters_input->flow_solver_param.grid_ymax;
    double ymin = parameters_input->flow_solver_param.grid_ymin;

    unsigned int n_subdivisions_x = parameters_input->flow_solver_param.number_of_grid_elements_x;
    unsigned int n_subdivisions_y = parameters_input->flow_solver_param.number_of_grid_elements_y;
    
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = xmin; p1[1] = ymin;
    p2[0] = xmax; p2[1] = ymax;
    
    std::vector<unsigned int> n_subdivisions(2);

    // n_subdivisions[0] = n_subdivisions_x;//log2(128);
    // n_subdivisions[1] = n_subdivisions_y;//log2(64);

    const double uniform_spacing_x = (xmax-xmin)/n_subdivisions_x;
    std::vector<std::vector<double> > step_sizes(dim);
    // x-direction
    for (unsigned int i=0; i<n_subdivisions_x; i++) {
        step_sizes[0].push_back(uniform_spacing_x);
    }
    // y-direction
    double y_spacing = uniform_spacing_x;
    for (unsigned int j=0; j<n_subdivisions_y; j++) {
        if(j < n_subdivisions_x/2.0)
            step_sizes[1].push_back(uniform_spacing_x);
        else{
            y_spacing *= 2.0;
            step_sizes[1].push_back(y_spacing);
        }
    }

    dealii::GridGenerator::subdivided_hyper_rectangle(grid, step_sizes, p1, p2, true);

    double bottom_x = 0.0;
    double right_y = 0.0;

    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0) {
                    if(right_y<=2.0){
                        right_y += cell->extent_in_direction(1);
                        cell->face(face)->set_boundary_id(1007); // x_left, post-shock
                    }
                    else {
                        cell->face(face)->set_boundary_id(1001);
                    }
                }
                else if (current_id == 1) {
                    cell->face(face)->set_boundary_id(1004); // x_right, riemann
                }
                else if (current_id == 2) {
                    if (bottom_x < (1.0 / 6.0)) {
                        //std::cout << "assigning post shock " << bottom_x << std::endl;
                        bottom_x += cell->extent_in_direction(0);
                        cell->face(face)->set_boundary_id(1007); // y_bottom, post-shock
                    }
                    else {
                        cell->face(face)->set_boundary_id(1001); // y_bottom, wall
                    }
                }
                else if (current_id == 3) {
                    cell->face(face)->set_boundary_id(1004);
                }
            }
        }
    } 
}

template<int dim, typename TriangulationType>
void sedov_blast_wave_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input) 
{
    double xmax = parameters_input->flow_solver_param.grid_xmax;
    double xmin = parameters_input->flow_solver_param.grid_xmin;
    double ymax = parameters_input->flow_solver_param.grid_ymax;
    double ymin = parameters_input->flow_solver_param.grid_ymin;

    unsigned int n_subdivisions_x = parameters_input->flow_solver_param.number_of_grid_elements_x;
    unsigned int n_subdivisions_y = parameters_input->flow_solver_param.number_of_grid_elements_y;
    
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = xmin; p1[1] = ymin;
    p2[0] = xmax; p2[1] = ymax;
    
    std::vector<unsigned int> n_subdivisions(2);

    n_subdivisions[0] = n_subdivisions_x;//log2(128);
    n_subdivisions[1] = n_subdivisions_y;//log2(64);

    dealii::GridGenerator::subdivided_hyper_rectangle(grid, n_subdivisions, p1, p2, true);

    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0 || current_id == 2) {
                    cell->face(face)->set_boundary_id(1001); // x_left, post-shock
                } else {
                    cell->face(face)->set_boundary_id(1001);
                }
            }
        }
    }
}

template<int dim, typename TriangulationType>
void mach_3_wind_tunnel_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input) 
{
    double xmax = parameters_input->flow_solver_param.grid_xmax;
    double xmin = parameters_input->flow_solver_param.grid_xmin;
    double ymax = parameters_input->flow_solver_param.grid_ymax;
    double ymin = parameters_input->flow_solver_param.grid_ymin;

    unsigned int n_subdivisions_x = parameters_input->flow_solver_param.number_of_grid_elements_x;
    unsigned int n_subdivisions_y = parameters_input->flow_solver_param.number_of_grid_elements_y;
    
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = xmin; p1[1] = ymin;
    p2[0] = xmax; p2[1] = ymax;
    
    std::vector<unsigned int> n_subdivisions(2);
    n_subdivisions[0] = n_subdivisions_x;//log2(128);
    n_subdivisions[1] = n_subdivisions_y;//log2(64);

    std::vector<int> n_cells_remove(2);
    n_cells_remove[0] = (-2.4/3.0)*n_subdivisions[0] - 1;
    n_cells_remove[1] = (0.2/1.0)*n_subdivisions[1];

    dealii::GridGenerator::subdivided_hyper_L(grid, n_subdivisions, p1, p2, n_cells_remove);

    // Set boundary type and design type
    double right_y = 0.0;
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                if (face == 0) {
                    cell->face(face)->set_boundary_id(1007); // x_left, Inflow
                }
                else if (face == 1) {
                    if (right_y < 0.2) {
                        right_y += cell->extent_in_direction(1);
                        cell->face(face)->set_boundary_id(1001); // x_right, Symmetry/Wall 
                    }
                    else {
                        cell->face(face)->set_boundary_id(1009); // x_right, Outflow
                    }
                }
                else if (face == 2 || face == 3) {
                        cell->face(face)->set_boundary_id(1001); // y_top, y_bottom, Symmetry/Wall
                }
            }
        }
    }
}

template<int dim, typename TriangulationType>
void shock_diffraction_grid(
    TriangulationType&  grid,
    const Parameters::AllParameters *const parameters_input) 
{
    double xmax = parameters_input->flow_solver_param.grid_xmax;
    double xmin = parameters_input->flow_solver_param.grid_xmin;
    double ymax = parameters_input->flow_solver_param.grid_ymax;
    double ymin = parameters_input->flow_solver_param.grid_ymin;

    unsigned int n_subdivisions_x = parameters_input->flow_solver_param.number_of_grid_elements_x;
    unsigned int n_subdivisions_y = parameters_input->flow_solver_param.number_of_grid_elements_y;
    
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = xmin; p1[1] = ymin;
    p2[0] = xmax; p2[1] = ymax;
    
    std::vector<unsigned int> n_subdivisions(2);
    n_subdivisions[0] = n_subdivisions_x;//log2(128);
    n_subdivisions[1] = n_subdivisions_y;//log2(64);

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
                        cell->face(face)->set_boundary_id(1001); // y_bottom, Symmetry/Wall
                    }
                    else {
                        cell->face(face)->set_boundary_id(1007); // x_right, Symmetry/Wall
                    }
                }
                else if (face == 1) {
                    cell->face(face)->set_boundary_id(1002); // x_right, Symmetry/Wall
                }
                else if (face == 2) {
                    if (left_y >= 6.0 && bottom_x < 1.0) {
                        bottom_x += cell->extent_in_direction(1);
                        cell->face(face)->set_boundary_id(1001); // y_bottom, Symmetry/Wall
                    }
                    else {
                        cell->face(face)->set_boundary_id(1002); // x_right, Symmetry/Wall
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
    const Parameters::AllParameters *const parameters_input) 
{
    double xmax = parameters_input->flow_solver_param.grid_xmax;
    double xmin = parameters_input->flow_solver_param.grid_xmin;
    double ymax = parameters_input->flow_solver_param.grid_ymax;
    double ymin = parameters_input->flow_solver_param.grid_ymin;

    unsigned int n_subdivisions_x = parameters_input->flow_solver_param.number_of_grid_elements_x;
    unsigned int n_subdivisions_y = parameters_input->flow_solver_param.number_of_grid_elements_y;
    
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = xmin; p1[1] = ymin;
    p2[0] = xmax; p2[1] = ymax;
    
    std::vector<unsigned int> n_subdivisions(2);

    n_subdivisions[0] = n_subdivisions_x;//log2(128);
    n_subdivisions[1] = n_subdivisions_y;//log2(64);

    dealii::GridGenerator::subdivided_hyper_rectangle(grid, n_subdivisions, p1, p2, true);

    double left_y = 0.0;
    double dy = (ymax-ymin)/n_subdivisions_y;

    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0) {
                    //cell->face(face)->set_boundary_id(1007); // x_left, Farfield
                    if (left_y >= 0.45 && left_y <=0.55-dy) {
                        left_y += cell->extent_in_direction(1);
                        cell->face(face)->set_boundary_id(1007); // y_bottom, Symmetry/Wall
                    }
                    else {
                        left_y += cell->extent_in_direction(1);
                        cell->face(face)->set_boundary_id(1008);
                    }
                }
                else if (current_id == 1) {
                    cell->face(face)->set_boundary_id(1002); // x_right, Symmetry/Wall
                }
                else if (current_id == 2) {
                    cell->face(face)->set_boundary_id(1002); // y_bottom, Symmetry/Wall
                }
                else if (current_id == 3) {
                    cell->face(face)->set_boundary_id(1002);
                }
            }
        }
    }
}

#if PHILIP_DIM==1
template void shock_tube_1D_grid<1, dealii::Triangulation<1>>(
    dealii::Triangulation<1>&   grid,
    const Parameters::AllParameters *const parameters_input);
#else
template void explosion_problem_grid<PHILIP_DIM, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>(
    dealii::parallel::distributed::Triangulation<PHILIP_DIM>& grid,
    const Parameters::AllParameters* const parameters_input);
template void nonsmooth_case_grid<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>& grid,
    const Parameters::AllParameters* const parameters_input);
template void double_mach_reflection_grid<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    const Parameters::AllParameters *const parameters_input);
template void sedov_blast_wave_grid<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    const Parameters::AllParameters *const parameters_input);
template void mach_3_wind_tunnel_grid<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    const Parameters::AllParameters *const parameters_input);
template void shock_diffraction_grid<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    const Parameters::AllParameters *const parameters_input);
template void astrophysical_jet_grid<2, dealii::parallel::distributed::Triangulation<2>>(
    dealii::parallel::distributed::Triangulation<2>&    grid,
    const Parameters::AllParameters *const parameters_input);
#endif
} // namespace PHiLiP::Grids