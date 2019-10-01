
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>

#include <exception>
#include <deal.II/fe/mapping.h> 
#include <deal.II/base/exceptions.h> // ExcTransformationFailed

#include <deal.II/fe/mapping_fe_field.h> 
#include <deal.II/fe/mapping_q.h> 

#include "dg/high_order_grid.h"
#include "parameters/all_parameters.h"

// https://en.wikipedia.org/wiki/Volume_of_an_n-ball#Recursions
template <int dim>
double volume_n_ball(const double radius)
{
    const double pi = dealii::numbers::PI;
    return (2.0*pi*radius*radius/dim)*volume_n_ball<dim-2>(radius);
}
template <> double volume_n_ball<0>(const double /*radius*/) { return 1.0; }
template <> double volume_n_ball<1>(const double radius) { return 2.0*radius; }

namespace dealii{

template <int dim, int spacedim = dim>
class FEFieldManifold : public Manifold<dim, spacedim>
{
public:
    //using MappingFEField = MappingFEField<dim,dim,LinearAlgebra::distributed::Vector<double>, DoFHandler<dim>>

    //const MappingFEField mapping;
    //const Mapping<dim> &mapping;
    const MappingFEField<dim,spacedim,LinearAlgebra::distributed::Vector<double>, DoFHandler<dim>> mapping;
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    using myTriangulation = dealii::Triangulation<dim>;
#else
    using myTriangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif
    const myTriangulation *tria;

    FEFieldManifold(
        const MappingFEField<dim,dim,LinearAlgebra::distributed::Vector<double>, DoFHandler<dim>> &mapping_input,
        //const Mapping<dim> &mapping_input,
        const myTriangulation *tria_input)
        : mapping(mapping_input)
        , tria(tria_input)
    { };

  
    virtual std::unique_ptr<Manifold<dim, spacedim>>
    clone() const override
    { return std_cxx14::make_unique<FEFieldManifold<dim, spacedim>>(mapping, tria); };
  
    virtual Point<spacedim>
    get_intermediate_point(const Point<spacedim> &p1,
                           const Point<spacedim> &p2,
                           const double           w) const override
    {
        std::cout << "get_intermediate_point p1:" << p1 << " p2: " << p2 << " w: " << w << std::endl;
        //bool found_p1 = false;
        //bool found_p2 = false;
        //std::vector<Triangulation::active_cell_iterator> cells_touching_p1, cells_touching_p2;
        std::vector< dealii::TriaActiveIterator < dealii::CellAccessor<dim,spacedim> > > cells_touching_p1, cells_touching_p2;
        dealii::Tensor<1,dim,double> zero_dist;

        // Check if the vertex if within diameter_factor*diameter to the cell center.
        // If it is, then attempt to transform_real_to_unit
        const double inside_tolerance = 1e-12;
        const double diameter_factor2 = 2.0 * 2.0;
        for (auto cell = tria->begin_active(); cell!=tria->end(); ++cell) {
            if (!cell->is_locally_owned()) continue;

            const double diameter = cell->diameter();
            const double acceptable_distance2 = diameter_factor2 * diameter * diameter;

            // Cell center of straight-sided element
            dealii::Point<spacedim> cell_center = cell->center(false, false);
            for (unsigned int iv=0; iv<dealii::GeometryInfo<dim>::vertices_per_cell; ++iv) {
                if (cell_center.distance_square(p1) < acceptable_distance2) {
                    bool is_inside = false;
                    try {
                        const dealii::Point<dim> ref_vertex = mapping.transform_real_to_unit_cell(cell, p1);
                        is_inside = GeometryInfo<dim>::is_inside_unit_cell(ref_vertex, inside_tolerance);
                    //} catch (const Mapping<dim, spacedim>::ExcTransformationFailed &) {
                    } catch (...) {
                    }
                    if (is_inside) cells_touching_p1.push_back(cell);
                }
                if (cell_center.distance_square(p2) < acceptable_distance2) {
                    bool is_inside = false;
                    try {
                        const dealii::Point<dim> ref_vertex = mapping.transform_real_to_unit_cell(cell, p2);
                        is_inside = GeometryInfo<dim>::is_inside_unit_cell(ref_vertex, inside_tolerance);
                    //} catch (const Mapping<dim, spacedim>::ExcTransformationFailed &) {
                    } catch (...) {
                    }
                    if (is_inside) cells_touching_p2.push_back(cell);
                }
            }
        }
        // Find the intersection of cells touching p1 and p2
        std::vector<dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>>> common_cells(cells_touching_p1.size());
        std::set_intersection(cells_touching_p1.begin(), cells_touching_p1.end(),
                              cells_touching_p2.begin(), cells_touching_p2.end(),
                              common_cells.begin());
        std::cout << std::endl;
        std::cout << "Cells touching p1 & p2 "<< std::endl;
        for (auto cell = common_cells.begin(); cell != common_cells.end(); ++cell) {
            std::cout << (*cell)->user_index() << std::endl;
        }
        // Use first cell in the line to evaluate the mapping
        //const dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> physical_cell = *(common_cells.begin());
        //const dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> physical_cell = common_cells[0];

        // Bring p1 and p2 back to reference coordinate and get the midpoint
        const dealii::Point<dim> p1_ref = mapping.transform_real_to_unit_cell(common_cells[0], p1);
        const dealii::Point<dim> p2_ref = mapping.transform_real_to_unit_cell(common_cells[0], p2);

        const dealii::Point<dim> intermediate_point_ref = (1.0-w) * p1_ref + w * p2_ref;

        const dealii::Point<dim> intermediate_point = mapping.transform_unit_to_real_cell(common_cells[0], intermediate_point_ref);

        std::cout << "New point located at: ";
        for (int d=0;d<dim;d++) {
            std::cout<<intermediate_point[d] << " ";
        }
        std::cout << std::endl;

        return intermediate_point;
    };
  
    //virtual Point< spacedim > get_intermediate_point (const Point< spacedim > &p1, const Point< spacedim > &p2, const double w) const override
    //{
    //};

    //virtual Point< spacedim > get_new_point (const ArrayView< const Point< spacedim >> &surrounding_points, const ArrayView< const double > &weights) const

    //virtual void get_new_points (const ArrayView< const Point< spacedim >> &surrounding_points, const Table< 2, double > &weights, ArrayView< Point< spacedim >> new_points) const

    //virtual Point< spacedim > project_to_manifold (const ArrayView< const Point< spacedim >> &surrounding_points, const Point< spacedim > &candidate) const

    virtual Point< spacedim > get_new_point_on_line (const typename Triangulation< dim, spacedim >::line_iterator &line) const override {
        std::cout << "get_new_point_on_line" << std::endl;
        const int n_points_to_find = 2;
        std::array <dealii::Point<spacedim>, n_points_to_find> points_to_find;
        for (int ip=0; ip<n_points_to_find; ++ip) {
            points_to_find[ip] = line->vertex(ip);
            std::cout << "Looking for Point " << ip+1 << ": " << points_to_find[ip] << std::endl;
        }

        const dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> cell_iterator = find_a_cell_given_vertices<n_points_to_find>(points_to_find);
        std::cout << "Found cell: " << cell_iterator << " with vertices: ";
        for (int ip=0; ip<n_points_to_find; ++ip) {
            std::cout << "Vertex " << ip+1 << ": " << cell_iterator->vertex(ip);
        }
        std::cout << std::endl;

        // Bring the points back to reference coordinate and get the midpoint
        dealii::Point<spacedim> new_reference_point;
        for (int ip=0; ip<n_points_to_find; ++ip) {
            new_reference_point += mapping.transform_real_to_unit_cell(cell_iterator, points_to_find[ip]);
        }
        new_reference_point /= n_points_to_find;

        const dealii::Point<dim> new_physical_point = mapping.transform_unit_to_real_cell(cell_iterator, new_reference_point);

        std::cout << "New point located at: ";
        for (int d=0;d<dim;d++) {
            std::cout<<new_physical_point[d] << " ";
        }
        std::cout << std::endl;

        return new_physical_point;
    };

    virtual Point< spacedim > get_new_point_on_quad (const typename Triangulation< dim, spacedim >::quad_iterator &quad) const override
    {
        std::cout << "get_new_point_on_quad" << std::endl;
        const int n_points_to_find = 4;
        std::array <dealii::Point<spacedim>, n_points_to_find> points_to_find;
        for (int ip=0; ip<n_points_to_find; ++ip) {
            points_to_find[ip] = quad->vertex(ip);
            std::cout << "Looking for Point " << ip+1 << ": " << points_to_find[ip] << std::endl;
        }

        const dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> cell_iterator = find_a_cell_given_vertices<n_points_to_find>(points_to_find);
        std::cout << "Found cell: " << cell_iterator << " with vertices: ";
        for (int ip=0; ip<n_points_to_find; ++ip) {
            std::cout << "Vertex " << ip+1 << ": ";
            for (int d=0;d<dim;d++) {
                std::cout<<cell_iterator->vertex(ip) << " ";
            }
        }
        std::cout << std::endl;

        // Bring the points back to reference coordinate and get the midpoint
        dealii::Point<spacedim> new_reference_point;
        for (int ip=0; ip<n_points_to_find; ++ip) {
            new_reference_point += mapping.transform_real_to_unit_cell(cell_iterator, points_to_find[ip]);
        }
        new_reference_point /= n_points_to_find;
        std::cout << "New reference point located at: " << new_reference_point << std::endl;

        const dealii::Point<dim> new_physical_point = mapping.transform_unit_to_real_cell(cell_iterator, new_reference_point);
        std::cout << "New physical point located at: " << new_physical_point << std::endl;

        return new_physical_point;
    };

    //virtual Point< spacedim > get_new_point_on_hex (const typename Triangulation< dim, spacedim >::hex_iterator &hex) const override
    //{
    //};
private:
    /** Given some points, this subroutine will return a cell iterator that has vertices at those locations.
     *  There may be many cells that fit the description if given an edge for example. In that case, the 
     *  first cell found to fit the criteria will be returned.
     */
    template<int n_points_to_find>
    dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> find_a_cell_given_vertices(
            const std::array< dealii::Point<spacedim>, n_points_to_find > &points_to_find) const
    {
        // Each array entry contains the list of cells touching vertex 1,2,..,(n_points_to_find) correspondingly
        std::array < std::vector< dealii::TriaActiveIterator < dealii::CellAccessor<dim,spacedim> > >
                        , n_points_to_find >
                    cells_touching_vertices;

        const dealii::Tensor<1,dim,double> zero_dist;
        for (auto cell = tria->begin_active(); cell!=tria->end(); ++cell) {
            if (!cell->is_locally_owned()) continue;
            for (unsigned int iv=0; iv<dealii::GeometryInfo<dim>::vertices_per_cell; ++iv) {
                const dealii::Point<dim> vertex = cell->vertex(iv);

                for (int ip=0; ip<n_points_to_find; ++ip) {
                    const dealii::Tensor<1,dim,double> distance = vertex - points_to_find[ip];
                    if(distance == zero_dist) {
                        cells_touching_vertices[ip].push_back(cell);
                    }
                }
            }
        }
        // Find the intersection of cells touching all the points.
        std::vector<dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>>> common_cells(cells_touching_vertices[0].size());
        for (int iv=0; iv<n_points_to_find-1; ++iv) {
            std::set_intersection(
                cells_touching_vertices[iv].begin(), cells_touching_vertices[iv].end(),
                cells_touching_vertices[iv+1].begin(), cells_touching_vertices[iv+1].end(),
                common_cells.begin());
        }
        // Use first cell in the line to evaluate the mapping
        const dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> cell = common_cells[0];

        return cell;
    }
    
};

}
 
int main (int argc, char * argv[])
{
    const int dim = PHILIP_DIM;
    int fail_bool = false;

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    using namespace PHiLiP;

    dealii::ParameterHandler parameter_handler;
    Parameters::AllParameters::declare_parameters (parameter_handler);
    Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);

    std::vector<int> fail_conv_poly;
    std::vector<double> fail_conv_slop;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    const unsigned int p_start = 1;
    const unsigned int p_end   = 4;
    const unsigned int n_grids = 3;
    //const std::vector<int> n_1d_cells = {2,4,8,16};

    //const unsigned int n_cells_circle = n_1d_cells[0];
    //const unsigned int n_cells_radial = 3*n_cells_circle;

    dealii::Point<dim> center; // Constructor initializes with 0;
    const double inner_radius = 1, outer_radius = inner_radius*10;
    const double exact_volume = (volume_n_ball<dim>(outer_radius) - volume_n_ball<dim>(inner_radius)) / std::pow(2,dim);

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {

        std::vector<double> grid_size(n_grids);

        dealii::ConvergenceTable convergence_table;

        // Generate grid and mapping
        dealii::parallel::distributed::Triangulation<dim> grid(
            MPI_COMM_WORLD,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));

        //dealii::Triangulation<dim> grid;

        const int n_cells = 0;//1*(dim-1);
        dealii::GridGenerator::quarter_hyper_shell<dim>(grid, center, inner_radius, outer_radius, n_cells);//, n_cells = 0, colorize = false);
        //dealii::GridGenerator::hyper_shell<dim>(grid, center, inner_radius, outer_radius, n_cells);//, n_cells = 0, colorize = false);
        grid.set_all_manifold_ids(0);
        grid.set_manifold(0, dealii::SphericalManifold<dim>(center));

        std::vector<double> volume_error(n_grids);

        HighOrderGrid<dim,double> high_order_grid(&all_parameters, poly_degree, &grid);
        dealii::MappingFEField<dim,dim,dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> mapping = high_order_grid.get_MappingFEField();
        grid.reset_all_manifolds();
        for (unsigned int igrid=0; igrid<n_grids; ++igrid) {

            //dealii::parallel::distributed::Triangulation<dim> grid_copy(
            //    MPI_COMM_WORLD,
            //    typename dealii::Triangulation<dim>::MeshSmoothing(
            //        dealii::Triangulation<dim>::smoothing_on_refinement |
            //        dealii::Triangulation<dim>::smoothing_on_coarsening));
            //grid_copy.copy_triangulation(grid);

            //dealii::MappingFEField<dim,dim,dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> mapping = high_order_grid.get_MappingFEField();
            //const dealii::FEFieldManifold<dim,dim> fefield_manifold(mapping, &grid);
            //grid.set_manifold(0, fefield_manifold);

            // Refine the grid globally once
            //if (igrid >= 1) grid.refine_global (1);

            //for (auto cell = grid.begin_active(); cell!=grid.end(); ++cell) {
            //    for (unsigned int iv=0; iv<dealii::GeometryInfo<dim>::vertices_per_cell; ++iv) {
            //        const dealii::Point<dim> phys_vertex = cell->vertex(iv);
            //        const dealii::Point<dim> ref_vertex = mapping.transform_real_to_unit_cell(cell, phys_vertex);

            //        std::cout << "vertex #" << iv << std::endl;
            //        std::cout << phys_vertex[0] << " " << phys_vertex[1] << std::endl;
            //        std::cout << ref_vertex[0] << " " << ref_vertex[1] << std::endl;

            //    }
            //}

            dealii::LinearAlgebra::distributed::Vector<double> old_nodes(high_order_grid.nodes);
            old_nodes.update_ghost_values();
            dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(high_order_grid.dof_handler_grid);
            solution_transfer.prepare_for_coarsening_and_refinement(old_nodes);

            grid.refine_global (1);
            high_order_grid.allocate();
            solution_transfer.interpolate(high_order_grid.nodes);

            const unsigned int n_dofs = high_order_grid.dof_handler_grid.n_dofs();

            const unsigned int n_global_active_cells = grid.n_global_active_cells();

            //const bool use_mapping_q_on_all_cells = true;
            //dealii::MappingQ<dim,dim> mapping(poly_degree, use_mapping_q_on_all_cells);
            //dealii::Mapping<dim> mapping = high_order_grid.get_MappingFEField();


            std::ofstream out_before("before_move_grid-" + std::to_string(poly_degree) + "-" + std::to_string(igrid) + ".eps");
            dealii::GridOut grid_out_before;
            dealii::GridOut::OutputFormat out_format = dealii::GridOut::OutputFormat::eps;

            const dealii::Mapping<dim,dim> *const base_mapping = &mapping;

            dealii::DataOut<dim> data_out;
            data_out.attach_dof_handler(high_order_grid.dof_handler_grid);
            std::vector<std::string> solution_names;
            switch (dim)
            {
              case 1:
                solution_names.emplace_back("x");
                break;
              case 2:
                solution_names.emplace_back("x");
                solution_names.emplace_back("y");
                break;
              case 3:
                solution_names.emplace_back("x");
                solution_names.emplace_back("y");
                solution_names.emplace_back("z");
                break;
              default:
                Assert(false, dealii::ExcNotImplemented());
            }
            data_out.add_data_vector(high_order_grid.nodes, solution_names);
            std::ofstream output("grid-" + std::to_string(poly_degree) + "-" + std::to_string(igrid) + ".vtk");
            data_out.build_patches(mapping, poly_degree+1, dealii::DataOut<dim>::CurvedCellRegion::curved_inner_cells);
            data_out.write_vtk(output);



            grid_out_before.write(grid, out_before, out_format, base_mapping);
            //grid_out_before.write(grid, out_before, out_format);
            //for (auto cell = high_order_grid.dof_handler_grid.begin_active(); cell!=high_order_grid.dof_handler_grid.end(); ++cell) {

            //    if (!cell->is_locally_owned()) continue;
            //    const unsigned int fe_index_cell = cell->active_fe_index();
            //    const dealii::FESystem<dim,dim> &fe_ref = high_order_grid.fe_system[fe_index_cell];
            //    const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();

            //    // Obtain the mapping from local dof indices to global dof indices
            //    std::vector<dealii::types::global_dof_index> dofs_indices;
            //    dofs_indices.resize(n_dofs_cell);
            //    cell->get_dof_indices (dofs_indices);

            //    for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            //        //const unsigned int idir_test = fe_values_volume.get_fe().system_to_component_index(idof).first;
            //        // Translate the grid
            //        high_order_grid.nodes[dofs_indices[idof]] += 1.0;
            //    }
            //}
            for (auto dof = high_order_grid.nodes.begin(); dof != high_order_grid.nodes.end(); ++dof) {
                *dof += 1.0;
            }
            high_order_grid.nodes.update_ghost_values();
            //dealii::GridTools::transform(
            //[](const dealii::Point<dim> &old_point) -> dealii::Point<dim> {
            //    dealii::Point<dim> new_point;
            //    for (int d=0;d<dim;++d) {
            //        new_point[d] = old_point[d] + 1.0;
            //    }
            //    return new_point;
            //},
            //grid);

            //for (int d=0;d<dim;++d) {
            //    center[d] += 1.0;
            //}
            //grid.set_manifold(0, dealii::SphericalManifold<dim>(center));

            double volume = 0;
            int n_cell = 2;
            // Integrate solution error and output error
            // Overintegrate the error to make sure there is not integration error in the error estimate
            const int overintegrate = 2;
            const unsigned int n_quad_pts_1D = poly_degree+1+overintegrate;
            dealii::QGauss<dim> quadrature(n_quad_pts_1D);
            const unsigned int n_quad_pts = quadrature.size();
            dealii::FEValues<dim,dim> fe_values(mapping, high_order_grid.fe_system, quadrature,
                dealii::update_jacobians
                //| dealii::update_volume_elements
                | dealii::update_JxW_values
                | dealii::update_quadrature_points);
            //for (auto cell = grid.begin_active(); cell!=grid.end(); ++cell) {
            for (auto cell = high_order_grid.dof_handler_grid.begin_active(); cell!=high_order_grid.dof_handler_grid.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;
                n_cell++;

                fe_values.reinit (cell);

                double cell_volume = 0.0;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    cell_volume += fe_values.JxW(iquad);
                }
                volume += cell_volume;
            }
            const double volume_mpi_sum = dealii::Utilities::MPI::sum(volume, MPI_COMM_WORLD);
            std::ofstream out("grid-" + std::to_string(poly_degree) + "-" + std::to_string(igrid) + ".eps");
            dealii::GridOut grid_out;
            grid_out.write(grid, out, out_format, base_mapping);
            //grid_out.write(grid, out, out_format);

            // Convergence table
            const double dx = 1.0/pow(n_dofs,(1.0/dim));
            grid_size[igrid] = dx;
            volume_error[igrid] = std::abs(1.0 - volume_mpi_sum/exact_volume);

            pcout << "P = " << poly_degree << " NCells = " << n_cell << " Estimated volume: " << volume_mpi_sum << " Exact Volume: " << exact_volume << std::endl;

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("volume_error", volume_error[igrid]);

        }
        pcout << " ********************************************" << std::endl
             << " Convergence rates for p = " << poly_degree << std::endl
             << " ********************************************" << std::endl;
        convergence_table.evaluate_convergence_rates("volume_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("volume_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());

        convergence_table_vector.push_back(convergence_table);

        const double expected_slope = 2*poly_degree;

        const double last_slope = log(volume_error[n_grids-1]/volume_error[n_grids-2])
                                  / log(grid_size[n_grids-1]/grid_size[n_grids-2]);
        double before_last_slope = last_slope;
        if ( n_grids > 2 ) {
        before_last_slope = log(volume_error[n_grids-2]/volume_error[n_grids-3])
                            / log(grid_size[n_grids-2]/grid_size[n_grids-3]);
        }
        const double slope_avg = 0.5*(before_last_slope+last_slope);
        const double slope_diff = slope_avg-expected_slope;

        const double slope_deficit_tolerance = -0.1;//-std::abs(manu_grid_conv_param.slope_deficit_tolerance);

        if (slope_diff < slope_deficit_tolerance) {
            pcout << std::endl << "Convergence order not achieved. Average last 2 slopes of " << slope_avg << " instead of expected "
                 << expected_slope << " within a tolerance of " << slope_deficit_tolerance << std::endl;
            if(poly_degree!=0) fail_conv_poly.push_back(poly_degree);
            if(poly_degree!=0) fail_conv_slop.push_back(slope_avg);
        }

    }
    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }
    int n_fail_poly = fail_conv_poly.size();
    if (n_fail_poly > 0) {
        for (int ifail=0; ifail < n_fail_poly; ++ifail) {
            const double expected_slope = fail_conv_poly[ifail]+1;
            const double slope_deficit_tolerance = -0.1;
            pcout << std::endl << "Convergence order not achieved for polynomial p = " << fail_conv_poly[ifail]
                 << ". Slope of " << fail_conv_slop[ifail] << " instead of expected " << expected_slope
                 << " within a tolerance of " << slope_deficit_tolerance << std::endl;
        }

        fail_bool = true;
    }
    return fail_bool;
}
