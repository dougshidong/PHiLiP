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

#include <exception>
#include <deal.II/fe/mapping.h> 
#include <deal.II/base/exceptions.h> // ExcTransformationFailed

#include <deal.II/fe/mapping_fe_field.h> 
#include <deal.II/fe/mapping_q.h> 

template <int dim, int spacedim = dim>
class FEFieldManifold : public Manifold<dim, spacedim>
{
public:

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
                    } catch (...) {
                        // Most likely a ExcTransformationFailed
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

        // Bring p1 and p2 back to reference coordinate and get the midpoint
        const dealii::Point<dim> p1_ref = mapping.transform_real_to_unit_cell(common_cells[0], p1);
        const dealii::Point<dim> p2_ref = mapping.transform_real_to_unit_cell(common_cells[0], p2);

        const dealii::Point<dim> intermediate_point_ref = (1.0-w) * p1_ref + w * p2_ref;

        const dealii::Point<dim> intermediate_point = mapping.transform_unit_to_real_cell(common_cells[0], intermediate_point_ref);

        return intermediate_point;
    };
};
