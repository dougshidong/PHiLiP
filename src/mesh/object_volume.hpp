#ifndef PHILIP_OBJECT_VOLUME_H
#define PHILIP_OBJECT_VOLUME_H

#include "mesh/high_order_grid.h"

#include "meshmover_linear_elasticity.hpp"

template <int dim, typename real>
class VolumeComputer
{
public:
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::Triangulation<dim>;
#else
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif

protected:

    template<typename real2>
    static real2 local_compute_volume(const HighOrderGrid<dim,real> &high_order_grid, const int boundary_wall_id) {
    }

public:

    template<typename real2>
    static real compute_volume(const HighOrderGrid<dim,real> &high_order_grid, const int boundary_wall_id) {

        // Decide on quadrature rule.
        const unsigned int degree = high_order_grid.max_degree;
        const unsigned int overintegration = 0;
        dealii::QGauss<dim-1> face_quadrature (degree+1+overintegration);

        for (; cell != this->triangulation->end(); ++cell)
            if (cell->is_locally_owned())
                for (const unsigned int f : dealii::GeometryInfo<dimension>::face_indices())
                    if (cell->face(f)->at_boundary())
                        if (cell->face(f)->boundary_id() == 1001) {
                            // Project quadrature onto the correct face.
                            const dealii::Quadrature<dim> face_quadrature_dim = dealii::QProjector<dim>::project_to_face( dealii::ReferenceCell::get_hypercube(dim), quadrature, f);
                            
                        }

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (!cell->is_locally_owned()) continue;

            for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
                if (current_face->at_boundary()
                    && current_face->boundary_id!current_cell->has_periodic_neighbor(iface) 
                    && !current_cell->has_periodic_neighbor(iface) ) {
                    

                }
            }
        }
    }


}

#endif

