#ifndef MESHES_INTERPOLATION_HPP
#define MESHES_INTERPOLATION_HPP

#include <memory>
#include <string>
#include <iostream>

#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP
{
    template <int dim>
    struct RemotePointQuery
    {
        dealii::Point<dim>                physical_point;
        dealii::types::global_dof_index target_dof_index;
        unsigned int                      component;
    };

    template <int dim>
    struct RemotePointResult
    {
        dealii::types::global_dof_index target_dof_index;
        double                          value;
    };


    template <int dim, int nstate, typename MeshType>
    class MeshInterpolation
    {
    public:
        explicit MeshInterpolation(std::ostream& out_stream);

        ~MeshInterpolation() = default;

        std::shared_ptr<DGBase<dim, double, MeshType>>
            perform_mesh_interpolation(
                const std::shared_ptr<DGBase<dim, double, MeshType>>& source_dg,
                const Parameters::AllParameters& param,
                const int poly_degree_interpolation,
                const std::string& target_mesh_file) const;

    private:
        /// An output stream for logging.
        std::ostream& out;
    };

} // namespace PHiLiP

#endif // MESHES_INTERPOLATION_HPP