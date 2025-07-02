#ifndef MESHES_INTERPOLATION_HPP
#define MESHES_INTERPOLATION_HPP

#include <memory>
#include <string>
#include <iostream>

// Forward declarations
#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"

namespace PHiLiP {

    template <int dim, int nstate, typename MeshType>
    class MeshInterpolation
    {
    public:
        /// Constructor
        explicit MeshInterpolation(std::ostream& out_stream);

        /// Destructor
        ~MeshInterpolation() = default;

        void perform_mesh_interpolation_test(
            const std::shared_ptr<DGBase<dim, double, MeshType>>& source_dg,
            const Parameters::AllParameters& param,
            const int poly_degree_interpolation,
            const std::string& target_mesh_file) const;

    private:
        /// Output stream
        std::ostream& out;
    };

} // namespace PHiLiP

#endif // MESHES_INTERPOLATION_HPP