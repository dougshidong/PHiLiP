#ifndef __LPCVT_MESH_ADAPTATION_SIMPLE_H__
#define __LPCVT_MESH_ADAPTATION_SIMPLE_H__

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

// Forward declarations
namespace PHiLiP {
    namespace FlowSolver {
        template <int dim, int nstate>
        class FlowSolver;
    }

    namespace GridRefinement {
        template <int dim, typename real>
        class Field;
    }
}

namespace PHiLiP {
    namespace Tests {

        template <int dim, int nstate>
        class LpCVTMeshAdaptationCases : public TestsBase
        {
        public:
            /// Constructor
            LpCVTMeshAdaptationCases(const Parameters::AllParameters* const parameters_input,
                const dealii::ParameterHandler& parameter_handler_input);

            /// Parameter handler
            const dealii::ParameterHandler& parameter_handler;

            /// Run test
            int run_test() const override;

            std::array<double,nstate> evaluate_soln_exact(const dealii::Point<dim> &point) const;

            /// Evaluates functional error
            double evaluate_functional_error(std::shared_ptr<DGBase<dim, double>> dg) const;

            /// Evaluates absolute DWR error
            double evaluate_abs_dwr_error(std::shared_ptr<DGBase<dim, double>> dg) const;

        private:
            /// Structure to hold extracted metric data
            struct MetricData {
                std::unique_ptr<PHiLiP::GridRefinement::Field<dim, double>> h_field;
                bool valid = false;
            };
            void evaluate_regularization_matrix(
                dealii::TrilinosWrappers::SparseMatrix &regularization_matrix, 
                std::shared_ptr<DGBase<dim,double>> dg) const;
            bool run_adaptation_loop(
                    std::unique_ptr<FlowSolver::FlowSolver<dim, nstate> > &flow_solver,
                    const Parameters::AllParameters &param,
                    unsigned int max_cycles,
                    std::vector<double> &functional_error_vector,
                    std::vector<unsigned int> n_dofs_vector,  
                    std::vector<unsigned int> &n_cycle_vector,
                    unsigned int outer_iter
                ) const;

            void extract_shock_nodes_from_msh(
                    const std::string& msh_filename,
                    const std::string& output_txt_filename,
                    const double x_tolerance,
                    const double min_distance_between_nodes) const;
            
            /// Extracts metric field from current mesh state
            MetricData extract_metric_field(
                const std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>>& flow_solver,
                const unsigned int outer_loop,
                const Parameters::AllParameters& param) const;

            /// Run LpCVT reconstruction
            void run_lpcvt_reconstruction(
                std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>>& flow_solver,
                const MetricData& metric_data) const;

            void write_lpcvt_background_mesh(
                const std::string& filename,
                const std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>>& flow_solver,
                const MetricData& metric_data) const;

            void write_dwr_cellwise_vtk(
                const std::unique_ptr<FlowSolver::FlowSolver<dim, nstate>>& flow_solver) const;

        };

    } // Tests namespace
} // PHiLiP namespace

#endif // __LPCVT_MESH_ADAPTATION_SIMPLE_H__